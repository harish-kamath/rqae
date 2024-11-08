import modal

app = modal.App("rqae-upload-gemmascope-features")
volume = modal.Volume.from_name("rqae-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.2.1", gpu="t4")
    .pip_install("scipy==1.11.4")
    .pip_install("modal==0.64.10")
    .pip_install("huggingface-hub==0.24.5")
    .pip_install("safetensors==0.4.2")
    .pip_install("transformers==4.44.0")
    .pip_install("tqdm")
    .pip_install("anthropic")
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from rqae.gemmascope import JumpReLUSAE
    from rqae.feature import Feature, RQAEFeature
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import numpy as np


@app.function(
    volumes={"/data": volume},
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
)
def pearson_by_cos_sim(
    feature_id: int,
    model_id: str,
    dataset: str,
):
    from tqdm import tqdm
    import json
    import torch
    import numpy as np
    from rqae.feature import RQAEFeature
    from rqae.model import RQAE
    import gc
    from scipy import stats

    # Take the feature
    feature = RQAEFeature.load(
        f"/data/datasets/{dataset}/features/{model_id}/{feature_id:06d}.npz"
    )
    rqae = RQAE.from_pretrained(model_id)
    feature.load_model(rqae)
    # Get the original token (top activating example)
    top_activating_example = feature.activations
    top_activating_example = sorted(
        top_activating_example[1023], key=lambda x: x["activations"].max(), reverse=True
    )[0]
    # {"text": [str], "activations": [float]}

    # Find the top activation sample location in the dataset
    with open(f"/data/datasets/{dataset}/text.json", "r") as f:
        texts = json.load(f)
    top_s, top_t = None, None
    for s in range(len(texts)):
        if texts[s] == top_activating_example["text"]:
            top_s = s
            top_t = top_activating_example["activations"].argmax()
            break

    # Save the activation of the top activating example
    shard = torch.load(
        f"/data/datasets/{dataset}/activations/raw/{top_s // 1024:06d}.pt"
    )
    top_activating_example_activation = shard[top_s % 1024, top_t]
    del shard
    gc.collect()

    # Get the locations of the top 128 activations by cosine similarity to the top activating example
    # Also, save the cosine similarities to the top activating example
    cosine_similarities = torch.ones((36864, 127))
    for shard_idx in tqdm(range(36), desc="Calculating cosine similarities"):
        shard = torch.load(
            f"/data/datasets/{dataset}/activations/raw/{shard_idx:06d}.pt"
        )  # 1024 x 128 x 2304
        for i in range(shard.shape[0]):
            for j in range(1, shard.shape[1]):
                cosine_similarities[i + shard_idx * 1024, j - 1] = (
                    torch.cosine_similarity(
                        top_activating_example_activation, shard[i, j], dim=0
                    )
                )

    # Get the RQAE indices of each of the top 128 activations
    top_128_indices = torch.tensor(
        np.unravel_index(
            torch.topk(cosine_similarities.flatten(), 128).indices,
            cosine_similarities.shape,
        )
    ).T
    rqae_indices = [None for _ in range(top_128_indices.shape[0])]
    for shard_idx in tqdm(range(36), desc="Getting RQAE indices"):
        shard = torch.load(
            f"/data/datasets/{dataset}/activations/{model_id}/{shard_idx:06d}.pt"
        )
        for idx_idx, idx in enumerate(top_128_indices):
            if shard_idx == idx[0] // 1024:
                rqae_indices[idx_idx] = shard[idx[0] % 1024, idx[1] + 1]
    rqae_indices = torch.stack(rqae_indices)
    # 128 x 1024

    # Get the layerwise intensity to the feature
    intensities = feature.intensity(rqae_indices, layers=list(range(1024)))

    # Get the actual top cosine similarities
    top_cosine_similarities = cosine_similarities[
        top_128_indices[:, 0], top_128_indices[:, 1]
    ]
    # Get layerwise Pearson correlation
    pearson_corrs = [
        stats.pearsonr(top_cosine_similarities.numpy(), intensities[:, i].numpy())[0]
        for i in range(intensities.shape[1])
    ]
    spearman_corrs = [
        stats.spearmanr(top_cosine_similarities.numpy(), intensities[:, i].numpy())[0]
        for i in range(intensities.shape[1])
    ]

    return pearson_corrs, spearman_corrs


@app.local_entrypoint()
def main():
    futures = []

    for i in range(75):
        futures.append(
            pearson_by_cos_sim.spawn(
                feature_id=i,
                model_id="rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
                dataset="monology_pile",
            )
        )

    pearson_corrs, spearman_corrs = [], []
    for future in futures:
        try:
            pearson_corr, spearman_corr = future.get()
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
        except Exception as e:
            print(e)

    import torch

    pearson_corrs = torch.tensor(pearson_corrs)
    spearman_corrs = torch.tensor(spearman_corrs)
    torch.save(pearson_corrs, "pearson_corrs.pt")
    torch.save(spearman_corrs, "spearman_corrs.pt")


if __name__ == "__main__":
    from scripts.plotting_utils import *
    import torch

    pearson_corrs = torch.load("pearson_corrs.pt")
    spearman_corrs = torch.load("spearman_corrs.pt")

    color = create_blog_colormaps()["primary_accent"]
    color = color(0.9)

    plot_array = spearman_corrs
    plt.plot(range(plot_array.shape[1]), plot_array.mean(dim=0), color=color)
    plt.fill_between(
        range(plot_array.shape[1]),
        plot_array.mean(dim=0) - plot_array.std(dim=0),
        plot_array.mean(dim=0) + plot_array.std(dim=0),
        alpha=0.2,
        color=color,
    )

    plt.xlabel("Layer Number")
    plt.ylabel("Spearman Correlation")
    plt.show()
    # plt.savefig("spearman_corrs.jpg", dpi=300, bbox_inches="tight")
