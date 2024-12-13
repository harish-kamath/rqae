import modal

app = modal.App("rqae-upload-gemmascope-features")
volume = modal.Volume.from_name("rqae-volume", create_if_missing=True)


def download_gemma():
    from huggingface_hub import snapshot_download

    snapshot_download("google/gemma-2-2b")


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
    .run_function(
        download_gemma, secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from rqae.gemmascope import JumpReLUSAE
    from rqae.feature import Feature
    from transformers import AutoTokenizer
    from tqdm import tqdm


def front_weighted_sample(lst, k):
    """
    Sample k elements from a range, and return the indices.
    Parameters:
        lst: Input list
        k: Number of samples to take
    Returns:
        List of k sampled elements
    """
    import numpy as np

    if k > len(lst):
        return lst
    x = np.linspace(0.01, 0.99, k)
    # Quadratic spacing
    values = (1 - x * x) * (lst.max() - lst.min()) + lst.min()
    # Linear spacing
    # values = (1 - x) * (lst.max() - lst.min()) + lst.min()
    # Find the closest values in lst
    # indices = np.searchsorted(
    #     lst, values, sorter=np.array(list(reversed(range(len(lst)))))
    # )

    # Top k
    # indices = list(range(k))
    # Half top k, half middle k
    indices = list(range(k // 2)) + list(
        range(len(lst) // 2 - k // 2, len(lst) // 2 + k // 2)
    )
    return indices


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("anthropic")],
)
def spearman(
    feature_index: int,
    num_layers: int = 4,
    num_samples: int = 30,
    gemmascope_model_id: str = "gemmascope-gemma-2-2b-res-12-w16k-l82",
    rqae_model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
):
    import gc
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import json
    import random

    print(f"Calculating feature {feature_index} with {num_layers} layers")

    with open("/data/datasets/monology_pile/text.json", "r") as f:
        text = json.load(f)

    gemmascope_activations = []
    gemmascope_indices = []
    for batch_idx in tqdm(range(36), desc="Loading gemmascope activations"):
        gemma_activations = f"/data/datasets/monology_pile/activations/{gemmascope_model_id}/{batch_idx:06d}.pt"
        gemma_indices = f"/data/datasets/monology_pile/activations/{gemmascope_model_id}/{batch_idx:06d}_indices.pt"

        gemma_activations = torch.load(gemma_activations)
        gemma_indices = torch.load(gemma_indices)

        gemma_activations = gemma_activations[gemma_indices[:, 2] == feature_index]
        gemma_indices = gemma_indices[gemma_indices[:, 2] == feature_index][:, :2]
        gemma_indices[:, 0] += batch_idx * 1024

        gemmascope_activations.append(gemma_activations.clone())  # num_tokens
        gemmascope_indices.append(gemma_indices.clone())  # num_tokens, 2

        del gemma_activations, gemma_indices
        gc.collect()

    gemmascope_activations = torch.cat(gemmascope_activations, dim=0)
    gemmascope_indices = torch.cat(gemmascope_indices, dim=0)

    # Now, we have the full distribution of the feature
    # Let's reorder, and sample across the distribution (not including zero activations)
    order = gemmascope_activations.argsort(descending=True)
    gemmascope_indices = gemmascope_indices[order]
    gemmascope_activations = gemmascope_activations[order]

    print(f"Top Activations:")
    for i in range(10):
        print(
            gemmascope_activations[i].item(),
            gemmascope_indices[i].tolist(),
            text[gemmascope_indices[i][0]][gemmascope_indices[i][1]],
        )

    # Get indices that are spread across the distribution
    num_samples = min(num_samples, gemmascope_activations.shape[0])
    sampled_indices = front_weighted_sample(
        gemmascope_activations.clone().numpy(), num_samples // 3 * 2
    )
    # Add back 10 zero activations
    zero_sequences = [k for k in range(36864) if k not in gemmascope_indices[:, 0]]
    if len(zero_sequences) < num_samples // 3:
        print(
            f"Not enough zero sequences, {len(zero_sequences)}, will choose from bottom"
        )
        random_indices = []
        while len(random_indices) < num_samples // 3:
            random_s = random.sample(range(36864), 1)[0]
            random_t = random.randint(1, 127)
            random_token = [random_s, random_t]
            if not any([random_token == s.tolist() for s in gemmascope_indices]):
                random_indices.append(random_token)
    else:
        random_indices = [
            [s, random.randint(1, 127)]
            for s in random.sample(zero_sequences, num_samples // 3)
        ]

    gemmascope_activations = gemmascope_activations[sampled_indices]
    gemmascope_indices = gemmascope_indices[sampled_indices]

    # Get the max activating token index
    # max_activating_token = gemmascope_indices[0].clone()

    # Add back the zero activations
    gemmascope_indices = torch.cat([gemmascope_indices, torch.tensor(random_indices)])
    gemmascope_activations = torch.cat(
        [gemmascope_activations, torch.zeros(num_samples // 3)]
    )

    # print(
    #     f"Max Activating Token: {max_activating_token.tolist()} ({text[max_activating_token[0]][max_activating_token[1]]}) with activation {gemmascope_activations[-1].item()}"
    # )
    # max_token_layers = torch.load(
    #     f"/data/datasets/monology_pile/activations/{rqae_model_id}/{max_activating_token[0] // 1024:06d}.pt"
    # )[max_activating_token[0] % 1024, max_activating_token[1], :num_layers].to(
    #     torch.int32
    # )

    all_layers = torch.zeros(gemmascope_indices.shape[0], 1024, dtype=torch.int32)
    for batch_idx in tqdm(range(36), desc="Loading RQAE activations"):
        if not any(gemmascope_indices[:, 0] // 1024 == batch_idx):
            continue
        batch_indices = torch.load(
            f"/data/datasets/monology_pile/activations/{rqae_model_id}/{batch_idx:06d}.pt"
        )
        for sample_idx in range(gemmascope_indices.shape[0]):
            index = gemmascope_indices[sample_idx]
            if index[0] // 1024 != batch_idx:
                continue
            all_layers[sample_idx] = batch_indices[index[0] % 1024, index[1]]
    # Now, get the values at the same indices, for an RQAE feature centered around the max activating token
    model = RQAE.from_pretrained("google/gemma-2-2b")
    model.eval()
    codebook = model.codebook.data.detach().clone()[0]  # 1024 x 625 x 4 => 625 x 4
    layer_norms = torch.tensor(
        [l[1].weight.data.norm(dim=0).mean().item() for l in model.layers]
    )

    from rqae.feature import RQAEFeature

    feature = RQAEFeature.from_quantizer(model)
    feature.fit(codebook[all_layers[:4]])
    feature.preload_sims(codebook)

    sims = feature.codebook_pdf(all_layers)[..., :num_layers]
    sims = sims * layer_norms[:num_layers]
    sims = sims.sum(dim=-1)

    for i, coord in enumerate(gemmascope_indices):
        print(
            f"{i}: {text[coord[0]][coord[1]]} {gemmascope_activations[i].item():.3f} {sims[i].item():.3f}"
        )

    return gemmascope_indices, gemmascope_activations, sims


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=60 * 20,
    secrets=[
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def repro_gemma(
    feature_index: int,
    num_layers: int = 1024,
    gemmascope_model_id: str = "gemmascope-gemma-2-2b-res-12-w16k-l82",
    rqae_model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
):
    import gc
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import json
    import random

    from rqae.llm import Gemma2
    from rqae.model import RQAE
    from rqae.gemmascope import JumpReLUSAE

    llm = Gemma2.from_pretrained("google/gemma-2-2b")
    llm.eval()
    llm.layers = None
    gc.collect()

    rqae = RQAE.from_pretrained("google/gemma-2-2b")
    rqae.eval()
    codebook = rqae.codebook.data.detach().clone()[0]  # 1024 x 625 x 4 => 625 x 4
    layer_norms = torch.tensor(
        [l[1].weight.data.norm(dim=0).mean().item() for l in rqae.layers]
    )
    # codebook_sims = (
    #     F.normalize(codebook, dim=-1) @ F.normalize(codebook, dim=-1).T
    # )  # 625 x 625
    codebook_sims = []
    for i in range(1024):
        codebook_sim = (
            F.normalize(rqae.subfeatures[i], dim=-1)
            @ F.normalize(rqae.subfeatures[i], dim=-1).T
        )
        codebook_sims.append(codebook_sim)
    codebook_sims = torch.stack(codebook_sims)  # 1024 x 625 x 625

    sae = JumpReLUSAE.from_pretrained("google/gemma-2-2b")
    sae.eval()

    with torch.inference_mode():
        feature = sae.W_enc.data.clone()[:, feature_index]
        feature_normalized = llm.norm(feature.unsqueeze(0).unsqueeze(0)).clone()
        quantized, indices = rqae(feature_normalized, max_layers=num_layers)
        quantized = llm.denorm(quantized, feature.unsqueeze(0).unsqueeze(0))
        quantized = quantized.squeeze(0).squeeze(0)
        indices = indices.squeeze(0).squeeze(0)
    # print(quantized.shape, feature.shape)
    print(f"Quantization MSE: {F.mse_loss(quantized, feature).item():.9f}")
    print(
        f"Quantization Cosine Similarity: {F.cosine_similarity(quantized.unsqueeze(0), feature.unsqueeze(0)).item():.9f}"
    )
    # print(quantized, feature)

    with open("/data/datasets/monology_pile/text.json", "r") as f:
        text = json.load(f)

    texts_and_similarities = []
    for batch_idx in tqdm(range(36), desc="Comparing against RQAE activations"):
        batch_indices = torch.load(
            f"/data/datasets/monology_pile/activations/{rqae_model_id}/{batch_idx:06d}.pt"
        )
        layers = batch_indices[..., :num_layers]
        sims = []
        for i in range(num_layers):
            sims.append(codebook_sims[i][indices[i].int(), layers[..., i].int()])
        # sims = codebook_sims[indices.int(), layers.int()]
        sims = torch.stack(sims).permute(1, 2, 0)
        sims = sims * layer_norms[:num_layers]
        sims = sims.sum(dim=-1)
        for s in range(batch_indices.shape[0]):
            for t in range(batch_indices.shape[1]):
                texts_and_similarities.append((text[s][t], sims[s, t].item()))
    texts_and_similarities.sort(key=lambda x: x[1], reverse=True)
    for i in range(100):
        print(texts_and_similarities[i])


@app.local_entrypoint()
def main():

    print(repro_gemma.remote(0))
    return

    NUM_TRIALS = 2
    import torch

    torch.set_printoptions(sci_mode=False)

    all_results = spearman.starmap(
        [(i, 32) for i in range(NUM_TRIALS)], return_exceptions=True
    )

    all_indices, all_activations, all_sims = [], [], []
    for result in all_results:
        try:
            all_indices.append(result[0])
            all_activations.append(result[1])
            all_sims.append(result[2])
        except Exception as e:
            continue

    for i in range(100):
        try:
            gemmascope_indices, gemmascope_activations, sims = (
                all_indices[i],
                all_activations[i],
                all_sims[i],
            )
            print(gemmascope_indices.shape, gemmascope_activations.shape, sims.shape)
            # print(gemmascope_indices)
            # print(gemmascope_activations)
            # print(sims)
            print(torch.stack([gemmascope_activations, sims], dim=-1))
        except Exception as e:
            print(f"Index {i} failed with error {e}")
