# Encode SAE encoder vectors with RQAE. Then, calculate the Pearson correlation across the activation distribution as you go up the layers.
# Finally, compare this to encoding the top activating token for that feature instead.

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
    from rqae.feature import Feature, RQAEFeature
    from transformers import AutoTokenizer
    from tqdm import tqdm


# Returns (1024,) codes
def encode_sae_vector(sae: JumpReLUSAE, rqae: RQAE, i: int, norm_fn: callable):
    encoding_vector = sae.W_enc[:, i]
    encoding_vector = encoding_vector
    encoding_vector = norm_fn(encoding_vector)
    return rqae(encoding_vector.unsqueeze(0).unsqueeze(0))[1][0, 0]


def cosine_sim(activation, sae: JumpReLUSAE, i: int):
    # activation: (128, 2304)
    # returns: (128)
    import torch

    encoding_vector = sae.W_enc[:, i]  # (2304)
    return torch.nn.functional.cosine_similarity(activation, encoding_vector, dim=1)


@app.cls(volumes={"/data": volume}, image=image, timeout=60 * 20, concurrency_limit=1)
class TextsAndIndices:
    @modal.enter()
    def setup(self):
        import torch
        import json
        from tqdm import tqdm

        with open("/data/datasets/monology_pile/text.json", "r") as f:
            self.texts = json.load(f)
        self.texts = ["".join(t) for t in self.texts]
        self.indices = []
        for i in tqdm(range(36), desc="Loading activations"):
            self.indices.append(
                torch.load(
                    f"/data/datasets/monology_pile/activations/rqae-rqae-round_fsq-cbd4-cbs5-nq1024/{i:06d}.pt"
                )
            )
        self.indices = torch.cat(self.indices, dim=0)

        self.raw_activations = []
        for i in tqdm(range(36), desc="Loading raw activations"):
            self.raw_activations.append(
                torch.load(f"/data/datasets/monology_pile/activations/raw/{i:06d}.pt")
            )

    @modal.method()
    def get(self, text: str):
        idx = self.texts.index("".join(text))
        return self.indices[idx].int()

    def get_activation_internal(self, text: str):
        import torch

        idx = self.texts.index("".join(text))
        return self.raw_activations[idx // 1024][idx % 1024].to(torch.float16)

    @modal.method()
    def get_activation(self, text: str):
        return self.get_activation_internal(text)

    @modal.method()
    def get_activations(self, texts: list[str]):
        from tqdm import tqdm
        import torch

        return torch.stack([self.get_activation_internal(t) for t in tqdm(texts)])


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("anthropic")],
)
def collect(
    feature_index: int,
    layers: list = None,
    gemmascope_model_id: str = "gemmascope-gemma-2-2b-res-12-w16k-l82",
    rqae_model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
):
    import gc
    import json
    import torch
    from tqdm import tqdm

    # Load required models
    rqae = RQAE.from_pretrained("google/gemma-2-2b").eval()
    sae = JumpReLUSAE.from_pretrained("google/gemma-2-2b").eval()
    llm = Gemma2.from_pretrained("google/gemma-2-2b").eval()
    del llm.model.model.layers
    gc.collect()

    # Load the Gemmascope Feature
    feature = Feature.load(
        f"/data/datasets/monology_pile/features/{gemmascope_model_id}/{feature_index:06d}.npz"
    )

    # Encode the feature with RQAE
    feature_codes = encode_sae_vector(sae, rqae, feature_index, llm.norm)

    # Create a new RQAE feature
    rqae_feature = RQAEFeature(center=feature_codes, layers=range(1024))
    rqae_feature.load_model(rqae)

    # Calculate the RQAE intensities for the same activation distribution
    all_a_and_is = []
    tai = TextsAndIndices()
    chosen_activations = feature.activations
    chosen_activations = sorted(
        chosen_activations, key=lambda x: x["activations"].max().item(), reverse=True
    )
    # Choose top 10, middle 10, bottom 10
    chosen_activations = (
        chosen_activations[:10]
        + chosen_activations[
            len(chosen_activations) // 2 - 5 : len(chosen_activations) // 2 + 5
        ]
        + chosen_activations[-10:]
    )
    raw_activations = tai.get_activations.remote(
        [a["text"] for a in chosen_activations]
    )
    print(f"Raw activations shape: {raw_activations.shape}")

    top_activation_index = tai.get.remote(chosen_activations[0]["text"])
    top_activation_index = top_activation_index[
        chosen_activations[0]["activations"].argmax()
    ]
    top_activation_feature = RQAEFeature(
        center=top_activation_index, layers=range(1024)
    )
    top_activation_feature.load_model(rqae)

    for i, a in tqdm(
        enumerate(chosen_activations),
        desc="Processing activations",
        total=len(chosen_activations),
    ):
        activation_indices = tai.get.remote(a["text"])
        intensities = []
        top_activation_intensities = []
        for ai in activation_indices:  # Doing all at once stalls a lot for some reason
            intensity = rqae_feature.intensity(ai)
            intensities.append(intensity)
            top_activation_intensity = top_activation_feature.intensity(ai)
            top_activation_intensities.append(top_activation_intensity)
        intensities = torch.stack(intensities, dim=0)
        top_activation_intensities = torch.stack(top_activation_intensities, dim=0)
        activations = a["activations"]
        raw_activation = raw_activations[i]
        cosine_similarities = cosine_sim(raw_activation, sae, feature_index)
        a_and_i = torch.cat(
            [
                cosine_similarities.unsqueeze(1),
                torch.from_numpy(activations).unsqueeze(1),
                intensities,
                top_activation_intensities,
            ],
            dim=-1,
        )
        all_a_and_is.append(a_and_i)
    all_a_and_is = torch.stack(all_a_and_is, dim=0)
    return all_a_and_is


@app.local_entrypoint()
def main():
    import os
    import torch

    os.makedirs("pearson_activations", exist_ok=True)
    futures = []
    for i in range(75):
        futures.append(collect.spawn(i))
    for i, future in enumerate(futures):
        print(f"Processing {i}")
        all_a_and_is = future.get()
        torch.save(all_a_and_is, f"pearson_activations/{i:06d}.pt")


def pearson_correlation_with_first(tensor):
    """
    Calculate Pearson correlation coefficient between the first entry and all other entries

    Args:
        tensor: Input tensor of shape [B, S, N]

    Returns:
        torch.Tensor: Correlation coefficients of shape [N-1]
    """
    # Flatten the first two dimensions
    flattened = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])

    # Get the first entry ([:, :, 0] flattened)
    first_entry = flattened[:, 0]
    nonzero_mask = first_entry != 0
    flattened = flattened[nonzero_mask]
    first_entry = first_entry[nonzero_mask]

    # Calculate mean and standard deviation
    first_mean = first_entry.mean()
    first_std = first_entry.std()

    # import matplotlib.pyplot as plt
    # plt.scatter(flattened[:, 1], flattened[:, -1])
    # plt.show()

    # Initialize array for correlation coefficients
    correlations = torch.zeros(tensor.shape[2] - 1, device=tensor.device)

    # Calculate correlation with each other entry
    for i in range(1, tensor.shape[2]):
        other_entry = flattened[:, i]
        other_mean = other_entry.mean()
        other_std = other_entry.std()

        # Calculate correlation coefficient
        correlation = (
            ((first_entry - first_mean) * (other_entry - other_mean)).mean()
        ) / (first_std * other_std)
        correlations[i - 1] = correlation

    return correlations


def spearman_correlation_with_first(tensor):
    """
    Calculate Spearman correlation coefficient between the first entry and all other entries

    Args:
        tensor: Input tensor of shape [B, S, N]

    Returns:
        torch.Tensor: Correlation coefficients of shape [N-1]


    Not used bc I don't want to figure out how to deal with ties hehe
    """
    # Flatten the first two dimensions
    flattened = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])

    # Get the first entry ([:, :, 0] flattened)
    first_entry = flattened[:, 0]
    nonzero_mask = first_entry != 0
    flattened = flattened[nonzero_mask]
    first_entry = first_entry[nonzero_mask]

    # Convert first entry to ranks
    first_ranks = first_entry.argsort().argsort().float()

    # Initialize array for correlation coefficients
    correlations = torch.zeros(tensor.shape[2] - 1, device=tensor.device)

    # Calculate correlation with each other entry
    for i in range(1, tensor.shape[2]):
        other_entry = flattened[:, i]
        # Convert other entry to ranks
        other_ranks = other_entry.argsort().argsort().float()

        # Calculate Spearman correlation using Pearson on ranks
        first_mean = first_ranks.mean()
        other_mean = other_ranks.mean()
        first_std = first_ranks.std()
        other_std = other_ranks.std()

        correlation = (
            ((first_ranks - first_mean) * (other_ranks - other_mean)).mean()
        ) / (first_std * other_std)
        correlations[i - 1] = correlation

    return correlations


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    from scripts.plotting_utils import *

    correlations1 = []
    correlations2 = []

    for i in tqdm(range(len(os.listdir("pearson_activations")) - 1)):
        feature = torch.load(f"pearson_activations/{i:06d}.pt").detach()

        # Compare cosine similarities to activations just as a sanity check
        # if i < 50:
        #     continue
        # print(feature.shape)
        # cs = feature[..., 0].flatten()
        # act = feature[..., 1000].flatten()
        # nonzero_mask = act != 0
        # cs = cs[nonzero_mask]
        # act = act[nonzero_mask]
        # print(act)
        # plt.scatter(cs, act)
        # plt.show()
        # exit(0)

        # First row is just cosine similarities for above plot
        feature = feature[..., 1:]

        c1 = feature[..., :1025]
        c2 = torch.cat((feature[..., 0].unsqueeze(-1), feature[..., 1025:]), dim=-1)

        correlation1 = pearson_correlation_with_first(c1)
        correlation2 = pearson_correlation_with_first(c2)
        correlations1.append(correlation1)
        correlations2.append(correlation2)

    correlations1 = torch.stack(correlations1, dim=0)
    correlations2 = torch.stack(correlations2, dim=0)

    # Choose the bottom 10 as measured by the value on the 64th index
    # sort by this value
    bottom = correlations1[..., 512].argsort()[5:]
    print(bottom)

    correlations1 = correlations1[bottom]
    correlations2 = correlations2[bottom]

    # Plot with error area shaded in
    plt.fill_between(
        range(correlations1.shape[1]),
        correlations1.mean(dim=0) - correlations1.std(dim=0),
        correlations1.mean(dim=0) + correlations1.std(dim=0),
        alpha=0.2,
        color=create_blog_colormaps()["primary_accent"](0.1),
    )
    plt.plot(
        range(correlations1.shape[1]),
        correlations1.mean(dim=0),
        label="Encoding Vector",
        color=create_blog_colormaps()["primary_accent"](0.1),
    )
    plt.fill_between(
        range(correlations2.shape[1]),
        correlations2.mean(dim=0) - correlations2.std(dim=0),
        correlations2.mean(dim=0) + correlations2.std(dim=0),
        alpha=0.2,
        color=create_blog_colormaps()["primary_accent"](0.9),
    )
    plt.plot(
        range(correlations2.shape[1]),
        correlations2.mean(dim=0),
        label="Top Activating Token",
        color=create_blog_colormaps()["primary_accent"](0.9),
    )

    # plt.plot(
    #     range(correlations1.shape[1]),
    #     correlations1.min(dim=0).values,
    #     linestyle="--",
    #     alpha=0.5,
    # )
    # plt.plot(
    #     range(correlations2.shape[1]),
    #     correlations2.min(dim=0).values,
    #     linestyle="--",
    #     alpha=0.5,
    # )
    plt.xlabel("Layer Number")
    plt.ylabel("Pearson Correlation")
    plt.legend(loc="lower right")
    plt.savefig("pearson_activations.jpg", bbox_inches="tight")
    plt.close()
