"""
Make RQAE features from the dataset. This is generally pretty hardcoded since we only had to do it once. Will maybe rewrite later.
"""

import modal

app = modal.App("rqae-upload-rqae-features")
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
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from rqae.gemmascope import JumpReLUSAE
    from rqae.feature import Feature, RQAEFeature
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import torch
    import json
    import os


@app.cls(volumes={"/data": volume}, image=image, timeout=60 * 20, concurrency_limit=256)
class FeatureHelper:

    @modal.enter()
    def setup(self):
        self.tokens = torch.load("/data/datasets/monology_pile/tokens.pt")

        with open("/data/datasets/monology_pile/text.json", "r") as f:
            self.texts = json.load(f)

        indices = []
        for i in tqdm(range(self.tokens.shape[0] // 1024), desc="Loading indices"):
            indices.append(
                torch.load(
                    f"/data/datasets/monology_pile/activations/rqae-rqae-round_fsq-cbd4-cbs5-nq1024/{i:06d}.pt"
                )
            )
        self.indices = torch.cat(indices, dim=0)

    @modal.method()
    def get_unique_token_indices(self):
        token_counts = torch.stack(self.tokens.unique(return_counts=True))
        token_counts = {
            token_counts[0][i].item(): token_counts[1][i].item()
            for i in range(token_counts.shape[-1])
        }
        token_counts_sorted = sorted(token_counts.items(), key=lambda x: x[1])
        # Get unique tokens as tensor
        unique_tokens = torch.tensor([t for t, _ in token_counts_sorted])
        # Create a random permutation of the flattened tokens
        flat_indices = torch.randperm(self.tokens.numel())
        flat_tokens = self.tokens.flatten()[flat_indices]
        # Get first occurrence of each unique token in the permuted sequence
        unique_tokens, first_occurrences, counts = torch.unique(
            flat_tokens, return_inverse=True, return_counts=True
        )
        _, ind_sorted = torch.sort(first_occurrences, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_occurrences = ind_sorted[cum_sum]
        # Map back to original indices
        random_token_indices = flat_indices[first_occurrences]
        random_token_indices = torch.tensor(
            [
                [k // self.tokens.shape[1], k % self.tokens.shape[1]]
                for k in random_token_indices
            ]
        ).to(torch.int32)
        return random_token_indices

    @modal.method()
    def get_token_indices(self, index: torch.Tensor):
        if len(index.shape) == 1:
            return self.indices[index[0], index[1]]
        else:
            return self.indices[index[:, 0], index[:, 1]]

    @modal.method()
    def get_text(self, index: torch.Tensor):
        if index.shape[0] == 1:
            return self.texts[index[0].item()]
        else:
            return self.texts[index[0].item()][index[1].item()]

    @modal.method()
    def get_activations(
        self, rqae: RQAEFeature, layers: list = None, top_k: int = 100, index=None
    ):
        import os

        if layers is None:
            layers = rqae.layers
        all_intensities = []
        for b in tqdm(
            range(0, self.indices.shape[0], 1024), desc="Getting intensities"
        ):
            batch_indices = self.indices[b : b + 1024]
            intensities = rqae.intensity(batch_indices, layers=layers)
            all_intensities.append(intensities)
        all_intensities = torch.cat(all_intensities, dim=0)
        all_intensities = all_intensities.flatten(0, 1)  # num_tokens x len(layers)
        activations = {}
        for l_idx, l in tqdm(
            enumerate(layers), desc="Getting activations", total=len(layers)
        ):
            layer_activations = all_intensities[:, l_idx]  # num_tokens
            sorted_indices = torch.argsort(layer_activations, descending=True)
            top_indices = sorted_indices[:top_k]
            bottom_indices = sorted_indices[-top_k:]
            middle_indices = sorted_indices[
                len(sorted_indices) // 2
                - top_k // 2 : len(sorted_indices) // 2
                + top_k // 2
            ]
            indices = torch.cat([top_indices, middle_indices, bottom_indices])
            indices = torch.tensor(
                [[k // self.tokens.shape[1], k % self.tokens.shape[1]] for k in indices]
            ).to(torch.int32)
            sequence_indices = indices[:, 0].tolist()
            seen_sequences = set()
            current_layer_activations = []
            for i, sequence_index in enumerate(sequence_indices):
                if sequence_index in seen_sequences:
                    continue
                seen_sequences.add(sequence_index)
                current_layer_activations.append(
                    {
                        "text": self.texts[sequence_index],
                        "activations": layer_activations[
                            sequence_index
                            * self.tokens.shape[1] : (sequence_index + 1)
                            * self.tokens.shape[1]
                        ].numpy(),
                    }
                )
            activations[l] = current_layer_activations
        if index is not None:
            rqae.activations = activations
            os.makedirs(
                f"/data/datasets/monology_pile/features/rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
                exist_ok=True,
            )
            rqae.save(
                f"/data/datasets/monology_pile/features/rqae-rqae-round_fsq-cbd4-cbs5-nq1024/{index:06d}.npz"
            )
        return activations


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 60)
def make_feature(model_id: str = "google/gemma-2-2b", num_tokens: int = 1024):
    rqae = RQAE.from_pretrained(model_id)
    feature_helper = FeatureHelper()
    random_token_indices = feature_helper.get_unique_token_indices.remote()
    random_token_indices = random_token_indices[200:-200]
    random_token_indices = random_token_indices[
        torch.randperm(random_token_indices.shape[0])
    ]
    random_token_indices = random_token_indices[:num_tokens]
    token_indices = feature_helper.get_token_indices.remote(random_token_indices)
    futures = []
    print(f"Creating features for the following tokens:")
    for rti in random_token_indices:
        print(feature_helper.get_text.remote(rti))
    features = []
    layers = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1023]
    for i in tqdm(
        range(random_token_indices.shape[0]), desc="Sending requests to make features"
    ):
        rti = random_token_indices[i]
        ti = token_indices[i]
        feature = RQAEFeature.from_quantizer(
            rqae,
            center=ti.numpy(),
            layers=layers,
        )
        features.append(feature)
        futures.append(
            feature_helper.get_activations.spawn(feature, index=i, top_k=100)
        )
    for i, f in tqdm(
        enumerate(futures), desc="Getting results and saving", total=len(futures)
    ):
        activation = f.get()


@app.local_entrypoint()
def main():
    feature = make_feature.remote(num_tokens=1024)
