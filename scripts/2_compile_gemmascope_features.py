"""
After creating the activations, we need to compile them into features.
We originally shard activations on the dataset axis, so we need to compile them back into features
I.e. for each feature, get the feature's activations across the whole dataset.
Unfortunately the way I set this up, it does run serially. It's hard to parallelize this across a lot of models without rewriting.

WARNING: We only keep features with >300 activations, because we keep the top, middle, and bottom 100 activations.
This works fine with the monology_pile dataset, because it has >4M tokens.
If your dataset is too small,then this might be too many features that get thrown out. You'll need to adjust the code accordingly.
"""

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
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from rqae.gemmascope import JumpReLUSAE
    from rqae.feature import Feature
    from transformers import AutoTokenizer
    from tqdm import tqdm


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 20)
def make_feature(i_list, ids, model_id: str, dataset: str):
    import torch
    import numpy as np
    import json
    import os

    with open(f"/data/datasets/{dataset}/text.json", "r") as f:
        texts = json.load(f)

    # (sequence, token, feature)
    indices = torch.load(f"/data/temp_indices_{model_id}.pt")
    # (num_tokens,) of just intensities
    activations = torch.load(f"/data/temp_activations_{model_id}.pt")

    os.makedirs(f"/data/datasets/{dataset}/features", exist_ok=True)
    os.makedirs(f"/data/datasets/{dataset}/features/{model_id}", exist_ok=True)

    for idx, i in enumerate(i_list):
        feature_mask = indices[:, 2] == i
        feature_mask = feature_mask & (indices[:, 1] != 0)  # Remove BOS
        feature_indices = indices[feature_mask]
        feature_activations = activations[feature_mask]

        if feature_activations.shape[0] < 300:  # Too few activations
            print(
                f"Skipping feature {i} with {feature_activations.shape[0]} nonzero activations"
            )
            continue
        sorted_indices = feature_activations.argsort(descending=True)
        top_100 = sorted_indices[:100]
        bottom_100 = sorted_indices[-100:]
        middle_100 = sorted_indices[
            len(sorted_indices) // 2 - 50 : len(sorted_indices) // 2 + 50
        ]
        if any(
            texts[f[0].item()][f[1].item()] == "<bos>"
            for f in feature_indices[top_100][:5]
        ):
            print(
                f"Skipping feature {i} because the max activations are BOS (i.e. likely not a covered feature):"
            )
            for f in feature_indices[top_100][:5]:
                print(f"\t{texts[f[0].item()][f[1].item()]}")
            continue

        top_sequences = feature_indices[top_100][:, 0].tolist()
        bottom_sequences = feature_indices[bottom_100][:, 0].tolist()
        middle_sequences = feature_indices[middle_100][:, 0].tolist()

        sequences = set(top_sequences + bottom_sequences + middle_sequences)

        result_activations = []

        for s, seq in enumerate(sequences):
            active_mask = feature_indices[:, 0] == seq
            active_indices = feature_indices[active_mask]
            active_activations = feature_activations[active_mask]
            result_activation = np.zeros(128)
            for active_index, active_activation in zip(
                active_indices, active_activations
            ):
                result_activation[active_index[1]] = active_activation
            result_activations.append(
                {"text": texts[seq], "activations": result_activation}
            )

        feature = Feature(
            id=ids[idx],
            activations=result_activations,
            model=model_id,
        )
        feature.save(f"/data/datasets/{dataset}/features/{model_id}/{ids[idx]}.npz")
    return None


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 20)
def upload_gemmascope_features(
    model_id: str, dataset: str = "monology_pile", max_features: int = 1024
):
    # After compiling the activations, we create the actual features.
    import torch
    import os
    from tqdm import tqdm
    import time

    root = f"/data/datasets/{dataset}/activations/{model_id}/"

    os.makedirs(f"/data/datasets/{dataset}/features/{model_id}", exist_ok=True)

    feature_ids = [f"{i:06d}" for i in range(max_features)]

    batches = [
        (range(i, i + 128), feature_ids[i : i + 128], model_id, dataset)
        for i in range(0, len(feature_ids), 128)
    ]

    res = list(make_feature.starmap(batches))

    os.remove(f"/data/temp_indices_{model_id}.pt")
    os.remove(f"/data/temp_activations_{model_id}.pt")
    print(f"Completed {len(res)} batches")


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 20)
def compile_activations(model_id: str, dataset: str = "monology_pile"):
    # Since we shard on the dataset dimension, we need to compile the activations to create features.

    import torch
    import os
    from tqdm import tqdm
    import time

    indices = []
    activations = []

    root = f"/data/datasets/{dataset}/activations/{model_id}/"

    for i in tqdm(range(36), desc="Loading gemmascope features"):
        indices_path = os.path.join(root, f"{i:06d}_indices.pt")
        activations_path = os.path.join(root, f"{i:06d}.pt")
        # num_tokens, 3 (sequence, token, feature)
        small_indices = torch.load(indices_path)
        small_indices = small_indices.to(torch.int32)
        small_indices[:, 0] += i * 1024  # add batch size
        # (num_tokens,) of just intensities
        small_activations = torch.load(activations_path)
        indices.append(small_indices)
        activations.append(small_activations)

    indices = torch.cat(indices, dim=0)
    activations = torch.cat(activations, dim=0)

    num_features = indices[:, 2].max().item() + 1

    torch.save(indices, f"/data/temp_indices_{model_id}.pt")
    torch.save(activations, f"/data/temp_activations_{model_id}.pt")


@app.local_entrypoint()
def main():
    # We just split it into two steps because for some reason, the file wasn't always saved.
    import time

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l82")
    # time.sleep(10)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l82")
    # time.sleep(10)

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l22")
    # time.sleep(10)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l22")
    # time.sleep(10)

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l41")
    # time.sleep(10)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l41")
    # time.sleep(10)

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l176")
    # time.sleep(20)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l176")

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l445")
    # time.sleep(20)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l445")

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w32k-l76")
    # time.sleep(20)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w32k-l76")

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w65k-l72")
    # time.sleep(20)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w65k-l72")

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w262k-l67")
    # time.sleep(20)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w262k-l67")

    # compile_activations.remote(model_id="gemmascope-gemma-2-2b-res-12-w524k-l65")
    # time.sleep(20)
    # upload_gemmascope_features.remote(model_id="gemmascope-gemma-2-2b-res-12-w524k-l65")
