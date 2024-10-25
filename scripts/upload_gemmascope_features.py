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
def make_feature(i_list, ids, model_id: str = "gemmascope-gemma-2-2b-res-12-w16k-l82"):
    import torch
    import numpy as np

    indices = torch.load("/data/temp_indices.pt")
    activations = torch.load("/data/temp_activations.pt")

    for idx, i in enumerate(i_list):
        feature_mask = indices[:, 2] == i
        feature_mask = feature_mask & (indices[:, 1] != 0)  # Remove BOS
        feature_indices = indices[feature_mask]
        feature_activations = activations[feature_mask]

        if feature_activations.shape[0] < 100:
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

        top_sequences = feature_indices[top_100][:, 0].tolist()
        bottom_sequences = feature_indices[bottom_100][:, 0].tolist()
        middle_sequences = feature_indices[middle_100][:, 0].tolist()

        sequences = set(top_sequences + bottom_sequences + middle_sequences)

        activation_sequences = np.array([[0, x] for x in sequences])
        result_activations = np.zeros((len(sequences), 128))
        for s, seq in enumerate(sequences):
            active_mask = feature_indices[:, 0] == seq
            active_indices = feature_indices[active_mask]
            active_activations = feature_activations[active_mask]
            for active_index, active_activation in zip(
                active_indices, active_activations
            ):
                result_activations[s, active_index[1]] = active_activation

        feature = Feature(
            id=ids[idx],
            activation_sequences=activation_sequences,
            activations=result_activations,
            model=model_id,
        )
        feature.save(f"/data/datasets/monology_pile/features/{model_id}/{ids[idx]}.npz")
    return None


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 20)
def upload_gemmascope_features(model_id: str = "gemmascope-gemma-2-2b-res-12-w16k-l82"):
    import torch
    import os
    from tqdm import tqdm
    import time

    root = f"/data/datasets/monology_pile/activations/{model_id}/"

    # indices = []
    # activations = []

    # for i in tqdm(range(36), desc="Loading gemmascope features"):
    #     indices_path = os.path.join(root, f"{i:06d}_indices.pt")
    #     activations_path = os.path.join(root, f"{i:06d}.pt")
    #     # num_tokens, 3 (sequence, token, feature)
    #     small_indices = torch.load(indices_path)
    #     small_indices = small_indices.to(torch.int32)
    #     small_indices[:, 0] += i * 1024  # add batch size
    #     # (num_tokens,) of just intensities
    #     small_activations = torch.load(activations_path)
    #     indices.append(small_indices)
    #     activations.append(small_activations)

    # indices = torch.cat(indices, dim=0)
    # activations = torch.cat(activations, dim=0)

    # num_features = indices[:, 2].max().item() + 1

    # torch.save(indices, f"/data/temp_indices.pt")
    # torch.save(activations, f"/data/temp_activations.pt")
    # import time

    # time.sleep(10)

    num_features = 16384

    os.makedirs(f"/data/datasets/monology_pile/features/{model_id}", exist_ok=True)

    feature_ids = [f"{i:06d}" for i in range(num_features)]

    batches = [
        (range(i, i + 100), feature_ids[i : i + 100])
        for i in range(0, len(feature_ids), 100)
    ]

    res = list(make_feature.starmap(batches))

    os.remove("/data/temp_indices.pt")
    os.remove("/data/temp_activations.pt")
    print(f"Completed {len(res)} batches")


@app.local_entrypoint()
def main():
    upload_gemmascope_features.remote()
