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


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 30)
def make_feature(
    token_idx,
    feature_id,
    model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
    num_layers: int = 16,
    top_layers: int = 48,
):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from rqae.model import RQAE
    import os
    import numpy as np

    model = RQAE.from_pretrained("google/gemma-2-2b")
    model.eval()
    codebook = model.codebook.data.detach().clone()[0]  # 1024 x 625 x 4 => 625 x 4
    codebook_sims = F.normalize(codebook, dim=-1) @ F.normalize(codebook, dim=-1).T
    layer_norms = torch.tensor(
        [l[1].weight.data.norm(dim=0).mean().item() for l in model.layers]
    )

    s, t = token_idx
    if s // 1024 == 28:
        s = 27 * 1024 + s % 1024
    token_indices = torch.load(
        f"/data/datasets/monology_pile/activations/{model_id}/{s // 1024:06d}.pt"
    )[s % 1024, t]

    # Go through once, get the top correlated layers
    token_sims = []
    for batch_idx in tqdm(range(36)):
        if batch_idx == 28:
            continue  # missing for some weird reason
        batch_indices = torch.load(
            f"/data/datasets/monology_pile/activations/{model_id}/{batch_idx:06d}.pt"
        )
        for b in range(len(batch_indices)):
            sims = codebook_sims[token_indices.int(), batch_indices[b].int()]
            weighted_sims = sims * layer_norms
            weighted_sims_sum = weighted_sims.sum(dim=-1)
            for t, sim in enumerate(weighted_sims_sum):
                token_sims.append((sim.item(), sims[t], batch_idx * 1024 + b, t))

    token_sims.sort(key=lambda x: x[0], reverse=True)
    top_sims = token_sims[:100]

    print(f"Most correlated dimensions:")
    stacked_sims = torch.stack([sims for _, sims, _, _ in top_sims])
    mean_sims = stacked_sims.mean(dim=0)
    top_mean_sim_args = mean_sims.argsort(descending=True)[:top_layers]
    top_mean_sim_vals = mean_sims[top_mean_sim_args]
    for i, (arg, val) in enumerate(zip(top_mean_sim_args, top_mean_sim_vals)):
        print(f"\t{arg} ({val:.3f})")

    # Randomly select layers from the top top_layers
    selected_layers = torch.randperm(top_layers)[:num_layers]
    selected_layers = top_mean_sim_args[selected_layers]

    # Rerun with the selected layers masking
    # Pre-compute layer mask
    layer_mask = torch.zeros(token_indices.shape[-1], dtype=torch.bool)
    layer_mask[selected_layers] = True
    token_sims = []
    for batch_idx in tqdm(range(36)):
        if batch_idx == 28:
            continue  # missing for some weird reason
        batch_indices = torch.load(
            f"/data/datasets/monology_pile/activations/{model_id}/{batch_idx:06d}.pt"
        )
        for b in range(len(batch_indices)):
            sims = codebook_sims[token_indices.int(), batch_indices[b].int()]
            weighted_sims = sims * layer_mask * layer_norms
            weighted_sims_sum = weighted_sims.sum(dim=-1)
            for t, sim in enumerate(weighted_sims_sum):
                token_sims.append((sim.item(), sims[t], batch_idx * 1024 + b, t))
    token_sims.sort(key=lambda x: x[0], reverse=True)
    # token sims is a list of (sim, sim_layer (ignore for now), s, t)
    top_250 = token_sims[:200]
    bottom_25 = token_sims[-50:]
    middle_25 = token_sims[len(token_sims) // 2 - 25 : len(token_sims) // 2 + 25]
    chosen_sims = top_250 + middle_25 + bottom_25

    unique_sequences = set([s for _, _, s, _ in chosen_sims])
    activation_sequences = np.array([[0, x] for x in unique_sequences])
    result_activations = np.zeros((len(unique_sequences), 128))
    for s, seq in enumerate(unique_sequences):
        sims_in_seq = [sim for sim in chosen_sims if sim[2] == seq]
        for sim in sims_in_seq:
            result_activations[s, sim[3]] = sim[0]

    feature = Feature(
        id=feature_id,
        activation_sequences=activation_sequences,
        activations=result_activations,
        model=model_id,
    )
    os.makedirs(f"/data/datasets/monology_pile/features", exist_ok=True)
    os.makedirs(f"/data/datasets/monology_pile/features/{model_id}", exist_ok=True)
    feature.save(f"/data/datasets/monology_pile/features/{model_id}/{feature_id}.npz")


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 30)
def make_features(model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024"):
    import torch
    import random
    from rqae.model import RQAE

    model = RQAE.from_pretrained("google/gemma-2-2b")
    model.eval()

    tokens = torch.load("/data/datasets/monology_pile/tokens.pt")
    unique_tokens, counts = torch.unique(tokens, return_counts=True)
    unique_tokens = unique_tokens[
        counts <= counts.float().quantile(0.95)
    ]  # Don't choose the most common tokens

    NUM_FEATURES = 100

    random_sample_indices = torch.randperm(len(unique_tokens))[:NUM_FEATURES]
    random_sample_tokens = unique_tokens[random_sample_indices]

    random_token_locations = []
    for token in random_sample_tokens:
        token_indices = (tokens == token).nonzero(as_tuple=True)
        random_choice = random.choice(range(len(token_indices[0])))
        random_token_locations.append(
            (token_indices[0][random_choice], token_indices[1][random_choice])
        )

    feature_ids = [f"{i:06d}" for i in range(NUM_FEATURES)]

    list(
        make_feature.starmap(
            [
                (token_idx, feature_id, model_id)
                for token_idx, feature_id in zip(random_token_locations, feature_ids)
            ]
        )
    )


@app.local_entrypoint()
def main():
    make_features.remote(model_id="rqae-rqae-round_fsq-cbd4-cbs5-nq1024")
