"""
Runs detection eval for all features.
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
    .pip_install("anthropic")
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from rqae.gemmascope import JumpReLUSAE
    from rqae.feature import Feature
    from transformers import AutoTokenizer
    from tqdm import tqdm


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("anthropic")],
)
def detect_features_gemmascope(
    feature_id: str,
    model_id: str,
    dataset: str = "monology_pile",
    force: bool = False,
):
    from rqae.feature import Feature
    from rqae.evals.detection import detect
    import json
    import os

    feature = Feature.load(
        f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz"
    )
    if feature.scores.get("detection", None) is not None and not force:
        print(f"Skipping {feature_id} because it already has a detection score")
        return feature.scores["detection"]
    if feature.explanation is None or feature.explanation == "":
        print(f"Skipping {feature_id} because it has no explanation")
        return None
    score, messages = detect(feature, verbose=True)
    feature.scores["detection"] = score
    feature.save(f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz")
    os.makedirs(f"/data/datasets/{dataset}/api_outputs", exist_ok=True)
    os.makedirs(f"/data/datasets/{dataset}/api_outputs/{model_id}", exist_ok=True)
    os.makedirs(
        f"/data/datasets/{dataset}/api_outputs/{model_id}/{feature_id}",
        exist_ok=True,
    )
    with open(
        f"/data/datasets/{dataset}/api_outputs/{model_id}/{feature_id}/detection.txt",
        "w",
    ) as f:
        f.write(messages)
    print(feature_id)
    print(score)
    print("=" * 100)
    return score


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("anthropic")],
)
def detect_features_rqae(
    feature_id: str,
    model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
    dataset: str = "monology_pile",
    layer_whitelist=None,
    force: bool = False,
):
    from rqae.feature import RQAEFeature
    from rqae.evals.detection import detect
    import json
    import os
    import numpy as np

    feature = RQAEFeature.load(
        f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz"
    )
    for layer in range(len(feature.layers)):
        if layer_whitelist is not None and feature.layers[layer] not in layer_whitelist:
            continue
        f = feature.to_feature(layer)
        if f.scores.get("detection", None) is not None and not force:
            print(
                f"Skipping {feature_id} layer {feature.layers[layer]} because it already has a detection score"
            )
            continue
        if f.explanation is None or f.explanation == "":
            print(
                f"Skipping {feature_id} layer {feature.layers[layer]} because it has no explanation"
            )
            continue
        # Create new activations array with only top 5% of values
        new_activations = []
        for g in f.activations:
            activations = g["activations"].copy()  # Clone the array
            threshold = np.percentile(activations, 90)  # Get 90th percentile threshold
            activations[activations < threshold] = 0  # Zero out bottom 90%
            activations *= 50.0  # Scale up remaining values
            new_activations.append({"text": g["text"], "activations": activations})
        f.activations = new_activations
        score, messages = detect(f, verbose=True)
        feature.scores[layer]["detection"] = score
        feature.save(f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz")
        os.makedirs(f"/data/datasets/{dataset}/api_outputs", exist_ok=True)
        os.makedirs(f"/data/datasets/{dataset}/api_outputs/{model_id}", exist_ok=True)
        os.makedirs(
            f"/data/datasets/{dataset}/api_outputs/{model_id}/{feature_id}",
            exist_ok=True,
        )
        with open(
            f"/data/datasets/{dataset}/api_outputs/{model_id}/{feature_id}/detection_{feature.layers[layer]}.txt",
            "w",
        ) as f:
            f.write(messages)
        print(feature_id)
        print("=" * 100)
    return feature


@app.local_entrypoint()
def main():
    # for model_id in [
    #     "gemmascope-gemma-2-2b-res-12-w16k-l82",
    #     "gemmascope-gemma-2-2b-res-12-w16k-l22",
    #     "gemmascope-gemma-2-2b-res-12-w16k-l41",
    #     "gemmascope-gemma-2-2b-res-12-w16k-l176",
    #     "gemmascope-gemma-2-2b-res-12-w16k-l445",
    #     "gemmascope-gemma-2-2b-res-12-w32k-l76",
    #     "gemmascope-gemma-2-2b-res-12-w65k-l72",
    #     "gemmascope-gemma-2-2b-res-12-w262k-l67",
    #     "gemmascope-gemma-2-2b-res-12-w524k-l65",
    # ]:
    #     succeeded_count = 0
    #     i = 0
    #     while succeeded_count < 100:
    #         try:
    #             detect_features_gemmascope.remote(
    #                 f"{i:06d}", model_id=model_id, dataset="monology_pile"
    #             )
    #             succeeded_count += 1
    #         except Exception as e:
    #             print(f"Failed on {i:06d}: {e}")
    #         i += 1
    for i in range(100):
        try:
            detect_features_rqae.remote(
                f"{i:06d}",
                model_id="rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
                dataset="monology_pile",
                layer_whitelist=[4, 8, 16, 64, 1023],
            )
        except Exception as e:
            print(f"Failed on {i:06d}: {e}")
