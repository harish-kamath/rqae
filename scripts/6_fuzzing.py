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
def fuzz_features_gemmascope(
    feature_id: str,
    model_id: str,
    dataset: str,
    force: bool = False,
    verbose: bool = False,
):
    from rqae.feature import Feature
    from rqae.evals.fuzzing import fuzz
    import json
    import os

    feature = Feature.load(
        f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz"
    )
    if feature.scores.get("fuzzing", None) is not None and not force:
        print(f"Skipping {feature_id} because it already has a fuzzing score")
        return feature.scores["fuzzing"]
    if feature.explanation is None or feature.explanation == "":
        print(f"Skipping {feature_id} because it has no explanation")
        return None
    score, messages = fuzz(feature, verbose=verbose)
    feature.scores["fuzzing"] = score
    feature.save(f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz")
    os.makedirs(f"/data/datasets/{dataset}/api_outputs", exist_ok=True)
    os.makedirs(f"/data/datasets/{dataset}/api_outputs/{model_id}", exist_ok=True)
    os.makedirs(
        f"/data/datasets/{dataset}/api_outputs/{model_id}/{feature_id}",
        exist_ok=True,
    )
    with open(
        f"/data/datasets/{dataset}/api_outputs/{model_id}/{feature_id}/fuzzing.txt",
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
def fuzz_features_rqae(
    feature_id: str,
    model_id: str,
    dataset: str,
    layer_whitelist=None,
    force: bool = False,
    verbose: bool = False,
):
    from rqae.feature import RQAEFeature
    from rqae.evals.fuzzing import fuzz
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
        if f.scores.get("fuzzing", None) is not None and not force:
            print(
                f"Skipping {feature_id} layer {feature.layers[layer]} because it already has a fuzzing score"
            )
            continue
        if f.explanation is None or f.explanation == "":
            print(
                f"Skipping {feature_id} layer {feature.layers[layer]} because it has no explanation"
            )
            continue
        score, messages = fuzz(f, verbose=verbose)
        feature.scores[layer]["fuzzing"] = score
        feature.save(f"/data/datasets/{dataset}/features/{model_id}/{feature_id}.npz")
        os.makedirs(f"/data/datasets/{dataset}/api_outputs", exist_ok=True)
        os.makedirs(
            f"/data/datasets/monology_pile/api_outputs/{model_id}", exist_ok=True
        )
        os.makedirs(
            f"/data/datasets/monology_pile/api_outputs/{model_id}/{feature_id}",
            exist_ok=True,
        )
        with open(
            f"/data/datasets/monology_pile/api_outputs/{model_id}/{feature_id}/fuzzing_{feature.layers[layer]}.txt",
            "w",
        ) as f:
            f.write(messages)
        print(feature_id)
        print(score)
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
    #             fuzz_features_gemmascope.remote(
    #                 f"{i:06d}",
    #                 model_id=model_id,
    #                 dataset="monology_pile",
    #                 force=True,
    #                 verbose=True,
    #             )
    #             succeeded_count += 1
    #         except Exception as e:
    #             print(f"Failed on {i:06d}: {e}")
    #         i += 1
    for i in range(100):
        try:
            fuzz_features_rqae.remote(
                f"{i:06d}",
                model_id="rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
                dataset="monology_pile",
                layer_whitelist=[256],
                force=True,
                verbose=True,
            )
        except Exception as e:
            print(f"Failed on {i:06d}: {e}")
