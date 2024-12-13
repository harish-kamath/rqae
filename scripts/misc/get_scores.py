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
def get_scores(model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024", top_n: int = 10):
    import os
    import numpy as np
    from tqdm import tqdm

    scores = {}
    for feature_id in tqdm(
        os.listdir(f"/data/datasets/monology_pile/features/{model_id}")
    ):
        feature_id = feature_id.split(".")[0]
        feature = Feature.load(
            f"/data/datasets/monology_pile/features/{model_id}/{feature_id}.npz"
        )
        for k, v in feature.scores.items():
            if k not in scores:
                scores[k] = []
            scores[k].append((feature_id, v))

    # Show statistics for all scores
    for k, v in scores.items():
        value_scores = [x[1] for x in v]
        print(f"Eval {k}")
        print(f"\tN: {len(v)}")
        print(f"\tMean: {np.mean(value_scores)}")
        print(f"\tStd: {np.std(value_scores)}")
        print(f"\tMin: {np.min(value_scores)}")
        print(f"\tMax: {np.max(value_scores)}")
        print(f"\tMedian: {np.median(value_scores)}")
        print(f"\t25%: {np.percentile(value_scores, 25)}")
        print(f"\t75%: {np.percentile(value_scores, 75)}")
        feature_ids_and_scores = sorted(v, key=lambda x: -x[1])
        print(f"\tTop {top_n}:")
        for feature_id, score in feature_ids_and_scores[:top_n]:
            print(f"\t\t{feature_id}: {score}")
        print(f"\tBottom {top_n}:")
        for feature_id, score in feature_ids_and_scores[-top_n:]:
            print(f"\t\t{feature_id}: {score}")


@app.local_entrypoint()
def main():
    get_scores.remote(model_id="gemmascope-gemma-2-2b-res-12-w16k-l82")
