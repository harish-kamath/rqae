import modal
from fastapi.responses import StreamingResponse

app = modal.App("rqae-server")
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
    from rqae.feature import RQAEFeature
    from rqae.gemmascope import JumpReLUSAE
    from transformers import AutoTokenizer
    from tqdm import tqdm

DEFAULT_FUNCTION_ARGS = {
    "volumes": {"/data": volume},
    "image": image,
    "concurrency_limit": 1,
    "container_idle_timeout": 60 * 5,
}


@app.cls(**DEFAULT_FUNCTION_ARGS)
class Text:
    @modal.enter()
    def setup(self):
        import json

        with open("/data/datasets/monology_pile/text.json", "r") as f:
            self.texts = json.load(f)

    @modal.method()
    def sequence(self, idx=None):
        if idx is None:
            import random

            idx = random.randint(0, len(self.texts) - 1)
        return idx, self.texts[idx][1:]


@app.cls(**DEFAULT_FUNCTION_ARGS)
class Feature:
    @modal.enter()
    def setup(self):
        pass

    @modal.method()
    def get(self, model_id: str, id: str):
        from rqae.feature import Feature, RQAEFeature

        feature_path = f"/data/datasets/monology_pile/features/{model_id}/{id}.npz"
        if "rqae" in model_id:
            feature = RQAEFeature.load(feature_path)
        else:
            feature = Feature.load(feature_path)
        return feature


@app.function(**DEFAULT_FUNCTION_ARGS)
@modal.web_endpoint()
def sequence(idx=None):
    api = Text()
    if idx is not None:
        idx = int(idx)
    return api.sequence.remote(idx)


def describe_feature(feature):
    import numpy as np

    sequences = np.array([k["text"] for k in feature.activations])
    activations = np.array([k["activations"] for k in feature.activations])
    activations_order = activations.max(1).argsort()
    activations_order = activations_order[::-1]
    sequences = sequences[activations_order]
    activations = activations[activations_order]

    top_10 = range(20)
    middle_5 = range(len(sequences) // 2 - 5, len(sequences) // 2 + 5)
    last_5 = range(len(sequences) - 5, len(sequences))

    subsets = {
        "Top Activations": list(top_10),
        "Median Activations": list(middle_5),
        "Bottom/Zero Activations": list(last_5),
    }
    import json

    return {
        subset_name: {
            "explanation": feature.explanation,
            "scores": feature.scores,
            "sequences": sequences[subset][:, 1:].tolist(),
            "activations": activations[subset][:, 1:].tolist(),
        }
        for subset_name, subset in subsets.items()
    }


def get_feature(model_id: str, id: str):
    from rqae.feature import RQAEFeature
    import numpy as np

    features = Feature()
    texts = Text()
    feature = features.get.remote(model_id, id)
    all_features = {}
    if isinstance(feature, RQAEFeature):
        for layer_idx, layer in enumerate(feature.layers):
            all_features[layer.item()] = describe_feature(feature.to_feature(layer_idx))
    else:
        all_features[0] = describe_feature(feature)
    return all_features


@app.function(**DEFAULT_FUNCTION_ARGS)
@modal.web_endpoint()
def feature_web(model_id: str, id: str):
    import json

    return json.dumps(get_feature(model_id, id))
