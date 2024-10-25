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
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
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
        from rqae.feature import Feature

        feature_path = f"/data/datasets/monology_pile/features/{model_id}/{id}.npz"
        feature = Feature.load(feature_path)
        return feature


@app.function(**DEFAULT_FUNCTION_ARGS)
@modal.web_endpoint()
def sequence(idx=None):
    api = Text()
    if idx is not None:
        idx = int(idx)
    return api.sequence.remote(idx)


def feature(model_id: str, id: str):
    features = Feature()
    texts = Text()
    feature = features.get.remote(model_id, id)
    sequences = feature.activation_sequences
    activations = feature.activations
    activations_order = activations.max(1).argsort()
    activations_order = activations_order[::-1]
    sequences = sequences[activations_order]
    activations = activations[activations_order]

    top_10 = range(20)
    middle_5 = range(len(sequences) // 2 - 5, len(sequences) // 2 + 5)
    last_5 = range(len(sequences) - 5, len(sequences))
    subset = list(top_10) + list(middle_5) + list(last_5)

    sequences_and_activations = []
    import json

    for i, sequence in zip(subset, sequences[subset]):
        _, seq = texts.sequence.remote(sequence[1])
        seq_activations = activations[i].tolist()[1:]
        yield json.dumps({"sequence": seq, "activations": seq_activations}) + "\n"


@app.function(**DEFAULT_FUNCTION_ARGS)
@modal.web_endpoint()
def feature_web(model_id: str, id: str):
    return StreamingResponse(feature(model_id, id), media_type="text/event-stream")
