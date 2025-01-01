from typing import List, Callable
import modal
import torch
from fastapi.responses import StreamingResponse
import json
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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
    "container_idle_timeout": 60 * 5,
}


def get_intensities(activations, sims):
    """
    Given sims for codebooks and the indices of codebooks, return the intensities.
    """
    import torch

    # Get input tensor dimensions
    B, S, D = activations.shape  # Batch size, Sequence length, layers
    S2, D2, _ = sims.shape  # tensor2 has shape [S2, layers, num_indices]

    # Reshape tensor1 to [B*S, layers] and add dimension for gathering
    # Shape: [B*S, layers, 1]
    indices = activations.reshape(B * S, D).unsqueeze(-1)

    # Expand tensor2 for batched gathering
    # Original shape: [S2, layers, num_indices] -> [1, S2, layers, num_indices] -> [B*S, S2, layers, num_indices]
    expanded_sims = sims.unsqueeze(0).expand(B * S, -1, -1, -1)

    # Gather along last dimension and squeeze
    # Shape: [B*S, S2, layers]
    gathered = torch.gather(
        expanded_sims, -1, indices.unsqueeze(1).expand(-1, S2, -1, -1).long()
    ).squeeze(-1)

    # Reshape back to original dimensions [B, S, S2, layers]
    result = gathered.reshape(B, S, S2, D)

    return result


@app.cls(**DEFAULT_FUNCTION_ARGS, gpu="l4")
class IntensityEngine:

    def __init__(
        self,
        dataset: str = "monology_pile",
        model_id: str = "rqae-rqae-round_fsq-cbd4-cbs5-nq1024",
    ):
        self.dataset = dataset
        self.model_id = model_id

    @modal.build()
    def build(self):
        # Download model weights
        from rqae.model import RQAE

        model = RQAE.from_pretrained(self.model_id).eval()

    @modal.enter()
    def setup(self):
        dataset = self.dataset
        model_id = self.model_id

        from rqae.model import RQAE
        import concurrent.futures
        import os
        import gc
        from tqdm import tqdm

        print(f"Loading model {model_id}")
        model = RQAE.from_pretrained(model_id).eval().cuda()

        mode = "projected"
        if mode == "original":
            sims = model.codebook_sims.unsqueeze(0).repeat(model.num_quantizers, 1, 1)
        elif mode == "projected":
            sims = model.subfeature_sims
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # num_quantizers, codebook_size, codebook_size

        sims *= model.layer_norms.unsqueeze(-1).unsqueeze(-1)
        # layer_norms is of shape (num_quantizers,)

        self.sims = sims.clone().detach()
        # (num_quantizers, codebook_size, codebook_size)
        del model
        gc.collect()

        print(f"Loading activations for {dataset}")
        activations_folder = f"/data/datasets/{dataset}/activations/{model_id}"
        activation_files = [f for f in os.listdir(activations_folder) if "_ce" not in f]
        activation_files = sorted(activation_files, key=lambda x: int(x.split(".")[0]))

        def load_file(f):
            return torch.load(os.path.join(activations_folder, f)).int()[
                :, 1:
            ]  # skip BOS

        # TODO: Multithreading doesn't actually make this faster unfortunately
        with concurrent.futures.ThreadPoolExecutor() as executor:
            activations = list(
                tqdm(
                    executor.map(load_file, activation_files),
                    total=len(activation_files),
                    desc="Loading activations",
                )
            )
        self.activations = activations
        # Each activation is of shape (1024, 127, num_quantizers)

    @modal.method()
    def get_activations(
        self, source_idx: int, target_indices: List[int], layers: List[int]
    ):
        shard_idx = source_idx // 1024
        source_activation = self.activations[shard_idx][source_idx % 1024]
        source_activation = source_activation[:, : max(layers)]
        # (127, max(layers)) int

        shard_idxs = [t // 1024 for t in target_indices]
        target_activations = [
            self.activations[shard_idx][t % 1024] for t in target_indices
        ]
        target_activations = torch.stack(target_activations)
        target_activations = target_activations[..., : max(layers)]
        # (len(target_indices), 127, max(layers)) int

    @modal.method()
    def find_examples(
        self,
        idx: int = None,
        activation: torch.Tensor = None,
        top_examples: int = 30,
        middle_examples: int = 10,
        bottom_examples: int = 10,
        layers: List[int] = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1023],
    ):
        from tqdm import tqdm

        # Get the activation for the query
        if activation is not None and idx is not None:
            raise ValueError("Cannot specify both idx and activation")
        elif idx is not None:
            shard_idx = idx // 1024
            query_activation = self.activations[shard_idx][idx % 1024]
        elif activation is not None:
            query_activation = activation
        else:
            raise ValueError("Must specify either idx or activation")
        # (127, num_quantizers) int

        query_activation = query_activation[:, : max(layers)].T
        query_activation = query_activation.cuda()
        # (max(layers), 127) int

        query_sims = self.sims[: max(layers)]
        query_sims = query_sims.to(torch.float16)
        query_sims = query_sims.cuda()
        # query_sims = torch.gather(
        #     query_sims,
        #     1,
        #     query_activation.unsqueeze(-1).expand(-1, -1, query_sims.size(2)).long(),
        # )
        qsims = []
        for l in range(max(layers)):
            qsims.append(query_sims[l, query_activation[l]])
        query_sims = torch.stack(qsims)
        # (max(layers), 127, 625) float

        # Stream, so you don't have to wait for the very last layer.
        layer_ranges = [0] + layers
        layer_ranges = [
            (layer_ranges[i], layer_ranges[i + 1]) for i in range(len(layer_ranges) - 1)
        ]
        intensity_accumulation = None

        import time

        for layer, layer_range in tqdm(
            zip(layers, layer_ranges), total=len(layers), desc="Layers"
        ):
            intensities_across_shards = []
            for shard in self.activations:
                range_size = layer_range[1] - layer_range[0]
                if range_size > 64:
                    # Process in chunks of 64
                    intensities = None
                    for chunk_start in range(layer_range[0], layer_range[1], 64):
                        chunk_end = min(chunk_start + 64, layer_range[1])
                        partial_shard = shard[..., chunk_start:chunk_end]
                        partial_shard = partial_shard.cuda()
                        # (1024, 127, chunk_size) int
                        partial_sims = query_sims[chunk_start:chunk_end]
                        # (chunk_size, 127, 625) float

                        chunk_intensity = get_intensities(
                            partial_shard, partial_sims.transpose(0, 1)
                        )
                        # (1024, 127, 127, chunk_size) float
                        chunk_intensity = chunk_intensity.sum(dim=-1)
                        # (1024, 127, 127) float
                        if intensities is None:
                            intensities = chunk_intensity
                        else:
                            intensities += chunk_intensity
                else:
                    partial_shard = shard[..., layer_range[0] : layer_range[1]]
                    partial_shard = partial_shard.cuda()
                    # (1024, 127, layer range) int
                    partial_sims = query_sims[layer_range[0] : layer_range[1]]
                    # (layer range, 127, 625) float

                    intensities = get_intensities(
                        partial_shard, partial_sims.transpose(0, 1)
                    )
                    # (1024, 127, 127, layer range) float
                    intensities = intensities.sum(dim=-1)
                    # (1024, 127, 127) float

                intensities_across_shards.append(intensities)

            intensities_across_shards = torch.cat(intensities_across_shards, dim=0)
            # (dataset size, 127, 127) float

            if intensity_accumulation is None:
                intensity_accumulation = intensities_across_shards
            else:
                intensity_accumulation += intensities_across_shards

            # Find top, middle, and bottom examples
            # Get max values across sequence dimension for each position
            max_values = intensity_accumulation.max(dim=1).values
            # (dataset_size, 127)
            sorted_sequences = max_values.argsort(dim=0, descending=True)
            # (dataset_size, 127)

            # Get top examples for each position
            top_examples_list = sorted_sequences[:top_examples]  # (top_examples, 127)

            # Get middle examples for each position
            mid_start = sorted_sequences.shape[0] // 2 - middle_examples // 2
            mid_end = sorted_sequences.shape[0] // 2 + middle_examples // 2
            middle_examples_list = sorted_sequences[
                mid_start:mid_end
            ]  # (middle_examples, 127)

            # Get bottom examples for each position
            bottom_examples_list = sorted_sequences[
                -bottom_examples:
            ]  # (bottom_examples, 127)

            # Transpose to match original shape (127, #examples)
            top_examples_list = top_examples_list.T
            middle_examples_list = middle_examples_list.T
            bottom_examples_list = bottom_examples_list.T
            # (127, # examples) int

            # Get intensities for each example
            top_intensities = torch.stack(
                [
                    intensity_accumulation[top_examples_list[i], :, i]
                    for i in range(intensity_accumulation.shape[2])
                ]
            )
            middle_intensities = torch.stack(
                [
                    intensity_accumulation[middle_examples_list[i], :, i]
                    for i in range(intensity_accumulation.shape[2])
                ]
            )
            bottom_intensities = torch.stack(
                [
                    intensity_accumulation[bottom_examples_list[i], :, i]
                    for i in range(intensity_accumulation.shape[2])
                ]
            )
            # (127, # examples, 127) float
            yield (
                {
                    "top": {
                        "indices": top_examples_list.cpu().int(),
                        "intensities": top_intensities.cpu().to(torch.float16),
                    },
                    "middle": {
                        "indices": middle_examples_list.cpu().int(),
                        "intensities": middle_intensities.cpu().to(torch.float16),
                    },
                    "bottom": {
                        "indices": bottom_examples_list.cpu().int(),
                        "intensities": bottom_intensities.cpu().to(torch.float16),
                    },
                },
                layer,
            )


@app.cls(**DEFAULT_FUNCTION_ARGS)
class Dataset:

    def __init__(self, dataset: str = "monology_pile"):
        self.dataset = dataset

    @modal.enter()
    def setup(self):
        import json
        import time

        print(f"Loading text for {self.dataset}")
        start = time.time()
        with open(f"/data/datasets/{self.dataset}/text.json", "r") as f:
            self.text = json.load(f)
        self.text = [t[1:] for t in self.text]  # skip BOS
        print(f"Time taken: {time.time() - start:.3f}s")

    @modal.method()
    def get_text(self, idx: list = None):
        if idx is None:
            import random

            idx = random.randint(0, len(self.text) - 1)
        if type(idx) == int:
            idx = [idx]

        texts = [self.text[i] for i in idx]
        return list(zip(idx, texts))  # Return tuples of (idx, text)

    @modal.method()
    def search_texts(self, query: str, limit: int = 10):
        matching_texts = []
        for idx, text in enumerate(self.text):
            joined_text = "".join(text)
            if query.lower() in joined_text.lower():
                matching_texts.append({"text": text, "id": idx})
                if len(matching_texts) >= limit:
                    break
        return matching_texts


@app.local_entrypoint()
def main():
    engine = IntensityEngine(
        dataset="monology_pile", model_id="rqae-rqae-round_fsq-cbd4-cbs5-nq1024"
    )
    dataset = Dataset()
    import time

    start = time.time()

    seq = 10000
    tok = 100

    original_sequence = dataset.get_text.remote(seq)[0]
    print(repr(original_sequence[tok]))
    print("\t" + repr("".join(original_sequence[tok - 8 : tok + 8])))
    print()

    for result, layer in engine.find_examples.remote_gen(idx=seq):
        print(f"Layer {layer}")
        for c in result.keys():
            indices = result[c]["indices"]
            intensities = result[c]["intensities"]

            print("=" * 10)
            print(c)
            print(f"\t{indices.shape=}\n\t{intensities.shape=}")
            print("=" * 10)

            for i in range(4):
                top_sequence = dataset.get_text.remote(indices[tok][i].item())[0]
                max_intensity_idx = intensities[tok, i].argmax().item()
                print(
                    max_intensity_idx,
                    repr(top_sequence[max_intensity_idx]),
                    intensities[tok, i].max().item(),
                )
                print("\t" + repr("".join(top_sequence)))
                print()
            print("\n\n")
        print(f"Time taken: {time.time() - start:.3f}s")
        print("=" * 100)
        start = time.time()


@app.function(**DEFAULT_FUNCTION_ARGS)
@modal.asgi_app()
def fastapi_app() -> Callable:
    web_app = FastAPI()

    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, you should specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dataset_instance = Dataset()
    intensity_engine = IntensityEngine()

    @web_app.get("/stream_text")
    async def stream_text(dataset_name: str = "monology_pile"):
        try:
            dataset = Dataset(dataset=dataset_name)
            result = dataset.get_text.remote()[0]  # Get first tuple
            return {
                "text": result[1],  # The text
                "id": result[0],  # The index
                "success": True,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/get_samples")
    async def get_samples(
        idx: int,
        dataset_name: str = "monology_pile",
        layers: str = "4,6,8,12,16,24,32,48,64,128,256,512,1023",
    ):
        async def generate_samples():
            try:
                # Parse layers from comma-separated string to list of integers
                try:
                    layers_list = [int(l.strip()) for l in layers.split(",")]
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid layer format. Expected comma-separated integers.",
                    )

                cache_path = f"/data/cache/{dataset_name}/samples/{idx}.json"
                volume.reload()

                # Check if cache exists and contains all requested layers
                cached_results = None
                missing_layers = layers_list.copy()
                if os.path.exists(cache_path):
                    with open(cache_path, "r") as f:
                        cached_results = json.load(f)
                        # Check which layers we already have
                        cached_layers = {result["layer"] for result in cached_results}
                        missing_layers = [
                            layer for layer in layers_list if layer not in cached_layers
                        ]

                        # Stream cached results first
                        for result in sorted(
                            cached_results,
                            key=lambda x: (
                                layers_list.index(x["layer"])
                                if x["layer"] in layers_list
                                else float("inf")
                            ),
                        ):
                            if result["layer"] in layers_list:
                                yield json.dumps(result) + "\n"

                # Only calculate for missing layers
                if missing_layers:
                    engine = IntensityEngine(dataset=dataset_name)
                    results = []
                    for result, layer in engine.find_examples.remote_gen(
                        idx=idx, layers=missing_layers
                    ):
                        # Convert tensors to Python native types
                        processed_result = {}
                        for category in result:  # top, middle, bottom
                            processed_result[category] = {
                                "indices": result[category]["indices"]
                                .cpu()
                                .numpy()
                                .tolist(),
                                "intensities": result[category]["intensities"]
                                .cpu()
                                .numpy()
                                .tolist(),
                            }
                        result_obj = {"layer": layer, "samples": processed_result}
                        results.append(result_obj)

                        # Stream the result immediately
                        yield json.dumps(result_obj) + "\n"

                    # Cache the new results
                    if cached_results:
                        results.extend(cached_results)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(results, f)
                    volume.commit()

            except Exception as e:
                import traceback

                print(f"Failed at generating samples! {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        return StreamingResponse(
            generate_samples(),
            media_type="application/x-ndjson",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    @web_app.options("/{path:path}")
    async def options_handler(path: str):
        return StreamingResponse(
            iter([""]),
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    @web_app.get("/get_text_by_id")
    async def get_text_by_id(idx: int, dataset_name: str = "monology_pile"):
        try:
            dataset = Dataset(dataset=dataset_name)
            text = dataset.get_text.remote([idx])[0]
            return {
                "text": text[1],  # Skip the first token (BOS)
                "id": text[0],
                "success": True,
            }
        except Exception as e:
            import traceback

            print(f"Failed at getting text by id! {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/check_cache")
    async def check_cache(idx: int, dataset_name: str = "monology_pile"):
        try:
            cache_path = f"/data/cache/{dataset_name}/samples/{idx}.json"
            volume.reload()
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                    return {
                        "exists": True,
                        "layers": sorted(
                            list({result["layer"] for result in cached_data})
                        ),
                    }
            return {"exists": False, "layers": []}
        except Exception as e:
            import traceback

            print(f"Failed at checking cache! {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/get_token_samples")
    async def get_token_samples(
        idx: int,
        token_position: int,
        layer: int,
        dataset_name: str = "monology_pile",
        limit: int = 10,  # Limit number of examples per category
    ):
        try:
            cache_path = f"/data/cache/{dataset_name}/samples/{idx}.json"
            volume.reload()
            if not os.path.exists(cache_path):
                raise HTTPException(status_code=404, detail="Cache not found")

            with open(cache_path, "r") as f:
                cached_data = json.load(f)

            # Find the layer data
            layer_data = next(
                (item for item in cached_data if item["layer"] == layer), None
            )
            if not layer_data:
                raise HTTPException(
                    status_code=404, detail=f"Layer {layer} not found in cache"
                )

            # Get dataset instance to fetch texts
            dataset = Dataset(dataset=dataset_name)

            # Extract samples just for the requested token position
            result = {}
            for category in ["top", "middle", "bottom"]:
                samples = layer_data["samples"][category]
                indices = samples["indices"][token_position][:limit]
                # Get full sequence intensities for each example
                intensities = samples["intensities"][token_position][:limit]

                # Fetch texts for these indices
                texts = dataset.get_text.remote(indices)

                result[category] = {
                    "indices": indices,
                    "intensities": intensities,  # This is now a list of lists, one per token
                    "texts": texts,
                }

            return result
        except Exception as e:
            import traceback

            print(f"Failed at getting token samples! {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/search_text")
    async def search_text(
        query: str, dataset_name: str = "monology_pile", limit: int = 10
    ):
        try:
            dataset = Dataset(dataset=dataset_name)
            matching_texts = dataset.search_texts.remote(query, limit)
            return {"results": matching_texts, "success": True}
        except Exception as e:
            import traceback

            print(f"Failed at searching text! {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return web_app
