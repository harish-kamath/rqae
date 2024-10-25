# Offline work to set up demo

import modal

app = modal.App("rqae-offline")
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


@app.cls(
    gpu="t4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
    timeout=60 * 10,
)
class GemmaModal:
    @modal.build()
    def build(self):
        from rqae.model import RQAE
        from rqae.llm import Gemma2
        from rqae.gemmascope import JumpReLUSAE
        from transformers import AutoTokenizer

        RQAE.from_pretrained("google/gemma-2-2b")
        JumpReLUSAE.from_pretrained("google/gemma-2-2b")
        Gemma2.from_pretrained("google/gemma-2-2b")
        AutoTokenizer.from_pretrained("google/gemma-2-2b")

    @modal.enter()
    def load(self):
        import torch

        self.llm = (
            Gemma2.from_pretrained("google/gemma-2-2b")
            .eval()
            .to(torch.float16)
            .to("cuda")
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        # Don't load them to GPU, because we don't know which to use, and they're small anyways
        self.rqae = RQAE.from_pretrained("google/gemma-2-2b").eval()
        self.gemmascope = JumpReLUSAE.from_pretrained("google/gemma-2-2b").eval()

    @modal.method()
    def name(self):
        return self.model.name

    @modal.method()
    def batch_run(
        self,
        texts=None,
        tokens=None,
        methods=["raw", "rqae", "gemmascope"],
        save_folder=None,
        save_name=None,
    ):
        import torch
        from tqdm import tqdm
        import os

        if tokens is not None and texts is not None:
            raise ValueError("Cannot provide both texts and tokens")
        if tokens is None and texts is None:
            raise ValueError("Must provide either tokens or texts")
        if texts:
            tokens = self.tokenizer(texts, return_tensors="pt").to("cuda")
            # Ensure that no padding was needed, as it's not supported yet
            assert (
                tokens.attention_mask.all()
            ), "Padding was needed to tokenize texts, but is not supported yet"
            tokens = tokens.input_ids

        if "rqae" in methods:
            self.rqae = self.rqae.to("cuda")
        if "gemmascope" in methods:
            self.gemmascope = self.gemmascope.to("cuda")

        batch_size = 4
        return_values = []

        for b in tqdm(
            range(0, len(tokens), batch_size), desc="Batch Running Activations"
        ):
            batch_tokens = tokens[b : b + batch_size]
            inputs = {
                "input_ids": batch_tokens.to("cuda"),
                "attention_mask": torch.ones_like(batch_tokens).to("cuda"),
            }

            # First, original model
            return_value = {}
            if "raw" in methods:
                with torch.inference_mode():
                    raw_output = self.llm(**inputs, labels=batch_tokens)
                return_value["original_ce"] = raw_output.loss.item()

            if "rqae" in methods:
                self.llm.clear_hooks()
                rqae_val = {}
                store = lambda name, value: rqae_val.update({name: value})
                rqae_hook = self.rqae.hook(llm=self.llm, store=store)
                self.llm.hook(rqae_hook)
                with torch.inference_mode():
                    rqae_output = self.llm(**inputs, labels=batch_tokens)
                return_value["rqae_ce"] = rqae_output.loss.item()
                return_value["original_activations"] = (
                    rqae_val["original"].detach().cpu().to(torch.float16)
                )
                return_value["rqae_indices"] = (
                    rqae_val["indices"].detach().cpu().to(torch.int16)
                )

            if "gemmascope" in methods:
                self.llm.clear_hooks()
                gemmascope_val = {}
                store = lambda name, value: gemmascope_val.update({name: value})
                gemmascope_hook = self.gemmascope.hook(llm=self.llm, store=store)
                self.llm.hook(gemmascope_hook)
                with torch.inference_mode():
                    gemmascope_output = self.llm(**inputs, labels=batch_tokens)
                return_value["gemmascope_ce"] = gemmascope_output.loss.item()
                gs_intensities = (
                    gemmascope_val["intensities"].detach().cpu().to(torch.float16)
                )
                gs_indices = (
                    gs_intensities.nonzero()
                )  # num_tokens, 3 (sequence, token, feature)
                return_value["gemmascope_intensities"] = gs_intensities[
                    gs_indices[:, 0], gs_indices[:, 1], gs_indices[:, 2]
                ]
                # convert batch index into dataset index
                gs_indices[:, 0] = gs_indices[:, 0] + b
                return_value["gemmascope_indices"] = gs_indices.to(torch.int32)

            return_values.append(return_value)
        return_values = {
            "original_ce": torch.tensor(
                [v.get("original_ce", 0) for v in return_values]
            ),
            "rqae_ce": torch.tensor([v.get("rqae_ce", 0) for v in return_values]),
            "gemmascope_ce": torch.tensor(
                [v.get("gemmascope_ce", 0) for v in return_values]
            ),
            "original_activations": torch.cat(
                [
                    v.get("original_activations", torch.tensor([0]))
                    for v in return_values
                ]
            ),
            "rqae_indices": torch.cat(
                [v.get("rqae_indices", torch.tensor([0])) for v in return_values]
            ),
            "gemmascope_intensities": torch.cat(
                [
                    v.get("gemmascope_intensities", torch.tensor([0]))
                    for v in return_values
                ]
            ),
            "gemmascope_indices": torch.cat(
                [v.get("gemmascope_indices", torch.tensor([0])) for v in return_values]
            ),
        }
        self.rqae = self.rqae.cpu()
        self.gemmascope = self.gemmascope.cpu()
        self.llm.clear_hooks()

        if save_folder and save_name:
            os.makedirs(save_folder, exist_ok=True)
            if "raw" in methods:
                os.makedirs(os.path.join(save_folder, "raw"), exist_ok=True)
                torch.save(
                    return_values["original_ce"],
                    os.path.join(save_folder, "raw", f"{save_name}_ce.pt"),
                )
                torch.save(
                    return_values["original_activations"],
                    os.path.join(save_folder, "raw", f"{save_name}.pt"),
                )
            if "rqae" in methods:
                rqae_folder = os.path.join(save_folder, self.rqae.name)
                os.makedirs(rqae_folder, exist_ok=True)
                torch.save(
                    return_values["rqae_ce"],
                    os.path.join(rqae_folder, f"{save_name}_ce.pt"),
                )
                torch.save(
                    return_values["rqae_indices"],
                    os.path.join(rqae_folder, f"{save_name}.pt"),
                )
            if "gemmascope" in methods:
                gs_folder = os.path.join(save_folder, self.gemmascope.name)
                os.makedirs(gs_folder, exist_ok=True)
                torch.save(
                    return_values["gemmascope_ce"],
                    os.path.join(gs_folder, f"{save_name}_ce.pt"),
                )
                torch.save(
                    return_values["gemmascope_intensities"],
                    os.path.join(gs_folder, f"{save_name}.pt"),
                )
                torch.save(
                    return_values["gemmascope_indices"],
                    os.path.join(gs_folder, f"{save_name}_indices.pt"),
                )
            return
        return return_values


@app.cls(
    gpu="t4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
    timeout=60 * 10,
)
class RQAEModal:
    @modal.build()
    def build(self):
        from rqae.model import RQAE

        RQAE.from_pretrained("google/gemma-2-2b")

    @modal.enter()
    def load(self):
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        self.model = (
            RQAE.from_pretrained("google/gemma-2-2b")
            .eval()
            .to(torch.float16)
            .to("cuda")
        )

    @modal.method()
    def name(self):
        return self.model.name

    @modal.method()
    def batch_decode(
        self,
        tokens,
        layers=None,
    ):
        import torch

        with torch.inference_mode():
            return self.model.decode(tokens.to("cuda"), layers).cpu()


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 30)
def save_monology_activations(methods=["raw", "rqae", "gemmascope"]):
    import torch
    import os

    gemma = GemmaModal()
    retvals = []
    tokens = torch.load("/data/datasets/monology_pile/tokens.pt")
    batch_size = 1024
    os.makedirs("/data/datasets/monology_pile/activations", exist_ok=True)

    for i in range(0, len(tokens), batch_size):
        save_folder = f"/data/datasets/monology_pile/activations/"
        save_name = f"{i // batch_size:06d}"
        retvals.append(
            gemma.batch_run.spawn(
                tokens=tokens[i : i + batch_size],
                methods=methods,
                save_folder=save_folder,
                save_name=save_name,
            )
        )
    for retval in retvals:
        retval.get()


def format_text(texts, token):
    sequence, token = token
    up_to = texts[sequence][:token]
    after = texts[sequence][token + 1 :]
    print(
        "".join(up_to)
        + "<RQAE_TOKEN>"
        + texts[sequence][token]
        + "</RQAE_TOKEN>"
        + "".join(after)
    )


@app.function(volumes={"/data": volume}, image=image)
def show_text(sequence_token_pairs):
    import json

    with open("/data/datasets/monology_pile/text.json", "r") as f:
        texts = json.load(f)
    for sequence, token in sequence_token_pairs:
        print(f"({sequence}, {token})")
        print(format_text(texts, (sequence, token)))
        print("\n" + "=" * 100 + "\n")


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 10)
def show_rank_helper(token, path, method="raw", layers=None):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    s, c = token
    if method == "raw":
        act_path = torch.load(
            f"/data/datasets/monology_pile/activations/raw/{s // 1024:06d}.pt"
        )
        activation = act_path[s % 1024, c].clone().unsqueeze(0).unsqueeze(0)
        del act_path
    elif method == "rqae_cossim":
        model = RQAEModal()
        act_path = torch.load(
            f"/data/datasets/monology_pile/activations/{model.name.remote()}/{s // 1024:06d}.pt"
        )
        activation = act_path[s % 1024, c].clone().int()
        del act_path
        activation = model.batch_decode.remote(
            activation.unsqueeze(0).unsqueeze(0), layers
        )
    elif method == "rqae":
        from rqae.model import RQAE

        model = RQAE.from_pretrained("google/gemma-2-2b").eval()
        codebook = model.codebook[0].data.cpu().detach().clone()
        codebook_sims = F.normalize(codebook, dim=-1) @ F.normalize(codebook, dim=-1).T
        layer_norms = torch.tensor(
            [l[1].weight.data.norm(dim=0).mean().item() for l in model.layers]
        )
        act_path = torch.load(
            f"/data/datasets/monology_pile/activations/{model.name}/{s // 1024:06d}.pt"
        )
        activation = act_path[s % 1024, c].clone().int()
        layer_mask = torch.ones(act_path.shape[2], dtype=torch.bool)
        if layers is not None:
            layer_mask[:] = False
            layer_mask[layers] = True
        del act_path
        del model

    activations = torch.load(path)
    scores = torch.zeros((activations.shape[0], activations.shape[1]))
    batch_size = {
        "raw": 512,
        "rqae_cossim": 128,
        "rqae": 128,
    }[method]
    for i in tqdm(range(0, len(activations), batch_size), desc="Calculating scores"):
        batch = activations[i : i + batch_size]
        if method == "raw":
            raw_activation = batch
            scores[i : i + batch_size] = torch.nn.functional.cosine_similarity(
                activation.cpu(), raw_activation.cpu(), dim=-1
            )
        elif method == "rqae_cossim":
            raw_activation = model.batch_decode.remote(batch.int(), layers)
            scores[i : i + batch_size] = torch.nn.functional.cosine_similarity(
                activation.cpu(), raw_activation.cpu(), dim=-1
            )
        elif method == "rqae":
            sims = codebook_sims[activation, batch.int()]
            weighted_sims = sims * layer_mask * layer_norms
            scores[i : i + batch_size] = weighted_sims.sum(dim=-1)

    print(f"Calculated {scores.shape} scores for {path}")
    return scores


@app.function(volumes={"/data": volume}, image=image)
def show_rank(token, method="raw", rank=50, **kwargs):
    import torch
    import json
    import os
    import multiprocessing

    s, c = token
    with open("/data/datasets/monology_pile/text.json", "r") as f:
        texts = json.load(f)

    if method == "raw":
        act_folder = "/data/datasets/monology_pile/activations/raw/"
        act_files = [
            os.path.join(act_folder, f)
            for f in os.listdir(act_folder)
            if "_ce" not in f
        ]
        scores = show_rank_helper.starmap(
            [((s, c), act_file, method) for act_file in act_files]
        )
    elif method in ["rqae_cossim", "rqae"]:
        model = RQAE.from_pretrained("google/gemma-2-2b").eval()
        act_folder = f"/data/datasets/monology_pile/activations/{model.name}/"
        act_files = [
            os.path.join(act_folder, f)
            for f in os.listdir(act_folder)
            if "_ce" not in f
        ]
        del model
        scores = show_rank_helper.starmap(
            [
                ((s, c), act_file, method, kwargs.get("layers", None))
                for act_file in act_files
            ]
        )
    scores = torch.cat(list(scores))
    print(f"Calculated {scores.shape} scores, now getting top rank")
    top_scores, top_indices = scores.view(-1).topk(rank)
    top_indices = [
        (idx.item() // scores.shape[1], idx.item() % scores.shape[1])
        for idx in top_indices
    ]
    print(scores.shape, top_scores.shape, len(top_indices))

    print(f"Showing top {rank} activations for ({s}, {c})")
    print(format_text(texts, (s, c)))
    for i in range(rank):
        print(i)
        print(top_scores[i].item())
        print(top_indices[i])
        print(format_text(texts, top_indices[i]))
        print("\n" + "=" * 100 + "\n")


@app.local_entrypoint()
def main():
    save_monology_activations.remote(methods=["gemmascope"])
    # import random

    # random_seqtoks = [
    #     [random.randint(0, 36864), random.randint(1, 127)] for _ in range(100)
    # ]
    # show_text.remote(random_seqtoks)
    # token = (19210, 56)
    # show_rank.remote(token, method="raw", rank=20)
    # show_rank.remote(token, method="rqae", rank=20, layers=range(8))
