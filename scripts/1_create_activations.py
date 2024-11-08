"""
After uploading the dataset, you need to calculate activations for it.
This includes the raw LLM activations, as well as the activations for RQAE and Gemmascope.

This script works on monology and the specific models used in the report, but it should be straightforward
to modify this for your own models/datasets (just change the IDs).

To work on another dataset, just update the save_monology_activations function.
To work on other models, you need to update more of the ActivationsModal class.
"""

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


def run_download():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from transformers import AutoTokenizer
    from rqae.gemmascope import JumpReLUSAE

    RQAE.from_pretrained("google/gemma-2-2b")
    Gemma2.from_pretrained("google/gemma-2-2b")
    AutoTokenizer.from_pretrained("google/gemma-2-2b")

    # Download all SAEs, to avoid redownloading all the time

    # Sweep on L0
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l22")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l41")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l82")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l176")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l445")

    # Sweep on width
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w32k-l76")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w65k-l72")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w262k-l67")
    JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w524k-l65")


@app.cls(
    gpu="t4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
    timeout=60 * 10,
)
class ActivationsModal:
    """
    This class is used to create activations for a given dataset, for all models (RQAE, Gemmascope, and raw LLM activations)
    Currently, it only works with Gemma 2 2B. Also, we preload the models used in the report (e.g. the ones for sweeps)
    Of course, you can modify this to work with other models (e.g. 9B).
    """

    @modal.build()
    def build(self):
        from rqae.model import RQAE
        from rqae.llm import Gemma2
        from transformers import AutoTokenizer
        from rqae.gemmascope import JumpReLUSAE

        RQAE.from_pretrained("google/gemma-2-2b")
        Gemma2.from_pretrained("google/gemma-2-2b")
        AutoTokenizer.from_pretrained("google/gemma-2-2b")

        # Download all SAEs, to avoid redownloading all the time

        # Sweep on L0
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l22")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l41")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l82")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l176")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w16k-l445")

        # Sweep on width
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w32k-l76")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w65k-l72")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w262k-l67")
        JumpReLUSAE.from_pretrained("gemmascope-gemma-2-2b-res-12-w524k-l65")

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

    @modal.method()
    def batch_run(
        self,
        texts=None,
        tokens=None,
        methods=["raw", "rqae", "gemmascope"],
        model_id: str = "google/gemma-2-2b",
        max_features: int = None,  # Use with gemmascope, if you only want a subset of features (e.g. not 1M features lol)
        save_name: str = "ACTS",
        save_folder: str = "DFOLDER",
    ):
        import torch
        from tqdm import tqdm
        import os
        import gc

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
            rqae = RQAE.from_pretrained(model_id).to("cuda")
        if "gemmascope" in methods:
            gemmascope = JumpReLUSAE.from_pretrained(model_id)
            if max_features is not None:
                gemmascope.crop(max_features)
                gc.collect()
            gemmascope = gemmascope.to("cuda")

        batch_size = 4  # Since we use T4's
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
                self.llm.clear_hooks()
                with torch.inference_mode():
                    raw_output = self.llm(**inputs, labels=batch_tokens)
                return_value["original_ce"] = raw_output.loss.item()

            if "rqae" in methods:
                self.llm.clear_hooks()
                rqae_val = {}
                store = lambda name, value: rqae_val.update({name: value})
                rqae_hook = rqae.hook(llm=self.llm, store=store)
                self.llm.hook(rqae_hook)
                with torch.inference_mode():
                    rqae_output = self.llm(**inputs, labels=batch_tokens)
                return_value["rqae_ce"] = rqae_output.loss.item()
                return_value["original_activations"] = (
                    rqae_val["original"].detach().cpu().to(torch.float16)
                )
                return_value["rqae_indices"] = (
                    rqae_val["indices"].detach().cpu().to(torch.int32)
                )

            if "gemmascope" in methods:
                self.llm.clear_hooks()
                gemmascope_val = {}
                store = lambda name, value: gemmascope_val.update({name: value})
                gemmascope_hook = gemmascope.hook(llm=self.llm, store=store)
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
        self.llm.clear_hooks()

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
            rqae_folder = os.path.join(save_folder, rqae.name)
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
            gs_folder = os.path.join(save_folder, gemmascope.name)
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
        import time

        time.sleep(10)
        return


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 30)
def save_monology_activations(**kwargs):
    import torch
    import os

    gemma = ActivationsModal()
    retvals = []
    tokens = torch.load("/data/datasets/monology_pile/tokens.pt")
    batch_size = 1024  # Set for the rest of all scripts
    os.makedirs("/data/datasets/monology_pile/activations", exist_ok=True)

    for i in range(0, len(tokens), batch_size):
        save_name = f"{i // batch_size:06d}"
        retvals.append(
            gemma.batch_run.spawn(
                tokens=tokens[i : i + batch_size],
                save_name=save_name,
                save_folder="/data/datasets/monology_pile/activations/",
                **kwargs,
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


@app.local_entrypoint()
def main():
    # All methods used for the report

    # Default

    # save_monology_activations.remote(methods=["rqae", "gemmascope", "raw"])

    # All sweeps max out at 1024 features for efficiency (JumpReLU does not require all features)
    # This means CE is useless

    # Sweep on L0
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w16k-l22",
    #     max_features=1024,
    # )

    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w16k-l41",
    #     max_features=1024,
    # )
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w16k-l82",
    #     max_features=1024,
    # )
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w16k-l176",
    #     max_features=1024,
    # )
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w16k-l445",
    #     max_features=1024,
    # )

    # Sweep on width
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w32k-l76",
    #     max_features=1024,
    # )
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w65k-l72",
    #     max_features=1024,
    # )
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w262k-l67",
    #     max_features=1024,
    # )
    # save_monology_activations.remote(
    #     methods=["gemmascope"],
    #     model_id="gemmascope-gemma-2-2b-res-12-w524k-l65",
    #     max_features=1024,
    # )
