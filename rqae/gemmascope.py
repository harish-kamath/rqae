import torch
import torch.nn as nn
import numpy as np
from typing import Union
from huggingface_hub import HfApi, hf_hub_download


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae, name=""):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.name = name

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    def hook(self, **kwargs):
        store = kwargs.get("store", lambda x, y: None)
        skip_bos = kwargs.get("skip_bos", True)
        replace = kwargs.get("replace", True)

        def hook_fn(module, input, output):
            hs = output[0].float()  # (B, S, dim)
            store("original", hs.detach().clone())
            acts = self.encode(hs)
            store("intensities", acts.detach().clone())
            recon = self.decode(acts)
            if skip_bos:
                recon[:, 0] = hs[:, 0]
            store("new", recon.detach().clone())
            if replace:
                output[0].data.copy_(recon)

        return hook_fn

    @classmethod
    def from_pretrained(
        cls,
        model: str = "google/gemma-2-2b",
        layer_type: str = "res",
        layer: Union[str, int] = "half",
        width: int = 16,
        l0: int = 82,
        find_closest_l0: bool = False,
    ):
        # Hard-coded
        repos = {
            "google/gemma-2-2b": {
                "res": "google/gemma-scope-2b-pt-res",
                "mlp": "google/gemma-scope-2b-pt-mlp",
                "att": "google/gemma-scope-2b-pt-att",
            },
            "google/gemma-2-9b": {
                "res": "google/gemma-scope-9b-pt-res",
                "mlp": "google/gemma-scope-9b-pt-mlp",
                "att": "google/gemma-scope-9b-pt-att",
            },
            "google/gemma-2-27b": {
                "res": "google/gemma-scope-27b-pt-res",
            },
            "google/gemma-2-9b-it": {
                "res": "google/gemma-scope-9b-it-res",
            },
        }
        num_layers = {
            "google/gemma-2-2b": 26,
            "google/gemma-2-9b": 42,
            "google/gemma-2-27b": 46,
            "google/gemma-2-9b-it": 42,
        }

        repo = repos[model][layer_type]

        if layer == "half":
            layer = num_layers[model] // 2 - 1  # -1 because layer indices are 0-indexed

        if width >= 1000:
            width = f"{width // 1000}m"
        else:
            width = f"{width}k"

        path = f"layer_{layer}/width_{width}/average_l0_"

        if find_closest_l0:
            all_files = HfApi().list_repo_files(repo)
            all_files = [f for f in all_files if f.startswith(path)]
            l0s = [int(f.replace(path, "").split("/")[0]) for f in all_files]
            l0 = min(l0s, key=lambda x: abs(x - l0))
            path += f"{l0}/"
        else:
            path += f"{l0}/"

        path += "params.npz"

        file_path = hf_hub_download(repo, path)
        params = np.load(file_path)
        params = {k: torch.from_numpy(v) for k, v in params.items()}

        d_model = params["W_enc"].shape[0]
        d_sae = params["W_enc"].shape[1]

        name = f"gemmascope-{model.split('/')[-1]}-{layer_type}-{layer}-w{width}-l{l0}"

        model = cls(d_model, d_sae, name)
        model.load_state_dict(params)
        return model


if __name__ == "__main__":
    model = JumpReLUSAE.from_pretrained()
    import pdb

    pdb.set_trace()
