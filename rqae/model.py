import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from huggingface_hub import hf_hub_download
import os
import json
from safetensors import safe_open


class RQAE(nn.Module):
    PRETRAINED = {
        "google/gemma-2-2b": "harish-kamath/rqae/gemma-2-2b",
        "rqae-rqae-round_fsq-cbd4-cbs5-nq1024": "harish-kamath/rqae/gemma-2-2b",
    }

    def __init__(
        self,
        dim: int = 2304,
        codebook_dim: int = 4,
        codebook_size: int = 5,
        num_quantizers: int = 1024,
        quantization_method: str = "round_fsq",
        name: str = "",
        **kwargs,
    ):
        super().__init__()
        layers = []
        for layer_idx in range(num_quantizers):
            layer = [
                torch.nn.Linear(dim, codebook_dim),
                torch.nn.Linear(codebook_dim, dim),
            ]
            layers.append(torch.nn.ModuleList(layer))
        self.layers = torch.nn.ModuleList(layers)
        if quantization_method in ["fsq", "round_fsq"]:
            self.codebook = torch.nn.Parameter(
                torch.randn(num_quantizers, codebook_size**codebook_dim, codebook_dim)
            )
            self.register_buffer(
                "codebook_counts",
                torch.zeros(num_quantizers, codebook_size**codebook_dim),
                persistent=True,
            )
        else:
            self.codebook = torch.nn.Parameter(
                torch.randn(num_quantizers, codebook_size, codebook_dim)
            )
            self.register_buffer(
                "codebook_counts",
                torch.zeros(num_quantizers, codebook_size),
                persistent=True,
            )

        self.quantization_method = quantization_method
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.name = name
        self.normalize_codebooks()

        if self.quantization_method in ["fsq", "round_fsq"]:
            codebook = np.linspace(-1, 1, self.codebook_size)
            codebook = list(product(codebook, repeat=self.codebook_dim))
            codebook = np.array(codebook)
            if self.quantization_method == "round_fsq":
                norms = np.linalg.norm(codebook, axis=-1, keepdims=True)
                # Avoid division by zero for all-zero vectors
                norms = np.where(norms == 0, 1.0, norms)
                codebook = np.divide(codebook, norms, where=norms != 0)
            self.codebook.data.copy_(torch.from_numpy(codebook))
            self.codebook.requires_grad = False

    @classmethod
    def from_pretrained(cls, model_name: str):
        if model_name in cls.PRETRAINED:
            model_name = cls.PRETRAINED[model_name]
        username, reponame, *rest = model_name.split("/")
        folder_name = "/".join(rest)
        model_path = os.path.join(folder_name, "model.safetensors")
        config_path = os.path.join(folder_name, "config.json")
        model_path = hf_hub_download(f"{username}/{reponame}", model_path)
        config_path = hf_hub_download(f"{username}/{reponame}", config_path)
        with open(config_path, "r") as f:
            params = json.load(f)
        name = f"rqae-{reponame}-{params['quantization_method']}-cbd{params['codebook_dim']}-cbs{params['codebook_size']}-nq{params['num_quantizers']}"
        model = cls(**params, name=name)
        if model_path.endswith(".safetensors"):
            state_dict = {}
            with safe_open(model_path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            model.load_state_dict(state_dict, strict=True)
        elif model_path.endswith(".pt"):
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            raise ValueError(f"Unknown file extension for {model_path}")
        return model

    def update_codebook_counts(self, indices):
        if not self.training:
            return
        return  # Use if you want, only used to monitor training (EMA for codebook usage)

        # indices is (B, S, num_quantizers). Flatten to (B * S, num_quantizers)
        flat_indices = indices.view(-1, indices.size(-1))

        # Count the number of times each codebook is used, per quantizer
        B, S = indices.size(0), indices.size(1)
        codebook_size = self.params["codebook_size"]
        num_quantizers = indices.size(-1)

        new_counts = torch.zeros(num_quantizers, codebook_size, device=indices.device)
        new_counts.scatter_add_(
            1, flat_indices.T, torch.ones_like(flat_indices.T, dtype=torch.float)
        )

        # Normalize counts by dividing by B * S
        new_counts /= B * S

        # Update the codebook with EMA based on codebook size
        decay = 1.0 / codebook_size
        self.codebook_counts.mul_(1 - decay).add_(new_counts, alpha=decay)

    def normalize_codebooks(self):
        if self.quantization_method in ["fsq", "round_fsq"]:
            return
        self.codebook.data.copy_(
            self.codebook.data / self.codebook.data.norm(dim=-1, keepdim=True)
        )

    @property
    def codebook_sims(self):
        # only works for round_fsq for now
        if hasattr(self, "_codebook_sims"):
            return self._codebook_sims
        if not self.quantization_method == "round_fsq":
            raise ValueError("Codebook sims only supported for round_fsq for now")
        codebook = self.codebook.data.detach().clone()[0]
        codebook_sims = F.normalize(codebook, dim=-1) @ F.normalize(codebook, dim=-1).T
        self._codebook_sims = codebook_sims.to(torch.float16)
        return self._codebook_sims

    @property
    def subfeatures(self):
        if hasattr(self, "_subfeatures"):
            return self._subfeatures
        self._subfeatures = []
        for layer_idx in range(self.num_quantizers):
            layer_codebook = self.codebook[layer_idx]
            lin_out = self.layers[layer_idx][1]
            subfeature = lin_out(layer_codebook)
            self._subfeatures.append(subfeature)
        self._subfeatures = torch.stack(self._subfeatures)
        return self._subfeatures  # (num_quantizers, codebook_size, dim)

    @property
    def subfeature_sims(self):
        if hasattr(self, "_subfeature_sims"):
            return self._subfeature_sims
        normalized_subfeatures = F.normalize(self.subfeatures, dim=-1)
        # (num_quantizers, codebook_size, codebook_size)
        self._subfeature_sims = (
            normalized_subfeatures @ normalized_subfeatures.transpose(-1, -2)
        )
        self._subfeature_sims = self._subfeature_sims.to(torch.float16)
        return self._subfeature_sims

    @property
    def layer_norms(self):
        if hasattr(self, "_layer_norms"):
            return self._layer_norms
        self._layer_norms = torch.tensor(
            [l[1].weight.data.norm(dim=0).mean().item() for l in self.layers],
            device=self.layers[0][1].weight.device,
        )
        return self._layer_norms

    def gumbel_sample(self, cos_sim, temperature=0.0):
        if temperature < 1e-7 or not self.training:
            return cos_sim.argmax(dim=-1)
        return torch.nn.functional.gumbel_softmax(
            cos_sim, tau=temperature, hard=True
        ).argmax(dim=-1)

    def quantize_gumbel(self, x, layer_idx, temperature=0.0):
        x = x / x.norm(dim=-1, keepdim=True)  # (B, S, codebook_dim)
        layer_codebook = self.codebook[layer_idx]  # (codebook_size, codebook_dim)
        cos_sim = torch.matmul(x, layer_codebook.T)  # (B, S, codebook_size)
        indices = self.gumbel_sample(cos_sim, temperature=temperature)  # (B, S)
        closest_codebook_vectors = layer_codebook[indices]  # (B, S, codebook_dim)
        return closest_codebook_vectors, indices

    def quantize(self, x, layer_idx, temperature=0.0):
        self.normalize_codebooks()
        return self.quantize_gumbel(x, layer_idx, temperature)

    def forward(self, x, max_layers: int = float("inf"), temperature=0.0):
        residual = x  # (B, S, dim)
        quantized_out = 0
        all_indices = []

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx >= max_layers:
                break

            lin_in = layer[0]
            lin_out = layer[1]

            quantized = lin_in(residual)  # (B, S, codebook_dim)

            closest_codebook_vectors, indices = self.quantize(
                quantized, layer_idx, temperature
            )
            all_indices.append(indices)
            # Straight through estimator
            closest_codebook_vectors = (
                quantized + (closest_codebook_vectors - quantized).detach()
            )
            quantized = lin_out(closest_codebook_vectors)  # (B, S, dim)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

        all_indices = torch.stack(all_indices).permute(1, 2, 0)
        self.update_codebook_counts(all_indices)

        # (B, S, dim), (B, S, num_quantizers)
        return quantized_out, all_indices

    def indices_to_codebook_values(self, indices):
        B, S, NQ = indices.shape
        return self.codebook[0][indices]

    def decode_from_codebook_values(self, codebook_values, layers=None):
        B, S, NQ, CD = codebook_values.shape
        quantized = None
        for layer_idx, layer in enumerate(self.layers):
            if layers is not None and layer_idx not in layers:
                continue
            lin_out = layer[1]
            residual = lin_out(codebook_values[:, :, layer_idx])
            if quantized is None:
                quantized = residual
            else:
                quantized += residual
        return quantized

    def decode(self, indices, layers=None):
        codebook_values = self.indices_to_codebook_values(indices)
        return self.decode_from_codebook_values(codebook_values, layers)

    def hook(self, **kwargs):
        if "llm" in kwargs:
            llm = kwargs.pop("llm")
            assert hasattr(llm, "norm") and hasattr(
                llm, "denorm"
            ), "RQAE hook requires norm and denorm from LLM"
            return self.hook(norm=llm.norm, denorm=llm.denorm, **kwargs)

        store = kwargs.get("store", lambda x, y: None)
        skip_bos = kwargs.get("skip_bos", True)
        replace = kwargs.get("replace", True)

        if "norm" in kwargs:
            norm = kwargs.pop("norm")
        else:
            raise ValueError("RQAE hook requires norm from LLM")

        if "denorm" in kwargs:
            denorm = kwargs.pop("denorm")
        else:
            raise ValueError("RQAE hook requires denorm from LLM")

        def hook_fn(module, input, output):
            hs = output[0].float()  # (B, S, dim)
            store("original", hs.detach().clone())
            rms_hs = norm(hs)  # (B, S, dim)
            store("normed", rms_hs.detach().clone())
            q_out, indices = self(rms_hs)  # (B, S, dim), (B, S, num_quantizers)
            store("quantized", q_out.detach().clone())
            store("indices", indices.detach().clone())
            q_out = denorm(q_out, hs)  # (B, S, dim)
            if skip_bos:
                q_out[:, 0] = hs[:, 0]
            store("new", q_out.detach().clone())
            if replace:
                output[0].data.copy_(q_out)

        return hook_fn


if __name__ == "__main__":
    model = RQAE.from_pretrained("google/gemma-2-2b")
    import pdb

    pdb.set_trace()
