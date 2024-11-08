import torch
import numpy as np
from scipy.stats import vonmises_fisher
import math
from typing import Optional, Tuple, Union, List
from rqae.model import RQAE


class Feature:
    def __init__(
        self,
        id: str = "",
        explanation: str = "",
        scores: dict = {},
        model: str = "",
        activations: list = [],
    ):
        self.id = str(id)
        self.explanation = str(explanation)
        self.scores = scores
        self.model = str(model)

        self.activations = (
            activations  # list of {"text": [str], "activations": [float]}
        )

    def save(self, file_path: str):
        np.savez(file_path, **self.__dict__)

    @classmethod
    def load(cls, file_path: str):
        params = dict(np.load(file_path, allow_pickle=True))
        for k, v in params.items():
            try:
                params[k] = v.item()
            except Exception as e:
                # print(f"Key {k} not converted to item: {e}")
                pass
        return cls(**params)


class RQAEFeature:
    def __init__(
        self,
        id: str = "",
        explanations: List[str] = None,
        scores: List[dict] = None,
        activations: list = None,
        model: str = "",
        num_quantizers: int = 1024,
        dim: int = 4,
        layers: List[int] = None,
        layer_weights: List[float] = None,
        center: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.num_quantizers = num_quantizers
        self.dim = dim
        self.model = model
        self.id = id

        if layers is None:
            layers = [num_quantizers - 1]
        if layer_weights is None:
            layer_weights = np.ones(num_quantizers)
        if center is None:
            center = np.zeros((num_quantizers,))

        self.layers = layers
        self.layer_weights = torch.tensor(layer_weights)
        self.center = torch.tensor(center).int()

        if explanations is None:
            explanations = ["" for _ in layers]
        if scores is None:
            scores = [{} for _ in layers]
        if activations is None:
            activations = {k: [] for k in layers}

        self.explanations = explanations
        self.scores = scores
        self.activations = activations

        self.rqae = None

    def to_feature(self, layer: int = 0):
        return Feature(
            id=self.id,
            model=self.model,
            explanation=self.explanations[layer],
            scores=self.scores[layer],
            activations=self.activations[self.layers[layer]],
        )

    def load_model(self, rqae):
        self.rqae = rqae
        self.layer_weights = torch.tensor(
            [l[1].weight.data.norm(dim=0).mean().item() for l in rqae.layers]
        ).to(torch.float16)
        return self

    def intensity(self, token_indices: torch.Tensor, layers=None):
        """
        Calculates intensity of a feature for a given set of token indices.

        Args:
            token_indices: (..., num_quantizers) tensor of token indices
            layers: List of layers to calculate intensity for. If None, use all layers.

        Returns:
            intensity: (..., ) tensor of intensities
        """
        if layers is None:
            layers = self.layers

        if self.rqae is None:
            raise ValueError("Model not loaded. Needed for intensity calculation.")

        max_layer = max(layers) + 1
        codebook_sims = self.rqae.codebook_sims

        sims = codebook_sims[
            self.center[:max_layer].int(), token_indices[..., :max_layer].int()
        ]
        sims *= self.layer_weights[:max_layer]
        sims = sims.cumsum(dim=-1)
        sims /= self.layer_weights[:max_layer].cumsum(dim=-1)
        sims = sims[..., layers]
        return sims

    @classmethod
    def from_quantizer(cls, quantizer: RQAE, **kwargs):
        return cls(
            num_quantizers=quantizer.num_quantizers,
            dim=quantizer.codebook_dim,
            **kwargs,
        ).load_model(quantizer)

    def save(self, file_path: str):
        np.savez(file_path, **{k: v for k, v in self.__dict__.items() if k != "rqae"})

    @classmethod
    def load(cls, file_path: str):
        params = dict(np.load(file_path, allow_pickle=True))
        for k, v in params.items():
            try:
                if k == "explanations":
                    params[k] = [str(e) for e in v]
                params[k] = v.item()
            except Exception as e:
                # print(f"Key {k} not converted to item: {e}")
                pass
        return cls(**params)
