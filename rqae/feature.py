import torch
import numpy as np
from scipy.stats import vonmises_fisher
import math
from typing import Optional, Tuple, Union
from rqae.model import RQAE


class VonMisesFisher4D:
    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        kappa: Optional[float] = None,
        dim: int = None,
    ):
        """
        Initialize a 4D von Mises-Fisher distribution

        Args:
        mu: Optional[np.ndarray] of shape [4], the mean direction
        kappa: Optional[float], concentration parameter

        If mu and kappa are not provided, they will be randomly initialized
        """
        self.mu = mu
        self.kappa = kappa

    @property
    def dim(self):
        if self.mu is None:
            return None
        return self.mu.shape[0]

    def _safe_normalize(self, v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Safely normalize a vector"""
        norm = np.linalg.norm(v)
        return v / (norm + eps) if norm > 0 else np.ones_like(v) / math.sqrt(self.dim)

    def fit(self, points: torch.Tensor) -> None:
        """
        Fit the vMF distribution to the given points using MLE

        Args:
        points: torch.Tensor of shape [B, 4], points on the 4D unit sphere

        """
        # Ensure points are normalized
        points = self._safe_normalize(points)

        mu, kappa = vonmises_fisher.fit(points.cpu().numpy())
        self.mu = torch.tensor(mu, device=points.device)
        self.kappa = kappa

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.kappa is None:
            raise ValueError("Mu and kappa must be initialized before computing pdf")
        return vonmises_fisher.pdf(x.cpu().numpy(), self.mu, self.kappa)

    def steer(self, points: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        """
        Move points closer to being in distribution

        Args:
        points: torch.Tensor of shape [..., 4], points to steer
        strength: float, controls how much to move the points (0 to 1)

        Returns:
        steered_points: torch.Tensor of shape [..., 4], steered points
        """
        # Ensure points are normalized
        points = self._safe_normalize(points)

        # Calculate effective_kappa using an exponential curve
        # The curve is controlled by a parameter 'curve_factor'
        # When curve_factor = 1, it's linear. Higher values create a steeper curve.
        curve_factor = 2.0  # Adjust this value to control the curve
        effective_kappa = (
            self.kappa
            * (1 - math.exp(-curve_factor * strength))
            / (1 - math.exp(-curve_factor))
        )

        mu = torch.from_numpy(self.mu).to(points.device)

        # Calculate the dot product between points and mu
        dot_product = torch.sum(points * mu, dim=-1, keepdim=True)

        # Calculate alpha based on effective_kappa using a smooth function
        alpha = 0.5 * (1 + math.tanh((effective_kappa - 1.2) / 0.5))
        alpha = max(min(alpha, 1), 0)

        # Adjust alpha based on dot product
        alpha = alpha * (1 - dot_product)

        # print(round(alpha.max().item(), 3), round(effective_kappa, 3), round(dot_product.max().item(), 3))

        # Move points towards mu
        steered_points = mu + (1 - alpha) * (points - mu)

        # Renormalize the steered points
        return self._safe_normalize(steered_points)


class Feature:
    def __init__(
        self,
        id: str,
        explanation: str = "",
        scores: dict = {},
        model: str = "",
        activation_sequences: Optional[np.ndarray] = np.ndarray((0, 2), dtype=np.int16),
        activations: Optional[np.ndarray] = np.ndarray((0, 128), dtype=np.float16),
    ):
        self.id = id
        self.explanation = explanation
        self.scores = scores
        self.model = model

        # num_samples, (dataset, sequence idx)
        self.activation_sequences = activation_sequences
        # num_samples, seq_len
        self.activations = activations

    def add_activation(
        self,
        dataset_idx: int,
        sequence_idx: int,
        activations: np.ndarray,
        scores: np.ndarray,
    ):
        self.activation_sequences = np.vstack(
            [self.activation_sequences, [dataset_idx, sequence_idx]]
        )
        self.activations = np.vstack([self.activations, activations])
        self.activation_scores = np.vstack([self.activation_scores, scores])

    def save(self, file_path: str):
        np.savez(file_path, **self.__dict__)

    @classmethod
    def load(cls, file_path: str):
        params = np.load(file_path, allow_pickle=True)
        return cls(**params)


class RQAEFeature(Feature):
    def __init__(
        self,
        num_quantizers: int = 1024,
        dim: int = 4,
        mask: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        kappa: Optional[np.ndarray] = None,
        queries: Optional[np.ndarray] = np.ndarray((0, 3), dtype=np.uint32),
        activation_subsample: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_quantizers = num_quantizers
        self.dim = dim

        self.mu = mu if mu is not None else np.zeros((num_quantizers, dim))
        self.kappa = kappa if kappa is not None else np.zeros(num_quantizers)
        self.mask = mask if mask is not None else np.ones(num_quantizers)

        self.queries = queries  # dataset, x, y

    @classmethod
    def from_quantizer(cls, quantizer: RQAE):
        return cls(num_quantizers=quantizer.num_quantizers, dim=quantizer.codebook_dim)

    def top_k(self, k: int):
        top_indices = np.argsort(self.kappa, descending=True)[:k]
        self.mask = [i in top_indices for i in range(self.num_quantizers)]
        self.mask = np.array(self.mask)

    def fit(self, points: torch.Tensor, **kwargs):
        """
        Fit VonMisesFisher4D distributions to the given points

        Args:
        points: torch.Tensor of shape [..., num_quantizers, dim], points on the 4D unit sphere
        **kwargs: Additional arguments to pass to the VonMisesFisher4D.fit method
        """
        points_reshaped = points.view(-1, self.num_quantizers, self.dim)
        for i in range(self.num_quantizers):
            distribution = VonMisesVisher4D()
            distribution.fit(points_reshaped[:, i], **kwargs)
            self.mu[i] = distribution.mu
            self.kappa[i] = distribution.kappa

    def pdf(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute probability for points x using the specified distributions

        Args:
        x: torch.Tensor of shape [..., num_quantizers, dim], points on the 4D unit sphere
        mask: Optional[torch.Tensor] of shape [num_quantizers], mask to ignore dimensions

        Returns:
        prob: torch.Tensor of shape [..., num_quantizers], probabilities
        """
        original_shape = x.shape[:-2] + (self.num_quantizers,)
        x_reshaped = x.view(
            -1,
            self.num_quantizers,
        )
        probs = torch.zeros(
            x_reshaped.shape[0],
            self.num_quantizers,
            dtype=torch.float32,
            device=x.device,
        )
        for i in range(self.num_quantizers):
            if self.mask[i] or mask is None or mask[i]:
                distribution = VonMisesFisher4D(mu=self.mu[i], kappa=self.kappa[i])
                probs[:, i] = distribution.pdf(x_reshaped[:, i])
        return probs.view(original_shape)

    def steer(
        self,
        points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength: float = 0.5,
    ) -> torch.Tensor:
        """
        Move points closer to being in their respective distributions

        Args:
        points: torch.Tensor of shape [..., num_quantizers, dim], points to steer
        mask: Optional[torch.Tensor] of shape [num_quantizers], mask to ignore dimensions
        strength: float, controls how much to move the points (0 to 1)

        Returns:
        steered_points: torch.Tensor of shape [..., num_quantizers, dim], steered points
        """
        steered_points = points.clone()
        for i in range(self.num_quantizers):
            if self.mask[i] or mask is None or mask[i]:
                distribution = VonMisesFisher4D(mu=self.mu[i], kappa=self.kappa[i])
                steered_points[..., i, :] = distribution.steer(
                    points[..., i, :], strength
                )
        return steered_points
