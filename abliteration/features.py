from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class Feature:
    """A feature direction associated with a TransformerLens hook name."""

    name: str
    hook_name: str
    direction: Tensor

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature name must not be empty")
        if not self.hook_name:
            raise ValueError("Hook name must not be empty")
        if self.direction.ndim != 1:
            raise ValueError("Feature direction must be a one-dimensional tensor")
        if not torch.isfinite(self.direction).all():
            raise ValueError("Feature direction must contain only finite values")
        if torch.linalg.vector_norm(self.direction) == 0:
            raise ValueError("Feature direction must not be zero")


def _orthonormal_basis(directions: Tensor) -> Tensor:
    if directions.ndim == 1:
        directions = directions.unsqueeze(0)
    if directions.ndim != 2:
        raise ValueError("Directions must have shape (features, d_model)")
    if directions.shape[0] == 0 or directions.shape[1] == 0:
        raise ValueError("At least one non-empty feature direction is required")
    if not torch.isfinite(directions).all():
        raise ValueError("Feature directions must contain only finite values")

    _, singular_values, right_vectors = torch.linalg.svd(
        directions, full_matrices=False
    )
    if singular_values.numel() == 0 or singular_values[0] == 0:
        raise ValueError("Feature directions must not be zero")
    tolerance = (
        torch.finfo(singular_values.dtype).eps
        * max(directions.shape)
        * singular_values[0]
    )
    rank = int((singular_values > tolerance).sum().item())
    if rank == 0:
        raise ValueError("Feature directions must not be zero")
    return right_vectors[:rank].transpose(0, 1)


def ablate_activations(activations: Tensor, directions: Tensor) -> Tensor:
    """Remove the projection onto feature direction(s) from the last axis."""
    if activations.ndim == 0:
        raise ValueError("Activations must have a model-dimension axis")
    basis = _orthonormal_basis(directions).to(
        device=activations.device, dtype=activations.dtype
    )
    return _ablate_with_basis(activations, basis)


def _ablate_with_basis(activations: Tensor, basis: Tensor) -> Tensor:
    if activations.ndim == 0:
        raise ValueError("Activations must have a model-dimension axis")
    if activations.shape[-1] != basis.shape[0]:
        raise ValueError(
            "Activation model dimension must match feature direction size "
            f"({activations.shape[-1]} != {basis.shape[0]})"
        )
    projected = (activations @ basis) @ basis.transpose(0, 1)
    return activations - projected


class FeatureAblator:
    """Build forward hooks that ablate features at their configured hookpoints."""

    def __init__(self, features: Iterable[Feature]) -> None:
        self.features = tuple(features)
        if not self.features:
            raise ValueError("At least one feature is required")

    def hooks(self) -> List[Tuple[str, Callable[[Tensor, Any], Tensor]]]:
        grouped = defaultdict(list)
        for feature in self.features:
            grouped[feature.hook_name].append(feature.direction)

        result = []
        for hook_name, directions in grouped.items():
            basis = _orthonormal_basis(torch.stack(directions))

            def ablate_hook(
                activation: Tensor,
                hook: Any,
                feature_basis: Tensor = basis,
            ) -> Tensor:
                basis_on_device = feature_basis.to(
                    device=activation.device, dtype=activation.dtype
                )
                return _ablate_with_basis(activation, basis_on_device)

            result.append((hook_name, ablate_hook))
        return result

    def run_with_hooks(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a TransformerLens model with these forward hooks applied."""
        if "fwd_hooks" in kwargs:
            raise ValueError("Pass forward hooks through FeatureAblator, not kwargs")
        return model.run_with_hooks(*args, fwd_hooks=self.hooks(), **kwargs)