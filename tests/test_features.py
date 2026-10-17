import pytest
import torch

from abliteration import Feature, FeatureAblator, ablate_activations


def test_ablation_removes_direction_and_preserves_orthogonal_component():
    direction = torch.tensor([1.0, 0.0, 0.0])
    activations = torch.tensor([[3.0, 4.0, 5.0]])

    result = ablate_activations(activations, direction)

    torch.testing.assert_close(result, torch.tensor([[0.0, 4.0, 5.0]]))


def test_ablation_removes_span_for_non_orthogonal_directions():
    directions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    activations = torch.tensor([[3.0, 4.0, 5.0]])

    result = ablate_activations(activations, directions)

    torch.testing.assert_close(result, torch.tensor([[0.0, 0.0, 5.0]]))


def test_feature_ablator_registers_hook_and_runs_model():
    feature = Feature("example", "blocks.0.hook_resid_post", torch.tensor([1.0, 0.0]))
    ablator = FeatureAblator([feature])

    class FakeModel:
        def run_with_hooks(self, prompt, fwd_hooks):
            assert prompt == "prompt"
            hook_name, hook_fn = fwd_hooks[0]
            assert hook_name == feature.hook_name
            return hook_fn(torch.tensor([[2.0, 3.0]]), None)

    result = ablator.run_with_hooks(FakeModel(), "prompt")

    torch.testing.assert_close(result, torch.tensor([[0.0, 3.0]]))


def test_feature_rejects_zero_direction():
    with pytest.raises(ValueError, match="must not be zero"):
        Feature("zero", "blocks.0.hook_resid_post", torch.zeros(3))