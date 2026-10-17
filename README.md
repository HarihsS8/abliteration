# abliteration

`abliteration` is a PyTorch library for removing known feature directions from
activations in models supported by
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
Interventions are applied with TransformerLens forward hooks; model weights are
not changed.

This library applies feature directions supplied by you. It does not discover
features or train sparse autoencoders. A direction might come from an SAE
decoder, a contrastive activation analysis, or another feature-extraction
workflow.

## Install

For use with TransformerLens:

```bash
pip install "abliteration[transformer-lens]"
```

For local development and tests:

```bash
pip install -e ".[test]"
pytest
```

PyTorch is the only required runtime dependency. TransformerLens is optional if
you only need the tensor-level ablation function.

## TransformerLens Example

Load a one-dimensional feature direction whose length matches the model's
residual-stream dimension. Choose a hook name corresponding to the layer and
activation point where that direction was measured.

```python
import torch
from transformer_lens import HookedTransformer

from abliteration import Feature, FeatureAblator

model = HookedTransformer.from_pretrained("gpt2-small")
feature_direction = torch.load(
	"feature_direction.pt", map_location="cpu", weights_only=True
)

ablator = FeatureAblator([
	Feature(
		name="refusal",
		hook_name="blocks.10.hook_resid_post",
		direction=feature_direction,
	),
])
logits = ablator.run_with_hooks(model, "The model prompt")
```

`FeatureAblator` accepts multiple features. Directions assigned to the same
hookpoint are treated as a subspace: the projection onto their combined span is
removed. Directions at different hookpoints are applied independently.

## Tensor API

Use `ablate_activations` without loading a TransformerLens model:

```python
from abliteration import ablate_activations

ablated = ablate_activations(activations, feature_direction)
```

`activations` can have any number of leading dimensions; its last dimension
must match the feature direction size. `feature_direction` can be a vector of
shape `(d_model,)` or a matrix of shape `(n_features, d_model)`. The function
removes the projection onto the supplied direction or their combined span.
