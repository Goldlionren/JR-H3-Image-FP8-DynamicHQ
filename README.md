# JR-H3-Image-FP8-DynamicHQ

Experimental MiniMax H3-Image FP8 Dynamic Activation Scaling for ComfyUI.

## Verified environment
- ComfyUI 0.45.0
- PyTorch 2.13.0+cu130
- comfy-kitchen 0.2.41
- RTX 4080 SUPER
- FP8 E4M3FN, 200 Linear layers
- Dynamic activation scaling
- 6 steps: ~24.4 seconds
- End-to-end: 48.05 seconds

## Installation
Copy custom_nodes/JR_H3_FP8_Dynamic into
ComfyUI/custom_nodes/ and restart ComfyUI.

Keep ENABLE_DYNAMIC_FP8.txt to enable the patch.
Remove the marker and restart to disable it.

Use a compatible FP8 H3-Image checkpoint with
full_precision_matrix_mult=false and loader dtype=default.

## Status
Experimental RC1. FP8 Tensor Core profiling pending.

## Model licensing
Weights are not included. Derived model redistribution
is subject to the MiniMax-H3-Image Community License.
