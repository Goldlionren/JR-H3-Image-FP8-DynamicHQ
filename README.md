# JR-H3-Image-FP8-DynamicHQ

### High-Quality FP8 Inference for Compute-Unlocked NVIDIA RTX 40-Series GPUs

**MiniMax H3-Image | FP8 E4M3FN | Dynamic Activation Scaling | ComfyUI**

> **IMPORTANT: This project is primarily designed for NVIDIA RTX 40-series GPUs with independently unlocked FP8 Tensor Core throughput.**
>
> GPU unlocking is not included in this repository. On stock RTX 40-series cards, FP8 may offer little or no performance advantage over the official INT8 ConvRot implementation. Actual performance depends on the GPU, kernels, memory, and workload.

**Project Status:** Experimental Release Candidate

**Target Hardware:** Compute-unlocked RTX 4080 SUPER / RTX 4090 and other compatible RTX 40-series configurations

**Framework:** ComfyUI 0.45.0 (tested)

**Quantization:** FP8 E4M3FN with dynamic activation scaling

---

## 1. Project Overview

JR-H3-Image-FP8-DynamicHQ is an independent optimization project for the MiniMax H3-Image model.

The goal is to explore high-quality FP8 inference on NVIDIA RTX 40-series GPUs with independently unlocked FP8 Tensor Core throughput.

The project provides:

- A BF16-to-FP8 model converter.
- FP8 E4M3FN quantization for 200 main DiT Linear modules.
- A dynamic FP8 activation scaling plugin for ComfyUI.
- Preservation of the original H3-Image PDD configuration.
- Compatibility with the existing ComfyUI model loader.
- Compatibility with existing CK Attention optimizations.

This project does NOT include GPU unlocking software, firmware patches, or driver modifications.

### Why unlocked RTX 40-series GPUs?

The primary motivation is to take advantage of additional FP8 Tensor Core compute throughput available on independently unlocked hardware.

On standard RTX 40-series GPUs, INT8 ConvRot may be equally fast or faster.

Therefore, users should not assume FP8 will outperform INT8 without measuring actual inference performance.

GPU compute throughput is only one part of performance. Memory bandwidth, activation scaling, CPU offloading, and attention kernels also affect generation speed.

---

## 2. Why Dynamic FP8?

During development, the initial FP8 checkpoint successfully reduced the model size but produced severely degraded images.

The symptoms included:

- Grainy skin and backgrounds.
- Excessive texture noise.
- Lost fine details.
- Noticeably worse image quality than BF16 and INT8 ConvRot.

The FP8 weight conversion itself was not the primary source of the observed quality failure.

A diagnostic test using FP8 weight storage with higher-precision matrix multiplication restored image quality.

This led us to investigate FP8 activation scaling.

### Fixed Activation Scaling

FP8 E4M3FN has a limited numerical range.

The maximum finite E4M3FN value is 448.

A simplified activation quantization operation is:

```text
scale = max(abs(activation)) / 448

quantized_activation =
    FP8(activation / scale)
```

ComfyUI supports storing a calibrated input_scale for an FP8 Linear module.

However, when no input_scale is supplied, the default scale is 1.0.

Our initial FP8 checkpoint did not contain calibrated activation scales.

Using the default activation scaling caused major image quality degradation.

### Why Not Use One Fixed Scale?

MiniMax H3-Image uses a six-step PDD generation process.

Activation distributions may change depending on:

1. Diffusion sampling step.
2. Prompt and semantic content.
3. Reference images.
4. Image resolution.
5. Token composition.
6. Individual Transformer layer.

A fixed activation scale may work well for one distribution but poorly for another.

If a fixed scale is too small, activations may be clipped.

If it is too large, the available FP8 precision is underutilized.

A properly calibrated fixed-scale system could still work, but it would require a representative calibration dataset and additional validation.

### Dynamic Activation Scaling

Instead of storing one fixed activation scale, DynamicHQ recalculates the scale at runtime.

For each eligible FP8 Linear input:

```python
scale = "recalculate"
```

The runtime uses the current activation tensor to determine its quantization scale.

This makes quantization adapt to the actual input rather than relying on an uncalibrated constant.

The FP8 weights remain stored in FP8 E4M3FN format.

The weight scales are stored in the checkpoint.

The activation scales are calculated dynamically during inference.

### Why This Approach?

Our local A/B experiments showed:

| Mode | Image Quality |
| --- | --- |
| Official BF16 | Clear |
| Official INT8 ConvRot | Clear |
| Initial FP8, default activation scaling | Severely degraded |
| FP8 weights, higher-precision GEMM | Clear |
| FP8 + DynamicHQ activation scaling | Clear |

Dynamic activation scaling restored the observed image quality while retaining the FP8 matrix-multiplication configuration.

Dynamic scaling also introduces a reduction and rescaling overhead. It is not guaranteed to improve speed or quality on every GPU.

---

## 3. Tested Hardware and Performance

The following measurements were obtained from a local RTX 4080 SUPER system.

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| GPU Configuration | Independently unlocked FP8 throughput |
| Physical VRAM | Approximately 16 GB |
| System RAM | 64 GB |
| Operating System | Windows |
| ComfyUI | 0.45.0 |
| PyTorch | 2.13.0+cu130 |
| comfy-kitchen | 0.2.41 |
| Attention | CK Attention enabled |
| Weight Precision | FP8 E4M3FN |
| FP8 Linear Modules | 200 |
| FP8 Checkpoint Size | Approximately 20.97 GB |
| Sampling Steps | 6 |
| Dynamic Scaling Calls | 200 on first forward |
| Six-Step Denoising | Approximately 24.4 seconds |
| Complete Workflow | Approximately 48.05 seconds |

The model used ComfyUI DynamicVRAM and asynchronous weight offloading.

The text encoder used in this measurement was Qwen3-VL-32B INT8 ConvRot, with a separate FP16 VAE.

### Performance Limitations

These measurements are from one successful local run.

They are not a controlled FP8-versus-INT8 ConvRot benchmark.

The 200 dynamic quantization calls confirm that the plugin's dynamic quantization path was exercised.

They do not independently prove that every GEMM executed FP8 Tensor Core MMA instructions.

Nsight kernel-level validation and broader performance testing remain future work.

---

## 4. Project Components

```text
JR-H3-Image-FP8-DynamicHQ/
|
|-- README.md
|
|-- tools/
|   |-- convert_h3_image_bf16_to_fp8.py
|
|-- custom_nodes/
|   |-- JR_H3_FP8_Dynamic/
|       |-- __init__.py
|       |-- ENABLE_DYNAMIC_FP8.txt
|
|-- .gitignore
```

### FP8 Converter

The converter transforms the 200 main DiT Linear weights from BF16 into FP8 E4M3FN.

Specifically, it processes:

- 50 attention QKV projection matrices.
- 50 attention output projection matrices.
- 50 MLP FC1 matrices.
- 50 MLP FC2 matrices.

The remaining weights retain their original precision.

It preserves the original MiniMax H3-Image configuration metadata, including PDD steps and sigma values.

The converter also generates a validation report containing quantization error statistics for all 200 converted matrices.

### DynamicHQ Plugin

The plugin modifies the activation quantization behavior during MiniMax H3-Image forward execution.

It uses a ContextVar to enable dynamic scaling only within the relevant model forward context.

It does not modify the model weights.

It does not disable CK Attention.

It does not change the user's GPU unlock configuration.

---

## 5. Model Weights and Licensing

**Pre-quantized FP8 weights are not publicly available at this time.**

MiniMax H3-Image is distributed under the MiniMax-H3-Image Community License Agreement.

The agreement contains restrictions on model use and derivative distribution, including territorial restrictions.

We are seeking clarification from MiniMax about public redistribution of the derived FP8 checkpoint.

Until an authorized distribution arrangement is confirmed, the converted checkpoint will remain private.

**Users must obtain the official BF16 model from authorized sources and perform the conversion locally, subject to the original license.**

Local conversion does not override the source model's license restrictions.

Official sources:

- MiniMax: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
- Comfy-Org: https://huggingface.co/Comfy-Org/MiniMax-H3-Image
- ComfyUI: https://github.com/Comfy-Org/ComfyUI

Please use the official Comfy-Org PRUNED BF16 DiT checkpoint.

Do not attempt to convert an existing INT8, W4A8, or other quantized checkpoint with this tool.

---

## 6. Convert BF16 to FP8

### Requirements

- Windows or another supported operating system.
- Python with CUDA-enabled PyTorch.
- safetensors.
- Official Comfy-Org MiniMax H3-Image PRUNED BF16 DiT.
- Sufficient disk space for both source and output files.

The following example uses Windows PowerShell and an existing ComfyUI Python environment.

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Goldlionren/JR-H3-Image-FP8-DynamicHQ.git

cd JR-H3-Image-FP8-DynamicHQ
```

### Step 2: Obtain the Source Model

Download:

```text
minimax_h3_image_pruned_bf16.safetensors
```

from the official Comfy-Org MiniMax H3-Image repository.

Place it into your ComfyUI diffusion_models folder.

### Step 3: Convert to FP8

Example:

```powershell
$Py = "F:\Comfy-Desktop\ComfyUI-Installs\default\ComfyUI\.venv\Scripts\python.exe"

$Models = "F:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models"

& $Py ".\tools\convert_h3_image_bf16_to_fp8.py" `
    --source "$Models\minimax_h3_image_pruned_bf16.safetensors" `
    --out "$Models\jr_h3_image_fp8_dynamic_v2.safetensors" `
    --device cuda:0
```

The output is a new FP8 E4M3FN checkpoint.

A companion validation JSON report is also generated.

The converter refuses to overwrite an existing output checkpoint.

### Important

The V2 converter has passed syntax and structural review.

A fresh full-size 40 GB conversion followed by independent image-quality regression testing is still pending.

Do not overwrite a known-good model during experimentation.

---

## 7. Install DynamicHQ in ComfyUI

Copy this directory:

```text
custom_nodes/JR_H3_FP8_Dynamic/
```

into:

```text
ComfyUI/custom_nodes/JR_H3_FP8_Dynamic/
```

The directory should contain:

```text
__init__.py
ENABLE_DYNAMIC_FP8.txt
```

The presence of ENABLE_DYNAMIC_FP8.txt enables the runtime patch when ComfyUI starts.

Restart ComfyUI after installation.

Expected startup message:

```text
[JR-H3-FP8] Dynamic activation scaling ENABLED
```

During the first H3-Image forward pass, the tested setup logged:

```text
[JR-H3-FP8] Dynamic FP8 inputs: 200
```

If zero dynamic scaling calls are reported, investigate the execution path before relying on the result.

### Disable the Plugin

1. Fully close ComfyUI.
2. Remove or rename ENABLE_DYNAMIC_FP8.txt.
3. Restart ComfyUI.

No ComfyUI core source files need to be modified.

---

## 8. ComfyUI Configuration

Use the official MiniMax H3-Image workflow.

Load the generated FP8 model with:

```text
Load Diffusion Model

unet_name:
    jr_h3_image_fp8_dynamic_v2.safetensors

weight_dtype:
    default
```

The checkpoint should contain:

```json
{
  "format": "float8_e4m3fn",
  "full_precision_matrix_mult": false
}
```

Do NOT change full_precision_matrix_mult to true if your goal is the FP8 GEMM path.

Setting it to true was useful during our diagnostic experiments, but it causes the quantized weights to be dequantized for higher-precision matrix multiplication.

Use the official six-step H3-Image sampling configuration.

The text encoder and VAE are separate downloads.

### CK Attention

This project is designed to coexist with existing ComfyUI attention optimizations.

Users can keep:

```text
--use-ck-attention
```

No attention optimization needs to be disabled to use DynamicHQ.

---

## 9. Why Not Simply Save Dynamic Scales in the Model?

The checkpoint stores quantized weights and their weight scales.

However, the current ComfyUI checkpoint format does not provide a native per-module option that instructs its FP8 loader to recalculate the activation scale on every forward pass.

A stored input_scale is a fixed value, not an instruction to recompute the current activation statistics.

DynamicHQ therefore implements this behavior at runtime.

Future versions may support native dynamic-scaling configuration within ComfyUI itself.

That could make it possible to use the same behavior without a separate custom node.

Until then, the tested combination is:

```text
FP8 E4M3FN Weights
        +
DynamicHQ Runtime Plugin
        +
ComfyUI FP8 Operations
        +
Existing CK Attention
```

---

## 10. Known Limitations

- Designed primarily for independently compute-unlocked RTX 40-series GPU configurations.
- No GPU unlocking code is provided.
- No guaranteed FP8 speed advantage on stock GPUs.
- No public download of derivative model weights at present.
- The plugin uses internal ComfyUI APIs.
- Model quality has not been benchmarked across a large evaluation dataset.
- A complete matched INT8 ConvRot performance comparison is pending.
- FP8 Tensor Core instruction profiling is pending.
- The final V2 converter requires independent full-model validation.
- Dynamic scaling adds its own runtime overhead.
- Other ComfyUI versions may require compatibility updates.
- The repository has not yet adopted a separate open-source license for its original code.

---

## 11. Roadmap

**v1.0 RC**

- FP8 E4M3FN conversion.
- DynamicHQ activation scaling.
- High-quality local image generation validated.

**v1.1**

- Full-model V2 converter validation.
- Multiple prompts and resolutions.
- Controlled INT8 ConvRot versus FP8 benchmarks.

**v1.2**

- Nsight FP8 Tensor Core kernel profiling.
- Additional GPU compatibility testing.

**v2.0**

- Explore native ComfyUI dynamic-scaling metadata support.
- Reduce dependence on runtime monkeypatching.
- Explore calibrated static and hybrid activation scaling.

---

## 12. Credits

Original model:

https://huggingface.co/MiniMaxAI/MiniMax-H3-Image

ComfyUI implementation:

https://github.com/Comfy-Org/ComfyUI

ComfyUI model weights:

https://huggingface.co/Comfy-Org/MiniMax-H3-Image

JR-H3-Image-FP8-DynamicHQ is an independent community research project and is not affiliated with or endorsed by MiniMax, Comfy-Org, or NVIDIA.

---

# 中文说明

## 项目目标

本项目主要面向**已经解锁 FP8 Tensor Core 计算吞吐性能的 NVIDIA RTX 40 系列显卡**。

目标是利用 FP8 E4M3FN 与动态激活缩放技术，实现 MiniMax H3-Image 的高质量本地推理。

本项目不包含显卡解锁程序。

对于未解锁的普通 RTX 40 系列显卡，FP8 相对于官方 INT8 ConvRot 的性能优势可能非常有限，甚至没有优势。

## 动态量化的意义

最初制作的 FP8 模型虽然压缩成功，但图像严重粗糙。

使用高精度 GEMM 后画质恢复，说明问题主要集中在 FP8 计算路径。

进一步实验发现，启用动态激活缩放后，可以在保持 FP8 计算配置的情况下恢复画质。

动态方案针对每次实际输入重新计算激活 Scale，而不是使用一个固定的数值。

这尤其适合研究 H3-Image 六步 PDD 推理中可能变化的激活范围。

固定 Scale 并非不可行，但需要额外的校准和验证。

目前已经实测成功的是 FP8 权重与 DynamicHQ 插件的组合。

## 为什么暂不公开权重？

MiniMax H3-Image 的社区许可证对衍生模型的使用和再分发存在限制。

我们正在确认公开分发 FP8 衍生权重所需的授权条件。

因此，目前不提供公开的 Hugging Face 权重下载。

用户需要在原始许可证允许的前提下，从官方获取 Pruned BF16 模型，并使用仓库中的 V2 转换器自行转换。

## 已验证结果

- RTX 4080 SUPER，解锁 FP8 算力。
- ComfyUI 0.45.0。
- FP8 E4M3FN，200 个核心 Linear。
- FP8 DiT 大约 20.97GB。
- 动态激活缩放 200 次调用。
- 六步去噪约 24.4 秒。
- 完整工作流约 48.05 秒。
- 单组画质对比测试通过。

以上数据不是 FP8 相对 INT8 ConvRot 的严格性能对照，后续还将继续验证。

## 使用方法

首先使用 tools/convert_h3_image_bf16_to_fp8.py 将官方 Pruned BF16 转成 FP8。

然后将 custom_nodes/JR_H3_FP8_Dynamic 安装到 ComfyUI。

启用 ENABLE_DYNAMIC_FP8.txt，重启 ComfyUI。

模型加载时选择 weight_dtype=default，使用官方六步工作流，并保留现有 CK Attention 优化。

**不要使用普通的固定 FP8 激活 Scale 代替动态插件，否则可能重新出现明显的画质退化。**

---

**JR-H3-Image-FP8-DynamicHQ — Experimental FP8 research for compute-unlocked RTX 40-series GPUs.**