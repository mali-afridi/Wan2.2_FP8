# Wan2.2 with FP8 Quantization (NVIDIA Transformer Engine)


<p align="center">
   <b>🚀 Training-free Inference Acceleration with FP8</b>
</p>

---

## ⚡ Introduction

**Want a training-free inference boost for Wan2.2 video generation without relying on `torch.compile`?** That's where NVIDIA's **Transformer Engine** comes into play! 

While `torch.compile` offers great speedups, using a full-graph compilation can be challenging in production environments due to inherent code graph breaks and complexities with FSDP (Fully Sharded Data Parallel)—especially when LoRAs are involved. Furthermore, distilling models works well for base model inference but often loses context when fine-tuned downstream task LoRAs are merged on top.

**This repository implements FP8 quantization using NVIDIA's Transformer Engine to achieve significant speedups while maintaining flexibility.**

### 📊 Performance Metrics (720P Generation)

Tested on **8x H100** GPUs, Video-to-Video (I2V), 40 Steps.

| Configuration | Inference Time (s) | Speedup vs Baseline |
| :--- | :---: | :---: |
| **Baseline (Flash Attn 2)** | 250.70s | 1.0x |
| **Flash Attn 3** | 195.13s | 1.28x |
| **Flash Attn 3 + FP8 (DelayedScaling)** | **146.55s** | **1.71x** 🚀 |

---

## 🧠 FP8 Quantization & Technical Implementation

We utilize the **E4M3** FP8 format, which is optimized for inference stability. This implementation supports the following Transformer Engine recipes:
* `Float8CurrentScaling`
* `Float8BlockScaling`
* `DelayedScaling` (Recommended for best performance/quality balance)

### 🛠️ Solving the "Divisible by 16" Constraint

A major challenge when applying FP8 quantization to video generation models like Wan2.2 is the strict tensor dimension requirement of the Transformer Engine kernel:

> *AssertionError: FP8 execution requires the product of all dimensions except the last to be divisible by 8 and the last dimension to be divisible by 16.*

In Wan2.2, standard input tensors (e.g., `[1, 5566, 5120]`) often violate this rule. We implemented a dynamic **Padding & Slicing** strategy to resolve this:

1.  **Padding:** Inputs are dynamically padded to the nearest multiple required by the TE kernel.
    * *Example:* `torch.Size([1, 5566, 5120])` $\rightarrow$ `torch.Size([1, 5568, 5120])`
2.  **Quantized Operation:** The computationally intensive FP8 Matrix Multiplication runs on the padded tensor.
3.  **Slicing:** The zero-padding is sliced off the output before passing it to the next layer, ensuring mathematical correctness without shape mismatches.

This support extends across all pipelines: **Text-to-Video (T2V), Image-to-Video (I2V), Video-to-Video (Animate), and Speech-to-Video (S2V).**

---

## 📦 Installation

To utilize FP8, you must install the NVIDIA Transformer Engine and a compatible PyTorch version.

```bash
# Clone the repository
git clone [https://github.com/your-username/Wan2.2-FP8.git](https://github.com/your-username/Wan2.2-FP8.git)
cd Wan2.2-FP8

# Run the installation script
# This handles PyTorch, Flash Attention 3 patching, and Transformer Engine
bash install_requirements.sh