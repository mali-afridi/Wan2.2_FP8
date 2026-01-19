# Wan2.2 with FP8 Quantization (NVIDIA Transformer Engine)


<p align="center">
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/c9e9327b-8c14-427f-9acd-1e0e084d424b" />
</p>

---

## ⚡ Introduction

**Want a training-free inference boost** using Floating Point 8 Quantization for Wan2.2 video generation **without relying on `torch.compile` to work with `torchao`?** That's where NVIDIA's **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine.git)** comes into play! 

If you quantize during inference using `torchao`, you need to use `torch.compile` in order to achieve speedup from the quantization ([source](https://pytorch.org/blog/pytorch-native-architecture-optimization/)).<br>
While `torch.compile` offers great speedups, using a full-graph compilation can be challenging in production environments due to inherent code graph breaks and complexities with FSDP (Fully Sharded Data Parallel)—especially when LoRAs are involved. <br>
Furthermore, distilling models works well for base model inference but often loses context when fine-tuned downstream task LoRAs are merged on top of it.<br>

This repository **implements FP8 quantization** using **NVIDIA's Transformer Engine** to achieve significant **speedups** while **maintaining quality** without relying on torch.compile.
I have also included the support of Magcache for I2V on 8xH100, which you can use with FA3 and FP8!

## Key Features:
I utilize the **E4M3** FP8 format, which is optimized for inference stability. This implementation supports the following Transformer Engine recipes:
* `Float8CurrentScaling`
* `Float8BlockScaling`
* `DelayedScaling` (Recommended for best performance/quality balance)

Additionally this repo has:
* Ready to install compatible Flash Attention 3 Wheel (check release)
* Magcache support for I2V on 8xH100
* FSDP2 Sharding support

## 📊 Performance Metrics (720P Generation)
Tested on **8x H100** GPUs, Image-to-Video (I2V), 40 Steps.

| Configuration | Inference Time (s) | Speedup vs Baseline |
| :--- | :---: | :---: |
| **Baseline (Flash Attn 2)** | 250.70s | 1.0x |
| **Flash Attn 3** | 195.13s | 1.28x |
| **Flash Attn 3 + FP8** | **146.55s** | **1.71x** 🚀 |
| **Flash Attn 3 + FP8 + Magcache (E012K2R20)** | **114.12s** | **2.2x** 🚀 |
| **Flash Attn 3 + FP8 + Magcache (E012K2R20) + Torch.compile (Fullgraph=False)** | **98.03.s** | **2.55x** 🚀 |
| **Flash Attn 3 + FP8 + Magcache (E024K2R10) + Torch.compile (Fullgraph=False)** | **89.36s** | **2.80x** 🚀 |

_Note: FP8 recipe used here was DelayedScaling with amax_history_len=16 & amax_compute_algo="max"_

## 📦 Installation

To utilize Transformer Engine, you must have Hopper H100 GPUs with CUDA toolkit version 12.9+ to support sm90 compute. To install Cuda toolkit, follow the steps: 
```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.17.0/local_installers/cudnn-local-repo-ubuntu2204-9.17.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.17.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.17.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get -y install cudnn9-cuda-12
```

```bash
# Clone the repository
git clone [https://github.com/mali-afridi/Wan2.2_FP8.git](https://github.com/mali-afridi/Wan2.2_FP8.git)
cd Wan2.2_FP8

#Make virtual environment
python3 -m venv fp8
source fp8/bin/activate

# Run the installation script
# This handles PyTorch, Flash Attention 3 patching, Transformer Engine and other necessary libraries needed
bash install_requirements.sh
```
## Usage Example
Make sure you download the models [T2V](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers), [I2V](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers), [Animate](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) and [S2V](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B).
I have prepared the bash scripts for `I2V`, `T2V`, `Animate` & `S2V`. Magcache is supported for I2V only for now. I can easily extend it to others if required. 
##### To use FP8 E4M3 DelayedScaling:
```bash
torchrun --nproc_per_node=8 generate.py \
 --task i2v-A14B \
 --size 1280*720 \
 --ckpt_dir /data/Ali/Wan2.2-I2V-A14B \
 --image examples/i2v_input.JPG \
 --dit_fsdp \
 --t5_fsdp \
 --ulysses_size 8 \
 --prompt "some prompt" \
 --quantize \
 --fp8_recipe DelayedScaling \
```
We can always choose between `DelayedScaling`, `Float8BlockScaling` & `Float8CurrentScaling`
##### To enable magcache, provide the extra arguments:
```bash
 --use_magcache \
 --magcache_thresh 0.12 \
 --retention_ratio 0.2 \
 --magcache_K 2 \
```
##### To enable torch.compile with fullgraph=False to gain extra speed, use:
```bash
--tf32 True \
--compile True \
```
But the speedup of FP8 quantization doesn't come from the use of torch.compile which differentiates it from torchao quantization.
##### Complete bash scripts can be run (adjust based on the requirements):
```bash
bash infer_i2v.sh
bash infer_t2v.sh
bash infer_animate.sh
bash infer_s2v.sh
```

## 🎥 Quality Comparison (I2V)

<table align="center">
  <tr>
    <th align="center" width="50%">Baseline (FA2)</th>
    <th align="center" width="50%">FP8 Quantized</th>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/165a4b4d-d684-4614-bdf6-f314f2339d3d" width="100%" controls autoplay loop muted></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/981bd07f-c8e3-4539-9625-01a0f02e5fc4" width="100%" controls autoplay loop muted></video>
    </td>
  </tr>
</table>

_Resolution: 1280*720, 40 steps & base_seed: 50. FP8 E4M3 DelayedScaling with amax_history_len=16 & amax_compute_algo="max"_


## 🧠 FP8 Quantization & Technical Implementation
### 🛠️ Solving the "Divisible by 8 and 16" Constraint

A major challenge when applying FP8 quantization to video generation models like Wan2.2 is the strict tensor dimension requirement of the Transformer Engine kernel:

> *AssertionError: FP8 execution requires the product of all dimensions except the last to be divisible by 8 and the last dimension to be divisible by 16.*

In Wan2.2, standard input tensors in Wan2.2 (e.g., `[1, 5566, 5120]`) often violate this rule. I implemented a dynamic **Padding & Slicing** strategy to resolve this:

1.  **Padding:** Inputs are dynamically padded to the nearest multiple required by the TE kernel.
    * *Example:* `torch.Size([1, 5566, 5120])` $\rightarrow$ `torch.Size([1, 5568, 5120])`
2.  **Quantized Operation:** The computationally intensive FP8 Matrix Multiplication runs on the padded tensor.
3.  **Slicing:** The zero-padding is sliced off the output before passing it to the next layer, ensuring mathematical correctness without shape mismatches.

This support extends across all pipelines: **Text-to-Video (T2V), Image-to-Video (I2V), Video-to-Video (Animate), and Speech-to-Video (S2V).**

---

## 💐 Acknowledgement
This repository is built based on [Wan2.2](https://github.com/Wan-Video/Wan2.2.git), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine.git) & [Magcache](https://github.com/Zehong-Ma/MagCache.git). Thanks for their contributions!

## Suggestions 
Suggestions and Contributions are welcome 🩷. Feel free to report any issues you face. 
