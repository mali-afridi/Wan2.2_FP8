
#sudo install this first before running the script 
# sudo apt-get install -y --no-install-recommends gcc-12 g++-12

#make sure you have CUDA 12.9 on your H100 system to support sm90 compute for transformer engine
export CUDA_PATH=/usr/local/cuda-12.9
export CUDNN_PATH=/usr
export CUDA_HOME=/usr/local/cuda-12.9

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install packaging ninja wheel pybind11 nvidia-mathdx
#install my flash attention 3 wheel compatible with torch 2.8.0

curl -L -H "Accept: application/octet-stream" \
       "https://api.github.com/repos/mali-afridi/Wan2.2_FP8/releases/assets/340271825" \
       -o flash_attn_3-3.0.0b1-py3-none-linux_x86_64.whl \
    && pip install flash_attn_3-3.0.0b1-py3-none-linux_x86_64.whl \

pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Apply patch to transformer_engine to use flash_attn_interface instead of flash_attn_3.flash_attn_interface
TE_PATH=$(python -c "import transformer_engine, os; print(os.path.dirname(transformer_engine.__file__))")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch -d "$TE_PATH" -p1 < "$SCRIPT_DIR/patches/transformer_engine_flash_attn_3.patch" || {
    echo "Warning: Failed to apply transformer_engine patch. Continuing anyway..."
}

# Install other requirements
pip install opencv-python>=4.9.0.80
pip install "diffusers>=0.31.0"
pip install "transformers>=4.49.0,<=4.51.3"
pip install "tokenizers>=0.20.3"
pip install "accelerate>=1.1.1"
pip install tqdm peft decord librosa loguru moviepy onnxruntime matplotlib
pip install "imageio[ffmpeg]"
pip install easydict
pip install ftfy sam2 sentencepiece
pip install dashscope
pip install imageio-ffmpeg
pip install "numpy>=2"