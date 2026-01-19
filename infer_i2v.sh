torchrun --nproc_per_node=8 generate.py \
 --task i2v-A14B \
 --size 1280*720 \
 --ckpt_dir /data/Ali/Wan2.2-I2V-A14B \
 --image examples/i2v_input.JPG \
 --dit_fsdp \
 --t5_fsdp \
 --ulysses_size 8 \
 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
 --quantize \
 --fp8_recipe DelayedScaling 
#  --tf32 True \
#  --compile True \
