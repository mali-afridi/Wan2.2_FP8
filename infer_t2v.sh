torchrun --nproc_per_node=8 generate.py \
 --task t2v-A14B \
 --size 1280*720 \
 --ckpt_dir /data/Ali/Wan2.2-T2V-A14B \
 --dit_fsdp \
 --t5_fsdp \
 --ulysses_size 8 \
 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
 --tf32 True \
 --quantize \
# --compile True \