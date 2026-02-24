# Patches

This directory contains patches applied to third-party dependencies.

## transformer_engine_flash_attn_3.patch

**Purpose**: Patches TransformerEngine to use `flash_attn_interface` directly instead of `flash_attn_3.flash_attn_interface` for compatibility with the custom flash_attn_3 wheel.

**Applied to**: TransformerEngine (installed via pip)

**Files modified**:
- `pytorch/attention/dot_product_attention/backends.py`

**Changes**: Replaces all occurrences of `flash_attn_3.flash_attn_interface` with `flash_attn_interface` in import statements.

**Usage**: This patch is automatically applied by `install_requirements.sh` after TransformerEngine installation.
