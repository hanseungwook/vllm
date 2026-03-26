# scripts/

Developer utility scripts for the vLLM project. These are standalone tools
meant to be run manually, outside of the normal build and test pipelines.

## Scripts

### `autotune_helion_kernels.py`

Autotunes registered [Helion](https://github.com/pytorch-labs/helion) GPU
kernels to find optimal configurations for the current hardware. Autotuned
configs are saved as per-platform JSON files managed by `vllm.kernels.helion.ConfigManager`
(see `vllm/kernels/helion/config_manager.py`).

**Requirements:** A CUDA-capable GPU, the `helion` Python package, and a
working vLLM installation.

**Usage:**

```bash
# List all registered kernels
python scripts/autotune_helion_kernels.py --list

# Autotune all registered kernels (skips configs that already exist)
python scripts/autotune_helion_kernels.py

# Autotune one or more specific kernels
python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8
python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8 rms_norm_fp8

# Force re-autotuning even when configs already exist
python scripts/autotune_helion_kernels.py --force

# Use a custom config output directory
python scripts/autotune_helion_kernels.py --config-dir /path/to/configs

# Control search budget ("quick" or "full"; default is "quick")
python scripts/autotune_helion_kernels.py --autotune-effort full

# Enable debug-level logging
python scripts/autotune_helion_kernels.py --verbose
```

The script exits with code 0 on success and 1 if any kernel fails to autotune.
