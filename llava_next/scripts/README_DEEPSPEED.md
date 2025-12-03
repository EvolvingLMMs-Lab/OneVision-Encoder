# DeepSpeed Configuration Guide

This directory contains DeepSpeed ZeRO configuration files for distributed training.

## Configuration Files

- **zero2.json**: ZeRO Stage 2 configuration
- **zero3.json**: ZeRO Stage 3 configuration

## Troubleshooting

### FusedAdam Import Error

**Problem:** When running training with DeepSpeed, you may encounter this error:
```
ImportError: /root/.cache/torch_extensions/py310_cu124/fused_adam/fused_adam.so: cannot open shared object file: No such file or directory
```

**Root Cause:** DeepSpeed by default tries to use its optimized FusedAdam optimizer, which requires a compiled CUDA extension. In distributed environments or when the extension is not properly compiled, this will fail.

**Solution:** Both `zero2.json` and `zero3.json` have been configured to:
1. Explicitly use PyTorch's native `AdamW` optimizer instead of FusedAdam
2. Set `"zero_allow_untested_optimizer": true` to allow using non-fused optimizers with ZeRO

These configurations ensure that training works without requiring the FusedAdam CUDA extension.

### Key Configuration Options

#### Optimizer Configuration
```json
"optimizer": {
    "type": "AdamW",
    "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto"
    }
}
```

This explicitly tells DeepSpeed to use PyTorch's AdamW optimizer instead of trying to use FusedAdam.

#### ZeRO Allow Untested Optimizer
```json
"zero_allow_untested_optimizer": true
```

This flag is required in the `zero_optimization` section to allow DeepSpeed to use non-fused optimizers (like PyTorch's AdamW) with ZeRO optimization stages.

## Performance Note

While FusedAdam can provide better performance when available, PyTorch's AdamW is a reliable alternative that works in all environments without requiring CUDA extensions to be compiled.
