# GPU Watchman

Small, read-only NVIDIA GPU diagnostics for inference operations.

It identifies VRAM owners, checks thermal and memory health, tracks process VRAM growth, probes inference endpoints, exports Prometheus metrics, and records reports as NDJSON.

Start with the [operator documentation](docs/README.md).

## Install

```sh
curl -fsSL https://tools.yshubham.com/watchman/install.sh | sh
```

Install to a user-writable directory when `/usr/local/bin` requires elevated permissions:

```sh
GPU_WATCHMAN_INSTALL_DIR="$HOME/.local/bin" \
  curl -fsSL https://tools.yshubham.com/watchman/install.sh | sh
```

```sh
cd code
go build -trimpath -ldflags='-s -w' -o gpu-watchman ./cmd/gpu-watchman
./gpu-watchman -all
```

Requirements: an NVIDIA driver and `nvidia-smi` in `PATH`.
