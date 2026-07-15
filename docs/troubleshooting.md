# Troubleshooting

## `nvidia-smi was not found`

Install or repair the NVIDIA driver and make sure `nvidia-smi` is in `PATH`. Watchman intentionally does not bundle NVML or a driver.

## No GPU Process Is Listed

NVIDIA process accounting can be unavailable or incomplete depending on driver, permissions, workload type, and host configuration. Watchman queries both compute and graphics process lists. If VRAM is above 256 MiB with no reported process, it emits `unattributed-vram`.

## Optional Fields Are Missing

MIG, throttle reasons, topology, ECC, retired pages, and Xid logs are conditional capabilities. Check the GPU model, driver support, and host permissions. Their absence does not invalidate base collection.

## Endpoint Probe Is Down

`-probe http://server:port` always requests `http://server:port/metrics`. Verify that endpoint, network path, and the three-second timeout expectation. A non-2xx/non-3xx response or read failure produces `inference-endpoint-down`.

## Prometheus Has No Data

Start with both a listener and a watch interval:

```sh
gpu-watchman -watch 5s -listen :9400
```

Check `curl http://localhost:9400/metrics`. The server exposes only the latest successful report; it is not a persistent metrics database.
