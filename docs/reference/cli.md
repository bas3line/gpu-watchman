# CLI Reference

```text
gpu-watchman [-format text|json] [-watch 2s] [-all] [-listen :9400] [-probe URL,...] [-history reports.ndjson] [version]
```

| Option | Default | Description |
| --- | --- | --- |
| `-format text|json` | `text` | Terminal report or one JSON report per collection cycle |
| `-watch duration` | disabled | Repeat collection at a Go duration such as `2s` or `1m` |
| `-all` | false | Keep healthy idle GPUs in output |
| `-listen address` | disabled | Serve latest report metrics at `/metrics`; also enables a 5 second watch interval if unset |
| `-probe URL,...` | disabled | Probe each base URL at `/metrics` |
| `-history path` | disabled | Append full report JSON as NDJSON |
| `version` | n/a | Print the version and exit |

## Examples

```sh
# One concise terminal snapshot
gpu-watchman

# Full inventory for a support bundle
gpu-watchman -all -format json

# Production node collection
gpu-watchman -all -watch 5s -listen :9400 -history /var/log/gpu-watchman.ndjson

# Inference-node collection with two model servers
gpu-watchman -watch 5s -probe http://vllm:8000,http://triton:8002
```
