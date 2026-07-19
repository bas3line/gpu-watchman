# Hardware health and transport

## Default rules

| Signal | Finding | Default condition |
| --- | --- | --- |
| VRAM | vram-high | At least 90 percent |
| VRAM | vram-critical | At least 99 percent |
| Temperature | temperature-high | At least 82 C |
| Temperature | temperature-critical | At least 90 C |
| Power | power-limit | Draw at least 98 percent of limit |
| PCIe generation | pcie-generation | Below max while GPU utilization is at least 50 percent |
| PCIe width | pcie-width | Below max while GPU utilization is at least 50 percent |
| ECC corrected | ecc-corrected | Volatile counter above zero |
| ECC uncorrected | ecc-uncorrected | Volatile counter above zero |
| Retired pages | retired-pages | Count above zero |
| Driver throttle | clock-throttled | Optional throttle reason active |
| Xid | xid-events | Accessible recent kernel log contains NVIDIA Xid |
| Temperature sensor | sensor-unavailable | Temperature is unsupported or unreadable |
| Required telemetry | telemetry-source-partial | Required source retained only partial evidence |
| Required telemetry | telemetry-source-required | Required source is unavailable, skipped, or absent |

VRAM, temperature, and KV-cache thresholds are configurable. Warning values must be below critical values.

## Signal quality

- PCIe downgrade warnings require meaningful load; idle links commonly enter lower-power states.
- Missing fan telemetry is normal on passively cooled datacenter GPUs and does not create a finding.
- Corrected ECC is a warning because trend matters. Uncorrected ECC and retired pages are critical.
- Xid access is best effort. Service accounts and containers commonly cannot read the kernel log; the report records **kernel.xid** as unavailable instead of implying that no events exist. Use **--no-xid** when denial is intentional; that records the source as skipped.
- Power limiting is informational by default because a deliberately capped, compute-bound workload can be healthy.

Every actionable finding carries a concise recommendation in JSON and the terminal view.

## Telemetry trust policy

Schema-v3 reports describe each NVIDIA inventory, process, optional-field, topology, and Xid source as **ok**, **partial**, **unavailable**, or **skipped**. Source evidence includes collection time, retained record count, and a one-line error capped at 512 bytes. A legitimate zero-record result, such as no active GPU processes, remains **ok**; sources that expect per-GPU rows or a topology matrix become **partial** when those records are missing.

The terminal **SOURCES** section shows partial, unavailable, and skipped sources without turning them into health findings by default. **--details** shows healthy sources too. This lets an operator distinguish an empty process list from failed process accounting without making every intentionally unavailable capability unhealthy.

Use **--require-source** when a deployment depends on that evidence:

~~~sh
gpu-watchman snapshot --require-source processes,topology --format json

# Kernel-log visibility is mandatory on this observer.
gpu-watchman serve --require-source xid --listen 127.0.0.1:9400 \
  --api-token-file /run/secrets/watchman-api-token
~~~

Accepted aliases are **inventory**, **processes**, **optional**, **topology**, and **xid**; reports retain their canonical stable names. A required partial source produces a warning finding. A required unavailable, skipped, or absent source produces a critical finding. Both fail closed with exit 2 for one-shot collection. Continuous workflows keep serving and recording the unhealthy report unless **--fail-on** requests termination.

## Failure policy

~~~sh
# CI or scheduler preflight
gpu-watchman snapshot --format json --fail-on critical

# Strict node admission
gpu-watchman snapshot --all --fail-on warning
~~~

Exit 2 indicates the selected policy was crossed. Exit 1 is reserved for failure to collect or encode the report.

Doctor also prints one check per collected source. **PASS** means source state **ok**; other source states are **WARN** with the bounded diagnostic and remediation hint. A failed base GPU collection remains **FAIL** and makes doctor exit 2.
