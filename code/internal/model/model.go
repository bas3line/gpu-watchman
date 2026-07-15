package model

import "time"

type GPU struct {
	Index               int       `json:"index"`
	Name                string    `json:"name"`
	UUID                string    `json:"uuid"`
	Driver              string    `json:"driver"`
	PCIBusID            string    `json:"pci_bus_id"`
	PState              string    `json:"performance_state"`
	TemperatureC        int       `json:"temperature_c"`
	FanPercent          int       `json:"fan_percent"`
	PowerDrawW          float64   `json:"power_draw_w"`
	PowerLimitW         float64   `json:"power_limit_w"`
	GraphicsClockMHz    int       `json:"graphics_clock_mhz"`
	MemoryClockMHz      int       `json:"memory_clock_mhz"`
	MaxGraphicsClockMHz int       `json:"max_graphics_clock_mhz"`
	MaxMemoryClockMHz   int       `json:"max_memory_clock_mhz"`
	MemoryTotalMiB      int       `json:"memory_total_mib"`
	MemoryUsedMiB       int       `json:"memory_used_mib"`
	MemoryFreeMiB       int       `json:"memory_free_mib"`
	GPUUtilPercent      int       `json:"gpu_util_percent"`
	MemoryUtilPercent   int       `json:"memory_util_percent"`
	PCIeGenCurrent      int       `json:"pcie_gen_current"`
	PCIeGenMax          int       `json:"pcie_gen_max"`
	PCIeWidthCurrent    int       `json:"pcie_width_current"`
	PCIeWidthMax        int       `json:"pcie_width_max"`
	ComputeMode         string    `json:"compute_mode"`
	PersistenceMode     bool      `json:"persistence_mode"`
	ECCEnabled          bool      `json:"ecc_enabled"`
	ECCCorrected        int       `json:"ecc_corrected_volatile"`
	ECCUncorrected      int       `json:"ecc_uncorrected_volatile"`
	RetiredPages        int       `json:"retired_pages"`
	MIGMode             string    `json:"mig_mode"`
	ThrottleReasons     []string  `json:"throttle_reasons,omitempty"`
	Processes           []Process `json:"processes"`
}

type Process struct {
	PID       int    `json:"pid"`
	Name      string `json:"name"`
	MemoryMiB int    `json:"memory_mib"`
	Owner     string `json:"owner,omitempty"`
	Command   string `json:"command,omitempty"`
	Cgroup    string `json:"cgroup,omitempty"`
}

type Finding struct {
	GPUIndex int    `json:"gpu_index"`
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Message  string `json:"message"`
}

type Report struct {
	CollectedAt time.Time  `json:"collected_at"`
	GPUs        []GPU      `json:"gpus"`
	Findings    []Finding  `json:"findings"`
	Topology    string     `json:"topology,omitempty"`
	XIDEvents   []string   `json:"xid_events,omitempty"`
	Endpoints   []Endpoint `json:"endpoints,omitempty"`
}

type Endpoint struct {
	URL       string             `json:"url"`
	Reachable bool               `json:"reachable"`
	LatencyMS int64              `json:"latency_ms"`
	Kind      string             `json:"kind"`
	Metrics   map[string]float64 `json:"metrics,omitempty"`
	Failure   string             `json:"failure,omitempty"`
}
