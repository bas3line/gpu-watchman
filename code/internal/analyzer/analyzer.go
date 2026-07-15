package analyzer

import (
	"fmt"
	"github.com/gpu-watchman/gpu-watchman/internal/model"
	"strings"
)

func Analyze(gpus []model.GPU) []model.Finding {
	var findings []model.Finding
	for _, g := range gpus {
		memoryPercent := percent(g.MemoryUsedMiB, g.MemoryTotalMiB)
		add := func(severity, code, message string) {
			findings = append(findings, model.Finding{GPUIndex: g.Index, Severity: severity, Code: code, Message: message})
		}
		if g.MemoryTotalMiB == 0 {
			add("critical", "memory-unavailable", "VRAM capacity could not be read")
			continue
		}
		if memoryPercent >= 95 {
			add("critical", "vram-critical", fmt.Sprintf("VRAM is %d%% full (%d/%d MiB)", memoryPercent, g.MemoryUsedMiB, g.MemoryTotalMiB))
		} else if memoryPercent >= 80 {
			add("warning", "vram-high", fmt.Sprintf("VRAM is %d%% full", memoryPercent))
		}
		if g.MemoryUsedMiB > 256 && len(g.Processes) == 0 {
			add("warning", "unattributed-vram", fmt.Sprintf("%d MiB VRAM is used but no compute process was reported", g.MemoryUsedMiB))
		}
		if g.TemperatureC >= 90 {
			add("critical", "temperature-critical", fmt.Sprintf("GPU temperature is %dC", g.TemperatureC))
		} else if g.TemperatureC >= 82 {
			add("warning", "temperature-high", fmt.Sprintf("GPU temperature is %dC", g.TemperatureC))
		}
		if g.PowerLimitW > 0 && g.PowerDrawW/g.PowerLimitW >= .98 {
			add("warning", "power-limit", "power draw is at the configured limit")
		}
		if g.GPUUtilPercent >= 95 {
			add("info", "gpu-saturated", fmt.Sprintf("GPU utilization is %d%%", g.GPUUtilPercent))
		}
		if g.GPUUtilPercent < 5 && memoryPercent >= 50 {
			add("info", "vram-reserved", "substantial VRAM is allocated while the GPU is idle")
		}
		if g.PCIeGenMax > 0 && g.PCIeGenCurrent < g.PCIeGenMax && g.GPUUtilPercent >= 50 {
			add("warning", "pcie-generation", fmt.Sprintf("PCIe is Gen %d; GPU supports Gen %d", g.PCIeGenCurrent, g.PCIeGenMax))
		}
		if g.PCIeWidthMax > 0 && g.PCIeWidthCurrent < g.PCIeWidthMax {
			add("warning", "pcie-width", fmt.Sprintf("PCIe link is x%d; GPU supports x%d", g.PCIeWidthCurrent, g.PCIeWidthMax))
		}
		if g.ECCUncorrected > 0 {
			add("critical", "ecc-uncorrected", fmt.Sprintf("%d uncorrected volatile ECC errors", g.ECCUncorrected))
		}
		if g.ECCCorrected > 0 {
			add("warning", "ecc-corrected", fmt.Sprintf("%d corrected volatile ECC errors", g.ECCCorrected))
		}
		if g.RetiredPages > 0 {
			add("critical", "retired-pages", fmt.Sprintf("%d retired memory pages", g.RetiredPages))
		}
		if len(g.ThrottleReasons) > 0 {
			add("warning", "clock-throttled", "active clock throttle: "+strings.Join(g.ThrottleReasons, ", "))
		}
		if g.MIGMode == "Enabled" {
			add("info", "mig-enabled", "MIG is enabled; inspect instances before attributing whole-GPU VRAM")
		}
		if g.ComputeMode != "Default" && g.ComputeMode != "" {
			add("info", "compute-mode", "non-default compute mode: "+g.ComputeMode)
		}
		if len(g.Processes) > 0 {
			add("info", "processes", fmt.Sprintf("%d compute process(es) using this GPU", len(g.Processes)))
		}
		if g.TemperatureC < 0 || g.FanPercent < 0 {
			add("warning", "sensor-unavailable", "one or more thermal sensors are unavailable")
		}
	}
	return findings
}

func percent(value, total int) int {
	if total == 0 {
		return 0
	}
	return value * 100 / total
}
