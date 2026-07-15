package render

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

func ActiveGPUs(gpus []model.GPU, findings []model.Finding) []model.GPU {
	active := make(map[int]bool)
	for _, f := range findings {
		if f.Severity != "info" || f.Code == "processes" {
			active[f.GPUIndex] = true
		}
	}
	for _, g := range gpus {
		if g.MemoryUsedMiB > 0 || g.GPUUtilPercent > 0 {
			active[g.Index] = true
		}
	}
	var result []model.GPU
	for _, g := range gpus {
		if active[g.Index] {
			result = append(result, g)
		}
	}
	return result
}

func Text(w io.Writer, report model.Report) {
	fmt.Fprintf(w, "GPU Watchman  %s\n", report.CollectedAt.Format("2006-01-02 15:04:05 UTC"))
	if len(report.GPUs) == 0 {
		fmt.Fprintln(w, "No active GPUs. Use -all to show healthy idle GPUs.")
		return
	}
	for _, g := range report.GPUs {
		fmt.Fprintf(w, "\nGPU %d  %s\n", g.Index, g.Name)
		fmt.Fprintf(w, "  VRAM %d/%d MiB (%d%%) | GPU %d%% | Memory bus %d%%\n", g.MemoryUsedMiB, g.MemoryTotalMiB, ratio(g.MemoryUsedMiB, g.MemoryTotalMiB), g.GPUUtilPercent, g.MemoryUtilPercent)
		fmt.Fprintf(w, "  Temp %dC | Fan %d%% | Power %.1f/%.1f W | P-State %s\n", g.TemperatureC, g.FanPercent, g.PowerDrawW, g.PowerLimitW, g.PState)
		fmt.Fprintf(w, "  Clocks %d/%d MHz core, %d/%d MHz memory | PCIe Gen %d/%d x%d/%d\n", g.GraphicsClockMHz, g.MaxGraphicsClockMHz, g.MemoryClockMHz, g.MaxMemoryClockMHz, g.PCIeGenCurrent, g.PCIeGenMax, g.PCIeWidthCurrent, g.PCIeWidthMax)
		fmt.Fprintf(w, "  Driver %s | ECC %t | Persistence %t\n", g.Driver, g.ECCEnabled, g.PersistenceMode)
		if g.MIGMode != "" {
			fmt.Fprintf(w, "  PCI %s | MIG %s\n", g.PCIBusID, g.MIGMode)
		}
		if len(g.ThrottleReasons) > 0 {
			fmt.Fprintf(w, "  Throttle: %s\n", strings.Join(g.ThrottleReasons, ", "))
		}
		for _, p := range g.Processes {
			fmt.Fprintf(w, "  Process %d: %s (%d MiB, %s)\n", p.PID, p.Name, p.MemoryMiB, p.Owner)
		}
	}
	findings := append([]model.Finding(nil), report.Findings...)
	sort.SliceStable(findings, func(i, j int) bool { return rank(findings[i].Severity) < rank(findings[j].Severity) })
	if len(findings) > 0 {
		fmt.Fprintln(w, "\nFindings")
	}
	for _, f := range findings {
		fmt.Fprintf(w, "  [%s] GPU %d %s: %s\n", f.Severity, f.GPUIndex, f.Code, f.Message)
	}
	if report.Topology != "" {
		fmt.Fprintf(w, "\nTopology\n%s\n", report.Topology)
	}
	for _, endpoint := range report.Endpoints {
		fmt.Fprintf(w, "\nEndpoint %s: up=%t latency=%dms metrics=%d\n", endpoint.URL, endpoint.Reachable, endpoint.LatencyMS, len(endpoint.Metrics))
	}
}

func ratio(value, total int) int {
	if total == 0 {
		return 0
	}
	return value * 100 / total
}
func rank(s string) int {
	if s == "critical" {
		return 0
	}
	if s == "warning" {
		return 1
	}
	return 2
}
