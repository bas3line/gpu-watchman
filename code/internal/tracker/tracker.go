package tracker

import (
	"fmt"
	"time"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

type sample struct {
	memory int
	at     time.Time
}
type Tracker struct{ processes map[string]sample }

func New() *Tracker { return &Tracker{processes: make(map[string]sample)} }

// Observe identifies sustained per-process VRAM growth between collection samples.
func (t *Tracker) Observe(gpus []model.GPU, now time.Time) []model.Finding {
	var findings []model.Finding
	seen := make(map[string]bool)
	for _, gpu := range gpus {
		for _, p := range gpu.Processes {
			key := fmt.Sprintf("%s/%d", gpu.UUID, p.PID)
			seen[key] = true
			if previous, ok := t.processes[key]; ok && now.Sub(previous.at) >= time.Second {
				growth := p.MemoryMiB - previous.memory
				if growth >= 256 {
					findings = append(findings, model.Finding{GPUIndex: gpu.Index, Severity: "warning", Code: "vram-growth", Message: fmt.Sprintf("process %d (%s) grew by %d MiB since the prior sample", p.PID, p.Name, growth)})
				}
			}
			t.processes[key] = sample{memory: p.MemoryMiB, at: now}
		}
	}
	for key := range t.processes {
		if !seen[key] {
			delete(t.processes, key)
		}
	}
	return findings
}
