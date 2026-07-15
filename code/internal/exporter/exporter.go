package exporter

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

type Server struct {
	mu     sync.RWMutex
	report model.Report
}

func New() *Server                        { return &Server{} }
func (s *Server) Set(report model.Report) { s.mu.Lock(); defer s.mu.Unlock(); s.report = report }
func (s *Server) Handler() http.Handler   { return http.HandlerFunc(s.serve) }

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/metrics" {
		http.NotFound(w, r)
		return
	}
	s.mu.RLock()
	report := s.report
	s.mu.RUnlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	for _, g := range report.GPUs {
		labels := fmt.Sprintf(`gpu="%d",uuid="%s",name="%s"`, g.Index, escape(g.UUID), escape(g.Name))
		metric(w, "gpu_watchman_vram_used_mib", labels, float64(g.MemoryUsedMiB))
		metric(w, "gpu_watchman_vram_total_mib", labels, float64(g.MemoryTotalMiB))
		metric(w, "gpu_watchman_utilization_percent", labels, float64(g.GPUUtilPercent))
		metric(w, "gpu_watchman_temperature_celsius", labels, float64(g.TemperatureC))
		metric(w, "gpu_watchman_power_watts", labels, g.PowerDrawW)
		metric(w, "gpu_watchman_processes", labels, float64(len(g.Processes)))
		metric(w, "gpu_watchman_ecc_uncorrected", labels, float64(g.ECCUncorrected))
		metric(w, "gpu_watchman_retired_pages", labels, float64(g.RetiredPages))
		for _, p := range g.Processes {
			metric(w, "gpu_watchman_process_vram_mib", labels+`,pid="`+strconv.Itoa(p.PID)+`",process="`+escape(p.Name)+`"`, float64(p.MemoryMiB))
		}
	}
	for _, endpoint := range report.Endpoints {
		metric(w, "gpu_watchman_inference_endpoint_up", `url="`+escape(endpoint.URL)+`"`, boolValue(endpoint.Reachable))
	}
}

func metric(w http.ResponseWriter, name, labels string, value float64) {
	fmt.Fprintf(w, "%s{%s} %g\n", name, labels, value)
}
func escape(s string) string { return strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), `"`, `\"`) }
func boolValue(v bool) float64 {
	if v {
		return 1
	}
	return 0
}
