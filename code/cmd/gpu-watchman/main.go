package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gpu-watchman/gpu-watchman/internal/analyzer"
	"github.com/gpu-watchman/gpu-watchman/internal/collector"
	"github.com/gpu-watchman/gpu-watchman/internal/exporter"
	"github.com/gpu-watchman/gpu-watchman/internal/model"
	"github.com/gpu-watchman/gpu-watchman/internal/probe"
	"github.com/gpu-watchman/gpu-watchman/internal/render"
	"github.com/gpu-watchman/gpu-watchman/internal/tracker"
)

const version = "0.2.0"

func main() {
	format := flag.String("format", "text", "output format: text or json")
	watch := flag.Duration("watch", 0, "refresh interval, for example 2s")
	all := flag.Bool("all", false, "include every GPU, including healthy idle GPUs")
	listen := flag.String("listen", "", "serve Prometheus metrics on this address, for example :9400")
	probes := flag.String("probe", "", "comma-separated vLLM, Triton, TGI, or Ollama metrics endpoints")
	history := flag.String("history", "", "append full JSON reports as NDJSON to this path")
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() > 0 && flag.Arg(0) == "version" {
		fmt.Println(version)
		return
	}
	if *format != "text" && *format != "json" {
		fatal("-format must be text or json")
	}

	trends := tracker.New()
	var metrics *exporter.Server
	if *listen != "" {
		metrics = exporter.New()
		go func() {
			fmt.Fprintln(os.Stderr, "gpu-watchman: Prometheus metrics on", *listen)
			fmt.Fprintln(os.Stderr, http.ListenAndServe(*listen, metrics.Handler()))
		}()
		if *watch <= 0 {
			*watch = 5 * time.Second
		}
	}
	run := func() error {
		report, err := collector.New().Collect()
		if err != nil {
			return err
		}
		report.Findings = analyzer.Analyze(report.GPUs)
		report.Findings = append(report.Findings, trends.Observe(report.GPUs, report.CollectedAt)...)
		if len(report.XIDEvents) > 0 {
			report.Findings = append(report.Findings, model.Finding{GPUIndex: -1, Severity: "critical", Code: "xid-events", Message: fmt.Sprintf("%d NVIDIA Xid driver event(s) found in kernel logs", len(report.XIDEvents))})
		}
		report.Endpoints = probe.Collect(strings.Split(*probes, ","))
		for _, endpoint := range report.Endpoints {
			if !endpoint.Reachable {
				report.Findings = append(report.Findings, model.Finding{GPUIndex: -1, Severity: "warning", Code: "inference-endpoint-down", Message: endpoint.URL + ": " + endpoint.Failure})
			}
		}
		if metrics != nil {
			metrics.Set(report)
		}
		if *history != "" {
			if err := appendHistory(*history, report); err != nil {
				return err
			}
		}
		if !*all {
			report.GPUs = render.ActiveGPUs(report.GPUs, report.Findings)
		}
		if *format == "json" {
			return json.NewEncoder(os.Stdout).Encode(report)
		}
		render.Text(os.Stdout, report)
		return nil
	}

	if *watch <= 0 {
		if err := run(); err != nil {
			fatal(err.Error())
		}
		return
	}
	for {
		if err := run(); err != nil {
			fmt.Fprintln(os.Stderr, "gpu-watchman:", err)
		}
		time.Sleep(*watch)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "GPU Watchman - compact NVIDIA GPU diagnostic tool")
	fmt.Fprintln(os.Stderr, "Usage: gpu-watchman [-format text|json] [-watch 2s] [-all] [-listen :9400] [-probe URL,...] [-history reports.ndjson] [version]")
}

func appendHistory(path string, value any) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open history: %w", err)
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(value)
}

func fatal(message string) {
	fmt.Fprintln(os.Stderr, "gpu-watchman:", message)
	os.Exit(1)
}
