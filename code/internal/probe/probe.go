package probe

import (
	"bufio"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

func Collect(urls []string) []model.Endpoint {
	var endpoints []model.Endpoint
	client := &http.Client{Timeout: 3 * time.Second}
	for _, base := range urls {
		base = strings.TrimRight(strings.TrimSpace(base), "/")
		if base == "" {
			continue
		}
		endpoints = append(endpoints, collectOne(client, base))
	}
	return endpoints
}

func collectOne(client *http.Client, base string) model.Endpoint {
	started := time.Now()
	response, err := client.Get(base + "/metrics")
	endpoint := model.Endpoint{URL: base, LatencyMS: time.Since(started).Milliseconds(), Kind: "metrics"}
	if err != nil {
		endpoint.Failure = err.Error()
		return endpoint
	}
	defer response.Body.Close()
	endpoint.Reachable = response.StatusCode >= 200 && response.StatusCode < 400
	if !endpoint.Reachable {
		endpoint.Failure = response.Status
		return endpoint
	}
	endpoint.Metrics = make(map[string]float64)
	scanner := bufio.NewScanner(response.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		// Keep only inference-relevant values; labels remain in the metric key.
		if strings.Contains(fields[0], "vllm") || strings.Contains(fields[0], "triton") || strings.Contains(fields[0], "tgi") || strings.Contains(fields[0], "ollama") {
			if value, err := strconv.ParseFloat(fields[1], 64); err == nil {
				endpoint.Metrics[fields[0]] = value
			}
		}
	}
	if err := scanner.Err(); err != nil {
		endpoint.Reachable = false
		endpoint.Failure = fmt.Sprintf("read metrics: %v", err)
		return endpoint
	}
	if len(endpoint.Metrics) == 0 {
		endpoint.Kind = "generic"
	}
	return endpoint
}
