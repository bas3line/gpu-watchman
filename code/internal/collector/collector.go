package collector

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

type Collector struct{ command string }

func New() Collector { return Collector{command: "nvidia-smi"} }

func (c Collector) Collect() (model.Report, error) {
	if _, err := exec.LookPath(c.command); err != nil {
		return model.Report{}, fmt.Errorf("%s was not found; install an NVIDIA driver and ensure nvidia-smi is in PATH", c.command)
	}
	fields := []string{"index", "name", "uuid", "driver_version", "pci.bus_id", "pstate", "temperature.gpu", "fan.speed", "power.draw", "power.limit", "clocks.current.graphics", "clocks.current.memory", "clocks.max.graphics", "clocks.max.memory", "memory.total", "memory.used", "memory.free", "utilization.gpu", "utilization.memory", "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max", "compute_mode", "persistence_mode", "ecc.mode.current", "ecc.errors.corrected.volatile.total", "ecc.errors.uncorrected.volatile.total", "retired_pages.single_bit_ecc.count", "retired_pages.double_bit_ecc.count"}
	out, err := exec.Command(c.command, "--query-gpu="+strings.Join(fields, ","), "--format=csv,noheader,nounits").Output()
	if err != nil {
		return model.Report{}, fmt.Errorf("could not query GPUs: %w", err)
	}
	gpus, err := parseGPUs(string(out))
	if err != nil {
		return model.Report{}, err
	}
	processes := c.processes()
	for i := range gpus {
		gpus[i].Processes = processes[gpus[i].UUID]
		gpus[i].MIGMode = c.optionalValue(gpus[i].UUID, "mig.mode.current")
		gpus[i].ThrottleReasons = c.throttleReasons(gpus[i].UUID)
	}
	return model.Report{CollectedAt: time.Now().UTC(), GPUs: gpus, Topology: c.topology(), XIDEvents: c.xidEvents()}, nil
}

func (c Collector) processes() map[string][]model.Process {
	result := make(map[string][]model.Process)
	seen := make(map[string]bool)
	for _, kind := range []string{"compute", "graphics"} {
		out, err := exec.Command(c.command, "--query-"+kind+"-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits").Output()
		if err != nil {
			continue
		} // No processes or an unsupported accounting mode are normal responses.
		r := csv.NewReader(strings.NewReader(string(out)))
		for {
			record, e := r.Read()
			if e != nil {
				break
			}
			if len(record) != 4 {
				continue
			}
			key := clean(record[0]) + ":" + strings.TrimSpace(record[1])
			if seen[key] {
				continue
			}
			seen[key] = true
			p := model.Process{PID: integer(record[1]), Name: clean(record[2]), MemoryMiB: integer(record[3])}
			p.Owner, p.Command, p.Cgroup = processInfo(p.PID)
			result[clean(record[0])] = append(result[clean(record[0])], p)
		}
	}
	return result
}

func parseGPUs(raw string) ([]model.GPU, error) {
	r := csv.NewReader(strings.NewReader(raw))
	var gpus []model.GPU
	for {
		x, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read nvidia-smi CSV: %w", err)
		}
		if len(x) != 30 {
			return nil, fmt.Errorf("unexpected nvidia-smi response with %d fields", len(x))
		}
		gpus = append(gpus, model.GPU{Index: integer(x[0]), Name: clean(x[1]), UUID: clean(x[2]), Driver: clean(x[3]), PCIBusID: clean(x[4]), PState: clean(x[5]), TemperatureC: integer(x[6]), FanPercent: integer(x[7]), PowerDrawW: decimal(x[8]), PowerLimitW: decimal(x[9]), GraphicsClockMHz: integer(x[10]), MemoryClockMHz: integer(x[11]), MaxGraphicsClockMHz: integer(x[12]), MaxMemoryClockMHz: integer(x[13]), MemoryTotalMiB: integer(x[14]), MemoryUsedMiB: integer(x[15]), MemoryFreeMiB: integer(x[16]), GPUUtilPercent: integer(x[17]), MemoryUtilPercent: integer(x[18]), PCIeGenCurrent: integer(x[19]), PCIeGenMax: integer(x[20]), PCIeWidthCurrent: integer(x[21]), PCIeWidthMax: integer(x[22]), ComputeMode: clean(x[23]), PersistenceMode: strings.EqualFold(clean(x[24]), "Enabled"), ECCEnabled: strings.EqualFold(clean(x[25]), "Enabled"), ECCCorrected: integer(x[26]), ECCUncorrected: integer(x[27]), RetiredPages: integer(x[28]) + integer(x[29])})
	}
	return gpus, nil
}

func clean(s string) string { return strings.TrimSpace(strings.TrimSuffix(s, " [Not Supported]")) }
func integer(s string) int {
	n, _ := strconv.Atoi(strings.TrimSpace(strings.TrimSuffix(clean(s), " %")))
	return n
}
func decimal(s string) float64 { n, _ := strconv.ParseFloat(strings.TrimSpace(clean(s)), 64); return n }

func (c Collector) optionalValue(uuid, field string) string {
	out, err := exec.Command(c.command, "--id="+uuid, "--query-gpu="+field, "--format=csv,noheader,nounits").Output()
	if err != nil {
		return ""
	}
	value := clean(string(out))
	if value == "N/A" || value == "[Not Supported]" {
		return ""
	}
	return value
}

func (c Collector) throttleReasons(uuid string) []string {
	fields := []struct{ field, label string }{{"clocks_event_reasons.sw_power_cap", "software power cap"}, {"clocks_event_reasons.hw_thermal_slowdown", "hardware thermal slowdown"}, {"clocks_event_reasons.sw_thermal_slowdown", "software thermal slowdown"}, {"clocks_event_reasons.hw_power_brake_slowdown", "external power brake"}, {"clocks_event_reasons.hw_slowdown", "hardware slowdown"}}
	var active []string
	for _, item := range fields {
		if strings.EqualFold(c.optionalValue(uuid, item.field), "Active") {
			active = append(active, item.label)
		}
	}
	return active
}

func (c Collector) topology() string {
	out, err := exec.Command(c.command, "topo", "-m").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func (c Collector) xidEvents() []string {
	for _, args := range [][]string{{"-k", "-n", "300", "--no-pager"}, {"--color=never"}} {
		command := "journalctl"
		if len(args) == 1 {
			command = "dmesg"
		}
		out, err := exec.Command(command, args...).Output()
		if err != nil {
			continue
		}
		var events []string
		for _, line := range strings.Split(string(out), "\n") {
			if strings.Contains(line, "Xid") {
				events = append(events, strings.TrimSpace(line))
			}
		}
		return events
	}
	return nil
}

func processInfo(pid int) (string, string, string) {
	base := filepath.Join("/proc", strconv.Itoa(pid))
	command, _ := osRead(base + "/cmdline")
	command = strings.ReplaceAll(command, "\x00", " ")
	cgroup, _ := osRead(base + "/cgroup")
	status, _ := osRead(base + "/status")
	for _, line := range strings.Split(status, "\n") {
		if strings.HasPrefix(line, "Uid:") {
			parts := strings.Fields(line)
			if len(parts) > 1 {
				if u, err := user.LookupId(parts[1]); err == nil {
					return u.Username, strings.TrimSpace(command), strings.TrimSpace(cgroup)
				}
				return parts[1], strings.TrimSpace(command), strings.TrimSpace(cgroup)
			}
		}
	}
	return "", strings.TrimSpace(command), strings.TrimSpace(cgroup)
}

var osRead = func(name string) (string, error) { b, err := os.ReadFile(name); return string(b), err }
