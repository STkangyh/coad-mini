"""
Jetson measurement run: records what the board actually is, then times the
streaming loop while sampling power.

Why this exists rather than just running bench_realtime_incremental.py: on a
Jetson a timing number is meaningless without the power mode and clock state it
was taken under (DVFS moves the clock by >2x between modes), and the one thing
only this hardware can give us -- energy per incremental update -- needs a
sampler running alongside the benchmark.

Ordering is deliberate. Phase 1 needs nothing but CPU torch, and it is the
number our claim actually rests on; the CUDA/TensorRT work that could eat the
whole session comes last, after the essential results are already on disk.

    python3 dev/bench_jetson.py --env                  # phase 0: what is this board
    python3 dev/bench_jetson.py --power                # phase 1: CPU timings + energy
    python3 dev/bench_jetson.py --power --tag 15W      # label the current nvpmodel

Every run appends to reports/jetson_raw.json keyed by tag, so re-running under a
different power mode accumulates the sweep instead of overwriting it.

UNTESTED ON HARDWARE: the tegrastats parsing below was written without a board.
It degrades to timings-only if tegrastats is missing or its format differs --
it will never abort the measurement. Check the reported watts look sane before
trusting the energy column.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.provenance import save_results  # noqa: E402

OUT = ROOT / "reports/jetson_raw.json"

# tegrastats prints e.g. "VDD_GPU_SOC 1234mW/1200mW" -- current/average per rail.
RAIL = re.compile(r"(VDD[_A-Z0-9]*|POM[_A-Z0-9]*)\s+(\d+)mW/(\d+)mW")


def read_env():
    """Everything needed to make the numbers reproducible and citable."""
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    env = {
        "l4t": sh("cat /etc/nv_tegra_release"),
        "model": sh("cat /proc/device-tree/model | tr -d '\\0'"),
        "jetpack": sh("dpkg-query --show nvidia-l4t-core 2>/dev/null"),
        "nvpmodel": sh("nvpmodel -q 2>/dev/null"),
        "cpu_online": sh("cat /sys/devices/system/cpu/online"),
        "cpu_governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_khz": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "mem_kb": sh("grep MemTotal /proc/meminfo"),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.cuda.is_available()
        env["threads"] = torch.get_num_threads()
    except Exception as e:
        env["torch"] = f"import failed: {e}"
    return env


class PowerSampler(threading.Thread):
    """Samples tegrastats in the background; total energy = mean watts x seconds."""

    def __init__(self, interval_ms=100):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.samples = []            # list of {rail: mW}
        self.available = shutil.which("tegrastats") is not None
        self.error = None
        self._proc = None
        self._stop = threading.Event()

    def run(self):
        if not self.available:
            self.error = "tegrastats not on PATH"
            return
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                found = RAIL.findall(line)
                if found:
                    self.samples.append({m[0]: int(m[1]) for m in found})
        except Exception as e:                     # never take down the benchmark
            self.error = str(e)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()

    def summary(self, seconds):
        if not self.samples:
            return {"available": False, "reason": self.error or "no samples parsed"}
        rails = sorted({r for s in self.samples for r in s})
        mean_mw = {r: sum(s.get(r, 0) for s in self.samples) / len(self.samples)
                   for r in rails}
        total_mw = sum(mean_mw.values())
        return {
            "available": True, "n_samples": len(self.samples), "rails": rails,
            "mean_mw_per_rail": mean_mw, "mean_total_mw": total_mw,
            "seconds": seconds, "energy_joules": total_mw / 1000.0 * seconds,
        }


def run_bench(extra_args):
    """Shells out to the verified benchmark so there is one implementation."""
    cmd = [sys.executable, str(ROOT / "dev/bench_realtime_incremental.py")] + extra_args
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
    return dt, p.returncode


def save(tag, payload):
    all_runs = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_runs.setdefault(tag, []).append(payload)
    OUT.parent.mkdir(exist_ok=True)
    save_results(OUT, all_runs)
    print(f"\nraw -> {OUT}  (tag={tag}, run #{len(all_runs[tag])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", action="store_true", help="phase 0: report the board only")
    ap.add_argument("--power", action="store_true", help="sample tegrastats during the run")
    ap.add_argument("--tag", default="default",
                    help="label for this run, e.g. the nvpmodel mode (15W/30W/MAXN)")
    ap.add_argument("--bench-args", default="--frames 40",
                    help="passed through to bench_realtime_incremental.py")
    args = ap.parse_args()

    env = read_env()
    print("=== board ===")
    for k, v in env.items():
        if v is not None:
            print(f"  {k:14s} {str(v).splitlines()[0][:90]}")
    if args.env:
        save(args.tag, {"env": env})
        return

    if env.get("cpu_governor") not in (None, "performance"):
        print(f"\n!! governor is {env['cpu_governor']!r}, not 'performance' -- "
              f"timings will drift. Run `sudo jetson_clocks` first.\n")

    sampler = PowerSampler()
    if args.power:
        sampler.start()
        time.sleep(1.0)                     # let a few idle samples land first

    seconds, rc = run_bench(args.bench_args.split())

    power = {"available": False, "reason": "not requested"}
    if args.power:
        sampler.stop()
        sampler.join(timeout=5)
        power = sampler.summary(seconds)
        if power["available"]:
            print(f"=== power ===\n  mean {power['mean_total_mw']/1000:.2f} W "
                  f"over {seconds:.1f}s = {power['energy_joules']:.1f} J "
                  f"({power['n_samples']} samples)")
        else:
            print(f"=== power ===\n  unavailable: {power['reason']}")

    save(args.tag, {"env": env, "wall_seconds": seconds, "returncode": rc,
                    "power": power, "bench_args": args.bench_args})


if __name__ == "__main__":
    main()
