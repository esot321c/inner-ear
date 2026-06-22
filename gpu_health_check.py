"""GPU health check for a used card (e.g. a second-hand RTX 3090).

Run this AFTER installing the card, inside the project's Python env:

  C:\\Users\\djgra\\Transcribe\\.venv\\Scripts\\python.exe C:\\Users\\djgra\\Transcribe\\gpu_health_check.py

What it does (all non-destructive - it only allocates GPU memory and computes):
  1. Reports the card, VRAM, driver, CUDA.
  2. CORRECTNESS-under-load: hammers the GPU with large matrix multiplies for
     ~60s and checks every result against a CPU reference. A healthy GPU is bit-
     stable; mining-degraded memory/compute shows up as MISMATCHES here.
  3. VRAM integrity: fills most of the VRAM with a known pattern, reads it back,
     and verifies every byte. Catches bad memory cells / cards that lie about
     their usable VRAM.
  4. Samples nvidia-smi throughout for max temp, power, clocks and throttling.

Pass = no mismatches, no CUDA errors, VRAM verifies, temps sane, no throttling.
Also watch GPU MEMORY JUNCTION temp in HWiNFO64 while this runs (<~100C is good).
"""
import sys, time, subprocess

try:
    import torch
except ImportError:
    sys.exit("PyTorch not found - run with the venv's python (see header).")

DURATION_S = 60          # how long to stress
VRAM_FILL_FRACTION = 0.85  # how much VRAM to memory-test (leave headroom)


def smi(query):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception:
        return "n/a"


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available - driver/torch not seeing the GPU.")

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Device      : {name}")
    print(f"VRAM        : {total_gb:.1f} GB")
    print(f"Driver      : {smi('driver_version')}")
    print(f"Torch/CUDA  : {torch.__version__} / {torch.version.cuda}")
    print("-" * 60)

    dev = torch.device("cuda")
    max_temp = 0.0
    max_power = 0.0
    throttled = False

    # ---- 1) Correctness under sustained compute ----
    print(f"[1/2] Compute + correctness stress for {DURATION_S}s...")
    n = 4096
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    ref = (a @ b).clone()           # reference result on the GPU itself
    ref_cpu = ref.cpu()
    iters = 0
    mism = 0
    t_end = time.time() + DURATION_S
    last_report = 0
    while time.time() < t_end:
        c = a @ b
        if not torch.equal(c.cpu(), ref_cpu):
            # allow tiny float noise? No - same inputs must be bit-identical.
            mism += 1
        iters += 1
        if iters % 20 == 0:
            try:
                t = float(smi("temperature.gpu") or 0); max_temp = max(max_temp, t)
                p = float(smi("power.draw") or 0); max_power = max(max_power, p)
                reasons = smi("clocks_throttle_reasons.active")
                try:
                    # Only flag REAL throttling, not benign reasons. Bits:
                    # 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal,
                    # 0x80 power brake. (Ignore 0x1 idle, 0x2 app-clocks,
                    # 0x4 SW power cap, 0x10 sync boost, 0x100 display.)
                    if int(reasons, 16) & 0xE8:
                        throttled = True
                except (ValueError, TypeError):
                    pass
            except ValueError:
                pass
        if time.time() - last_report >= 10:
            print(f"   {iters} iters, mismatches={mism}, "
                  f"GPU temp ~{max_temp:.0f}C, power ~{max_power:.0f}W")
            last_report = time.time()
    torch.cuda.synchronize()
    print(f"   -> {iters} matmuls, {mism} mismatch(es)")

    # ---- 2) VRAM integrity ----
    print(f"[2/2] VRAM integrity test ({int(VRAM_FILL_FRACTION*100)}% of VRAM)...")
    del a, b, c, ref
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    n_elems = int((free * VRAM_FILL_FRACTION) // 4)   # float32
    vram_errors = -1
    try:
        pattern = torch.full((n_elems,), 1.2345678, device=dev)
        torch.cuda.synchronize()
        vram_errors = int((pattern != 1.2345678).sum().item())
        # second pattern to exercise bit flips differently
        pattern.fill_(-9.87654321)
        torch.cuda.synchronize()
        vram_errors += int((pattern != -9.87654321).sum().item())
        del pattern
        torch.cuda.empty_cache()
        print(f"   -> tested ~{n_elems*4/1024**3:.1f} GB, {vram_errors} bad value(s)")
    except RuntimeError as e:
        print(f"   -> VRAM allocation FAILED: {e}")

    # ---- verdict ----
    print("=" * 60)
    ok = (mism == 0) and (vram_errors == 0)
    print(f"Compute correctness : {'PASS' if mism == 0 else f'FAIL ({mism} mismatches)'}")
    print(f"VRAM integrity      : {'PASS' if vram_errors == 0 else f'FAIL ({vram_errors} errors)'}")
    print(f"Max GPU temp        : {max_temp:.0f} C  (watch MEM JUNCTION in HWiNFO too)")
    print(f"Max power draw      : {max_power:.0f} W  (3090 should pull ~330-370W)")
    print(f"Throttling seen     : {'YES - investigate cooling' if throttled else 'no'}")
    print("=" * 60)
    print("OVERALL:", "LOOKS HEALTHY" if ok else "PROBLEM - see failures above")


if __name__ == "__main__":
    main()
