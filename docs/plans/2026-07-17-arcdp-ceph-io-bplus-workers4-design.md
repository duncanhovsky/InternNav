# ArcDP Ceph I/O B+ workers4 design

## Goal

Add a directly runnable, isolated B+ profile that increases DataLoader concurrency while preserving every optimization-semantic setting from profile B.

## Runtime configuration

- GPUs: 8 RTX 4090
- epochs: 1
- per-device micro-batch: 24
- gradient accumulation: 2
- workers per rank: 4
- prefetch factor per worker: 1
- effective global batch: `24 * 8 * 2 = 384`
- model: full ArcDP only
- monitoring: SwanLab

Profile B+ doubles DataLoader workers from 2 to 4 while keeping prefetch at 1. It can therefore prepare more samples concurrently without restoring the more aggressive per-worker prefetch used by the original training profile. Across eight ranks it uses 32 DataLoader workers.

## Isolation

The shared Ceph I/O launcher receives a new `b24_ga2_w4_pf1` profile. A dedicated root wrapper launches only that profile. The profile uses its own run name, SwanLab experiment, master port, checkpoint directory, log filename, and PID filename so it cannot auto-resume from A or B.

## Safety and operation

The existing busy-GPU guard remains active, so B must be stopped before B+ starts. Dataset, preload-index, checkpoint, CUDA, GPU-count, and SwanLab preflight checks remain unchanged. The implementation does not modify or delete A/B outputs.

## Success criteria

After at least 100 optimizer steps, compare B+ with B using the same effective global batch. B+ is useful when average optimizer-step time is materially below B's approximately 41.5 seconds without OOM, persistent rank imbalance, or repeated Ceph `D`-state workers. Reaching the ten-day target requires approximately 33.76 seconds per optimizer step or less.

## Verification

- Static tests assert the exact B+ batch, accumulation, worker, prefetch, run-name, and wrapper mapping.
- Shell syntax checks cover the shared launcher and the new wrapper.
- A dry run must print 8 GPUs, one epoch, batch 24, accumulation 2, workers 4, prefetch 1, effective global batch 384, and the isolated B+ run name.
