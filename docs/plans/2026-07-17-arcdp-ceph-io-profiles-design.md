# ArcDP Ceph I/O profile design

## Goal

Provide two directly runnable, full-model, one-epoch launch profiles for eight RTX 4090 GPUs so the user can compare a low-risk DataLoader concurrency reduction against a lower-micro-batch profile while keeping the effective global batch size unchanged.

## Context

The path named `/nvme` is a CephFS CSI mount. Sequential read tests report about 260 MB/s, but live ArcDP workers have been observed in `ceph_mdsc_wait_request` and `wait_on_page_bit_common` while opening JPEG and PLY training files. The experiment therefore targets metadata concurrency and tail latency, not peak sequential bandwidth.

## Profiles

### Profile A: batch 48, two workers

- 8 GPUs
- per-device batch: 48
- gradient accumulation: 1
- workers per rank: 2
- default prefetch behavior
- effective global batch: 384

This is the recommended first test because it preserves the current optimization and GPU micro-batch while halving DataLoader process concurrency from 32 to 16.

### Profile B: batch 24, accumulation 2

- 8 GPUs
- per-device micro-batch: 24
- gradient accumulation: 2
- workers per rank: 2
- prefetch factor: 1
- effective global batch: 384

This further reduces queued samples and metadata pressure while preserving the effective global batch and learning rate. It may have slightly lower pure-GPU efficiency because the micro-batch is smaller.

## Launching and isolation

Each profile gets a root-level executable wrapper and a dedicated internal launcher. Both run only the full model for one epoch. Run names, SwanLab experiment names, logs, and checkpoint directories are profile-specific so the experiments cannot auto-resume from each other or from the earlier full-plus-no-bridge run.

## Monitoring correctness

The existing detailed progress callback counts only the per-device batch. It must include world size and gradient accumulation when reporting effective global batch and samples per second. Runtime throughput should be based on optimizer steps completed during the current process invocation, while absolute progress and ETA continue to use the resumed global step.

The launchers print all effective settings before `torchrun` starts. The user should compare at least 100 stable optimizer steps using:

- optimizer-step duration;
- true global samples per second;
- per-GPU SM utilization and power;
- rank balance, especially zero-utilization samples;
- DataLoader worker states and Ceph wait channels;
- GPU memory peak and OOM status.

Profile B is tested only if Profile A still shows persistent rank imbalance, Ceph wait states, or materially poor throughput.

## Safety

- Both profiles use new run names and do not overwrite the current run.
- Neither script attempts to stop an existing training process.
- A preflight check refuses to start if the dataset, index, or model checkpoint is missing.
- SwanLab remains rank-zero-only through the existing compatibility layer.
