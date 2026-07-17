# ArcDP safe percentage checkpoint design

## Goal

Make the one-epoch 8x4090 ArcDP run resumable and resistant to partial or concurrent checkpoint writes while retaining checkpoints by training percentage.

## Approved retention policy

- Save at every integer percentage from 1% through 100%.
- Treat 10%, 20%, ..., 100% as permanent checkpoints.
- Keep only the newest five completed non-permanent checkpoints.
- Keep permanent checkpoints outside rolling deletion.
- Resume from the newest completed checkpoint after interruption.

For 25,590 optimizer steps, targets use `ceil(total_steps * percent / 100)`, so 1% is step 256, 10% is step 2,559, and 100% is step 25,590.

## Checkpoint write path

Transformers invokes checkpoint handling on every DDP rank. `BridgeDPTrainer.save_model()` must return immediately when `args.should_save` is false so only world rank zero writes model files. Rank zero writes `pytorch_model.bin` through a temporary file, flushes it, and atomically replaces the final path. `bridgedp.ckpt` is a hard-link alias when supported, with an atomic copy fallback.

Transformers continues to save optimizer, scheduler, trainer state, and per-rank RNG state. After every rank reaches `on_save`, a distributed barrier ensures all files have landed. Rank zero then loads the model, optimizer, scheduler, and RNG files, validates the trainer step, writes `CHECKPOINT_COMPLETE.json` atomically, updates a manifest, and applies retention. Any validation failure is broadcast and fails every rank instead of silently continuing.

## Resume safety

The new safe profile requires `CHECKPOINT_COMPLETE.json`. Auto-resume ignores directories without the marker or without any marker-listed file. Existing legacy profiles keep their current compatibility behavior unless they explicitly enable strict complete-checkpoint validation.

## Final checkpoint

The 100% target requests a normal checkpoint. After `trainer.train()` returns, every rank verifies that the final checkpoint has a valid 100% completion marker. If the normal callback did not create it, all ranks invoke one fallback checkpoint save and validate it before the process exits.

## Isolation

The new profile keeps B+ compute settings: batch 24, gradient accumulation 2, four workers, prefetch factor 1, eight GPUs, and one epoch. It receives a unique profile name, run name, SwanLab experiment, master port, checkpoint directory, log, and root wrapper. Existing A/B/B+ runs are not reused.

## Storage model

At most ten permanent checkpoints plus five rolling checkpoints remain after completion. The two model filenames share one inode when CephFS supports hard links, reducing duplicate model storage. Checkpoint creation may temporarily need space for the prior rolling set plus the new checkpoint and temporary model file.

## Verification

- Pure tests cover exact percentage targets and rolling/permanent retention.
- Resume tests prove incomplete higher-step checkpoints are skipped.
- Static tests prove rank-zero guarding, atomic model replacement, callback integration, strict resume, and the final fallback.
- Bash syntax and dry-run tests verify the isolated launch profile.
