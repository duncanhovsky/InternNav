"""Distributed, verified percentage checkpoint management for ArcDP training."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from scripts.train.base_train.percent_checkpoint_utils import (
    COMPLETE_MARKER_FILE,
    PercentCheckpointTarget,
    apply_percent_checkpoint_retention,
    atomic_write_json,
    compute_percent_checkpoint_targets,
)


MARKER_DISPLAY_NAME = "CHECKPOINT_COMPLETE.json"


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _broadcast_error(error: str | None) -> None:
    payload = [error]
    if dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else None
        dist.broadcast_object_list(payload, src=0, device=device)
    if payload[0]:
        raise RuntimeError(payload[0])


class PercentCheckpointCallback(TrainerCallback):
    """Request, validate, mark, and retain resumable checkpoints by training percentage."""

    def __init__(
        self,
        interval_percent: int = 1,
        rolling_keep: int = 5,
        permanent_interval_percent: int = 10,
        verify_loads: bool = True,
    ):
        self.interval_percent = max(int(interval_percent), 1)
        self.rolling_keep = max(int(rolling_keep), 1)
        self.permanent_interval_percent = max(int(permanent_interval_percent), 1)
        self.verify_loads = bool(verify_loads)
        self._targets_for_max_steps: int | None = None
        self._targets_by_step: dict[int, PercentCheckpointTarget] = {}

    def _ensure_targets(self, max_steps: int) -> None:
        max_steps = max(int(max_steps), 1)
        if self._targets_for_max_steps == max_steps:
            return
        targets = compute_percent_checkpoint_targets(
            total_steps=max_steps,
            interval_percent=self.interval_percent,
            permanent_interval_percent=self.permanent_interval_percent,
        )
        self._targets_by_step = {target.step: target for target in targets}
        self._targets_for_max_steps = max_steps

    def _target_for_state(self, state) -> PercentCheckpointTarget | None:
        self._ensure_targets(state.max_steps)
        return self._targets_by_step.get(int(state.global_step))

    def on_train_begin(self, args, state, control, **kwargs):
        self._ensure_targets(state.max_steps)
        if _is_main_process():
            permanent_count = sum(target.permanent for target in self._targets_by_step.values())
            print(
                "[PercentCheckpoint] enabled: "
                f"targets={len(self._targets_by_step)}, permanent={permanent_count}, "
                f"rolling_keep={self.rolling_keep}, marker={MARKER_DISPLAY_NAME}"
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._target_for_state(state) is not None:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        target = self._target_for_state(state)
        if target is not None:
            self._verify_and_finalize_checkpoint(args, state, target)
        return control

    @staticmethod
    def _required_files(world_size: int) -> list[str]:
        required = [
            "pytorch_model.bin",
            "bridgedp.ckpt",
            "training_args.bin",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
        ]
        if int(world_size) > 1:
            required.extend(f"rng_state_{rank}.pth" for rank in range(int(world_size)))
        else:
            required.append("rng_state.pth")
        return required

    @staticmethod
    def _torch_load(path: Path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def _validate_checkpoint(self, checkpoint_dir: Path, state, target: PercentCheckpointTarget) -> dict:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint directory was not created: {checkpoint_dir}")

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        required_files = self._required_files(world_size)
        file_sizes: dict[str, int] = {}
        for relative_name in required_files:
            path = checkpoint_dir / relative_name
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint file: {path}")
            size = path.stat().st_size
            if size <= 0:
                raise RuntimeError(f"empty checkpoint file: {path}")
            file_sizes[relative_name] = size

        trainer_state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
        saved_step = int(trainer_state.get("global_step", -1))
        if saved_step != int(state.global_step) or saved_step != target.step:
            raise RuntimeError(
                f"trainer state step mismatch: saved={saved_step}, runtime={state.global_step}, target={target.step}"
            )

        model_tensor_count = None
        if self.verify_loads:
            model_state = self._torch_load(checkpoint_dir / "pytorch_model.bin")
            if not isinstance(model_state, dict) or not model_state:
                raise RuntimeError("pytorch_model.bin did not contain a non-empty state dict")
            model_tensor_count = len(model_state)
            del model_state
            optimizer_state = self._torch_load(checkpoint_dir / "optimizer.pt")
            if not isinstance(optimizer_state, dict) or "state" not in optimizer_state:
                raise RuntimeError("optimizer.pt did not contain optimizer state")
            del optimizer_state
            scheduler_state = self._torch_load(checkpoint_dir / "scheduler.pt")
            if not isinstance(scheduler_state, dict):
                raise RuntimeError("scheduler.pt did not contain a state dict")
            del scheduler_state
            rng_names = [name for name in required_files if name.startswith("rng_state")]
            for rng_name in rng_names:
                rng_state = self._torch_load(checkpoint_dir / rng_name)
                if not isinstance(rng_state, dict):
                    raise RuntimeError(f"{rng_name} did not contain an RNG state dict")

        if file_sizes["bridgedp.ckpt"] != file_sizes["pytorch_model.bin"]:
            raise RuntimeError("BridgeDP and Transformers model weight sizes do not match")

        return {
            "format_version": 1,
            "verified_at": datetime.now().isoformat(),
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "percent": target.percent,
            "permanent": target.permanent,
            "required_files": required_files,
            "file_sizes": file_sizes,
            "model_tensor_count": model_tensor_count,
        }

    def _verify_and_finalize_checkpoint(self, args, state, target: PercentCheckpointTarget) -> Path:
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if dist.is_initialized():
            dist.barrier()

        error: str | None = None
        if _is_main_process():
            try:
                marker = self._validate_checkpoint(checkpoint_dir, state, target)
                atomic_write_json(checkpoint_dir / COMPLETE_MARKER_FILE, marker)
                removed = apply_percent_checkpoint_retention(args.output_dir, rolling_keep=self.rolling_keep)
                print(
                    f"[PercentCheckpoint] verified {target.percent}% at step {target.step}: {checkpoint_dir}; "
                    f"permanent={int(target.permanent)}, pruned={len(removed)}"
                )
            except Exception as exc:
                error = f"checkpoint verification failed for {checkpoint_dir}: {type(exc).__name__}: {exc}"
        _broadcast_error(error)
        return checkpoint_dir

    def ensure_final_checkpoint(self, trainer) -> Path:
        """Verify the 100% checkpoint, creating one through Trainer's full save path if absent."""
        state = trainer.state
        self._ensure_targets(state.max_steps)
        final_target = max(self._targets_by_step.values(), key=lambda target: target.percent)
        if int(state.global_step) != final_target.step or final_target.percent != 100:
            raise RuntimeError(
                f"training ended before the 100% target: step={state.global_step}, expected={final_target.step}"
            )

        checkpoint_dir = Path(trainer.args.output_dir) / f"checkpoint-{state.global_step}"
        need_fallback = [False]
        if _is_main_process():
            marker_path = checkpoint_dir / COMPLETE_MARKER_FILE
            need_fallback[0] = not marker_path.is_file()
        if dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else None
            dist.broadcast_object_list(need_fallback, src=0, device=device)
        if need_fallback[0]:
            if _is_main_process():
                print(f"[PercentCheckpoint] final marker missing; running fallback save: {checkpoint_dir}")
            trainer._save_checkpoint(trainer.model, trial=None)

        return self._verify_and_finalize_checkpoint(trainer.args, state, final_target)
