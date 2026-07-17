"""Dependency-free progress arithmetic for distributed training callbacks."""


def compute_progress_metrics(
    *,
    global_step: int,
    max_steps: int,
    elapsed_seconds: float,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    start_global_step: int = 0,
    dataset_size: int = 0,
):
    """Return optimizer-step and true global-throughput metrics."""

    global_step = max(int(global_step), 0)
    max_steps = max(int(max_steps), 0)
    start_global_step = max(int(start_global_step), 0)
    per_device_batch_size = max(int(per_device_batch_size), 1)
    world_size = max(int(world_size), 1)
    gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
    elapsed_seconds = max(float(elapsed_seconds), 0.0)

    effective_global_batch = per_device_batch_size * world_size * gradient_accumulation_steps
    optimizer_steps_this_run = max(global_step - start_global_step, 0)
    samples_completed = global_step * effective_global_batch
    samples_this_run = optimizer_steps_this_run * effective_global_batch

    if optimizer_steps_this_run > 0 and elapsed_seconds > 0:
        avg_step_time = elapsed_seconds / optimizer_steps_this_run
        samples_per_second = samples_this_run / elapsed_seconds
        eta_seconds = max(max_steps - global_step, 0) * avg_step_time
    else:
        avg_step_time = 0.0
        samples_per_second = 0.0
        eta_seconds = 0.0

    return {
        "effective_global_batch": effective_global_batch,
        "optimizer_steps_this_run": optimizer_steps_this_run,
        "samples_completed": samples_completed,
        "samples_this_run": samples_this_run,
        "total_samples": max_steps * effective_global_batch,
        "samples_per_second": samples_per_second,
        "avg_step_time": avg_step_time,
        "eta_seconds": eta_seconds,
        "progress_pct": global_step / max(max_steps, 1) * 100.0,
        "steps_per_epoch": max(int(dataset_size), 0) // effective_global_batch,
    }
