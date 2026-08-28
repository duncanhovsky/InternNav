# ArcDP Few-Step Inference Design

## Goal

Make ArcDP inference budgets of 2, 4, 8, and 10 model evaluations valid for the existing 10-step checkpoint. Reduced-step inference must span the complete reverse bridge interval rather than taking a low-noise prefix.

## Selected Design

The scheduler selects rounded, evenly spaced indices from training timestep `T-1` to timestep `0`. For `T=10`, the schedules are `9,0`, `9,6,3,0`, `9,8,6,5,4,3,1,0`, and the original `9,...,0` sequence.

The reverse bridge update receives the next selected timestep. It therefore jumps from the current normalized diffusion time to the actual next normalized diffusion time; the last update returns the predicted clean trajectory. The ten-step path remains identical to the existing implementation.

The training process, checkpoint tensors, trajectory-time ordering, and forward noising process are unchanged. Only inference scheduling changes.

## Validation

Focused scheduler tests cover exact schedules, invalid budgets, sparse jumps, and ten-step backward compatibility. Deployment-side ArcDP and NavDP experiment entrypoints live in the NavDP repository and use the same 2/4/8/10 matrix.

