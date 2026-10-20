# PWTR method and implementation

## Fixed mixed-pool replay

Each trajectory contains ordered histories and their stored denoising paths.
For a trajectory of T decisions, retain K=min(16,T) points at
`floor(j*(T-1)/(K-1))`, j=0,...,K-1. If K=1, retain index 0. Points are fixed across
all training updates. Trajectories are sampled uniformly, inducing
`q(i)=1/(N*K_tau)` on selected decisions, where N is the pool size. The fixed
recipe uses N=120. Training records and fixed-reference score maps are read from
the local bundle supplied with `--training-root`.

## Path compatibility

For each selected stored path, evaluate the current student snapshot, the initial
student and the task-specific teacher. The score is the negative sum of squared
Euler transition residuals projected to the executed action coordinates:

```
phi_nu = -sum_m ||P[x_(m+1) - x_m - delta_m*v_nu(x_m,s_m|h)]||²
                 / (2*sigma_m²*d)
sigma_m = max(1e-4, 0.025*sqrt(-delta_m))
```

Here d is the number of projected action coordinates. Residuals use FP32 and
reductions use FP64. Scoring uses batch size 1. The score is a compatibility proxy.

## Sampled-prefix weights

For each model, accumulate scores from earlier **selected** decisions in the
same trajectory. The current decision is excluded. Unsampled intervals receive
no interpolation or length multiplier.

```
log_weight = 0.1 * (Phi_current - logsumexp(
    log(2/3) + Phi_initial,
    log(1/3) + Phi_teacher))
```

Normalize exponentiated scores to unit mean under conditional q within each
task×condition group. Clip normalized weights at 2, then renormalize under the
same q. The final normalized weight may exceed 2. Both fixed references evaluate
both trajectory sources. Cached reference scores are reused; student scores and
weights refresh at updates 0,24,48,72.

## Student update

At a stored history, generate a fresh current-student denoising query. Select a
low-noise time using the fixed Beta(5,2) query-index sampler. Query the task teacher
at the same detached state, time and history. Optimize the weighted action-field
MSE with coefficient 10 and average by the trajectory's actual selected-point count.
Query states, teacher targets and weights are stop-gradient. Clip and step once
per sampled trajectory. The trainable modules are MoT and the proprioception
encoder; action-loss gradients also pass through the trainable condition cache.
There is no direct video loss.
