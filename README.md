## Optimal-stopping control for Bayesian sequential experimental design (OS-sOED)

This package implements policy gradient methods for learning optimal stopping policies in sequential optimal experimental design (sOED), as described in our paper on curriculum learning approaches for joint design and stopping optimization.

## Key Parameters

- n_stage: Maximum number of experiments
- step_cost: Cost per experiment (typically negative)
- stopping_curriculum=True: Enable curriculum approach
- n_update: Number of policy gradient updates
- n_traj: Trajectories per update
