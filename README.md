## Optimal-stopping control for Bayesian sequential experimental design (OS-sOED)

This package implements policy gradient methods for learning optimal stopping policies in sequential optimal experimental design (sOED), as described in our paper on curriculum learning approaches for joint design and stopping optimization.

## Dependencies

- Numpy 1.21.5
- emcee 3.0.2
- scikit-learn 1.3.2
- Pytorch 2.0.0

## Key Parameters

- n_stage: Maximum number of experiments
- step_cost: Cost per experiment (typically negative)
- stopping_curriculum=True: Enable curriculum approach
- n_update: Number of policy gradient updates
- n_traj: Trajectories per update
