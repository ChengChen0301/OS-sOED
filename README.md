This package implements policy gradient methods for learning optimal stopping policies in sequential optimal experimental design (sOED), as described in our paper on curriculum learning approaches for joint design and stopping optimization.

sOED/
├── pg_soed_optimal_stopping.py # Main curriculum learning implementation
├── pg_soed_fixed_stopping.py   # Fixed stopping baseline
├── pg_soed_thresholding.py     # Threshold-based stopping
├── soed.py                      # Base experimental design class
└── utils.py                     # Neural network utilities

examples/
├── PG-sOED-LinearGaussianCase.py        # Linear-Gaussian benchmark
└── PG-sOED-2DConvectionDiffusionCase.py # Sensor movement problem

Key Parameters
Core Setup:

n_stage: Maximum number of experiments
step_cost: Cost per experiment (typically negative)

Curriculum Learning:

stopping_curriculum=True: Enable curriculum approach
stopping_prob_target: Target stopping probability (default: 0.999)
stopping_prob_N_final: Episodes to reach target (default: 10)

Training:

n_update: Number of policy gradient updates
n_traj: Trajectories per update
actor_dimns, critic_dimns: Network architectures
