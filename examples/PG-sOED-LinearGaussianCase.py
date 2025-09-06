import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator, LinearLocator


from sOED import SOED
from sOED import PGsOED
from sOED.utils import *


def linear_model(stage, theta, d, xp=None):
    """
    Linear model function G(theta, d) = theta * d

    Parameters
    ----------
    stage : int
        The stage index of the experiment.
    theta : np.ndarray of size (n_sample, n_param)
        The value of unknown linear model parameters.
    d : np.ndarray of size (n_sample, n_design)
        The design variable.
    xp : np.ndarray of size (n_sample, n_phys_state), optional(default=None)
        The physical state.

    Returns
    -------
    numpy.ndarray of size (n_sample, n_obs)
        The output of the linear model.
    """
    global count
    count += max(len(theta), len(d))
    return theta * d


def reward_fun(stage, xb, xp, d, y):
    """
    Non-KL-divergence based reward function g_k(x_k, d_k, y_k)

    Parameters
    ----------
    stage : int
        The stage index of the experiment.
    xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1)
        Grid discritization of the belief state.
    xp : np.ndarray of size (n_phys_state)
        The physical state.
    d : np.ndarray of size (n_design)
        The design variable.
    y : np.ndarray of size (n_obs)
        The observation.

    Returns
    -------
    A float which is the reward.
    """
    return 0


n_stage = 3  # Number of stages.
n_param = 1  # Number of parameters.
n_design = 1  # Number of design variables.
n_obs = 1  # Number of observations.
step_cost = 0  # Step cost.

prior_type = "normal" # "normal" for normal dist, "uniform" for uniform dist.
prior_loc = 0 # mean for normal, lower bound for uniform.
prior_scale = 3 # std for normal, range for uniform.
prior_info = [(prior_type, prior_loc, prior_scale),]

design_bounds = [(0.1, 3.0),] # lower and upper bounds of design variables.

# Noise if following N(noise_loc, noise_base_scale + noise_ratio_scale * abs(G))
noise_loc = 0
noise_base_scale = 1
noise_ratio_scale = 0
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale),]

# Number of grid points on each dimension of parameter space to store PDFs.
n_grid = 50

# Method to sample posterior samples, could be "Rejection" or "MCMC", default
# is "MCMC".
post_rvs_method = "Rejection"


# Random state could be either an integer or None.
random_state = 2021  # the random seed
# random_state = random.randint(1, 100000)

soed = PGsOED(model_fun=linear_model,
              n_stage=n_stage,
              n_param=n_param,
              n_design=n_design,
              n_obs=n_obs,
              step_cost=step_cost,
              prior_info=prior_info,
              design_bounds=design_bounds,
              noise_info=noise_info,
              reward_fun=reward_fun,
              n_grid=n_grid,
              post_rvs_method=post_rvs_method,
              random_state=random_state,
              actor_dimns=[80, 80],
              critic_dimns=[80, 80],
              double_precision=True,
              stopping_curriculum=False)
              # stopping_prob_N_final=30)

actor_optimizer = optim.SGD(soed.actor_net.parameters(), lr=0.15)
n_critic_update = 500  # iterative update
critic_optimizer = optim.SGD(soed.critic_net.parameters(), lr=0.001)


count = 0
soed.soed(n_update=200,
          n_traj=1000,
          actor_optimizer=actor_optimizer,
          n_critic_update=n_critic_update,
          critic_optimizer=critic_optimizer,
          design_noise_scale=1.0,
          design_noise_decay=0.99)


def save_state(soed, filename):
    state = {
        'n_stage': soed.n_stage,
        'step_cost': soed.step_cost,
        'stopping_curriculum':soed.stopping_curriculum,
        'ys_hist': soed.ys_hist,
        'ds_hist': soed.ds_hist,
        'xps_hist': soed.xps_hist,
        'ycs_hist': soed.ycs_hist,
        'dcs_hist': soed.dcs_hist,
        'xpcs_hist': soed.xpcs_hist,
        'rewards_hist': soed.rewards_hist,
        'stages_hist': soed.stages_hist,
        'stopping_probs_hist': soed.stopping_probs_hist,
        'immediate_rewards_hist': soed.immediate_rewards_hist,
        'stop_rewards_hist': soed.stop_rewards_hist,
        'part_continue_rewards_hist': soed.part_continue_rewards_hist,
        'part_stop_rewards_hist': soed.part_stop_rewards_hist,
        'averaged_continue_reward_hist': soed.averaged_continue_reward_hist,
        'averaged_stop_reward_hist': soed.averaged_stop_reward_hist,
        'averaged_stopping_stage_hist': soed.averaged_stopping_stage_hist,
        'averaged_total_reward_hist': soed.averaged_total_reward_hist,
        'actor_net': soed.actor_net,
        'critic_net': soed.critic_net,
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

    torch.save(soed.actor_net.state_dict(), filename + '_actor_net.pth')
    torch.save(soed.critic_net.state_dict(), filename + '_critic_net.pth')


filename = 'results/optimal_stopping/soed-lg-T={}-Cost={}-curriculum.pkl'.format(soed.n_stage, soed.step_cost)
filename = 'results/optimal_stopping/soed-lg-T={}-Cost={}-vanilla.pkl'.format(soed.n_stage, soed.step_cost)
filename = 'results/fixed_stage/soed-lg-T={}-fix={}-Cost={}.pkl'.format(soed.n_stage, 3, soed.step_cost)
filename = 'results/threshold_stopping/soed-lg-T={}-Cost={}.pkl'.format(soed.n_stage, soed.step_cost)
save_state(soed, filename)
with open(filename, 'rb') as f:
    # state_vanilla = pickle.load(f)
    # state_curriculum = pickle.load(f)
    state_threshold = pickle.load(f)
    # state_fixed2 = pickle.load(f)
    # state_fixed3 = pickle.load(f)


def plot_reward_history(soed):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 28})
    title_size = 30
    label_size = 28
    lw = 3.5
    n_update = 200

    episodes = np.arange(n_update)
    plt.plot(range(n_update), state_vanilla['averaged_total_reward_hist'], label='Vanilla PG', linewidth=lw, color='C0')
    # plt.plot(episodes, state_curriculum['averaged_total_reward_hist'], label='Curriculum PG', linewidth=lw, color='C2')
    # plt.fill_between(episodes, soed.averaged_total_reward_hist - std_rewards, soed.averaged_total_reward_hist + std_rewards,
    #                  alpha=0.3, label='±1 std')
    plt.plot(range(n_update), state_threshold['averaged_total_reward_hist'], label='Threshold ($R_{th} = 2.6$)', linewidth=lw, color='C1')
    # plt.plot(range(n_update), state_fixed2['averaged_total_reward_hist'], label='Stop at $k=2$', linewidth=lw, color='C1')
    # plt.plot(range(n_update), state_fixed3['averaged_total_reward_hist'], label='Stop at $k=3$', linewidth=lw, color='C4')
    optimal_utilities = [2.203, 2.547, 2.749, 2.892, 3.003, 3.094]
    optimal_utility = 0
    for k in range(soed.n_stage):
        utility = optimal_utilities[k] + soed.step_cost * (k + 1)
        if utility > optimal_utility:
            optimal_utility = utility
    xmin = (0 - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0])
    xmax = n_update / plt.xlim()[1]
    plt.axhline(xmin=xmin, xmax=xmax, y=optimal_utility, color='red', linestyle='--', label='Analytical result', linewidth=lw)
    plt.title('N = {}, Cost = {}'.format(soed.n_stage, soed.step_cost), fontsize=title_size)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    # plt.xlabel('Gradient ascent step', fontsize=label_size)
    plt.ylabel('Reward', fontsize=label_size)
    plt.legend(fontsize=label_size-2, labelspacing=0.18)
    plt.tight_layout()
    plt.savefig('results/LinearGaussian-T={}-Cost={}-Reward-both.pdf'.format(soed.n_stage, soed.step_cost), dpi=300)
    plt.show()


def plot_stopping_history(soed):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 28})
    title_size = 30
    label_size = 28
    lw = 3.5

    n_update = 200
    episodes = np.arange(n_update)

    plt.plot(episodes, state_vanilla['averaged_stopping_stage_hist'], label='Vanilla optimal stopping', linewidth=lw, color='C0')
    plt.plot(episodes, state_curriculum['averaged_stopping_stage_hist'], label='Curriculum optimal stopping', linewidth=lw, color='C2')
    # plt.fill_between(episodes, soed.averaged_stopping_stage_hist - std_stages, soed.averaged_stopping_stage_hist + std_stages,
    #                  alpha=0.3, label='±1 std')

    # plt.plot(range(n_update), state_threshold['averaged_stopping_stage_hist'], label='Threshold stopping ($R_{th} = 2.6$)', linewidth=lw, color='C1')
    optimal_utilities = [2.203, 2.547, 2.749, 2.892, 3.003, 3.094]
    optimal_utility = 0
    for k in range(soed.n_stage):
        utility = optimal_utilities[k] + soed.step_cost * (k + 1)
        if utility > optimal_utility:
            optimal_utility = utility
            optimal_stopping = k + 1
    xmin = (0 - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0])
    xmax = n_update / plt.xlim()[1]
    plt.axhline(xmin=xmin, xmax=xmax, y=optimal_stopping, color='red', linestyle='--',
                label='Analytical result', linewidth=lw)
    plt.xlabel('Gradient ascent step', fontsize=label_size)
    # plt.ylabel('Stage', fontsize=label_size)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))
    # plt.title('N = {}, Cost = {}'.format(4, -0.25), fontsize=title_size)
    # plt.legend(fontsize=label_size, loc='upper right')
    plt.tight_layout()
    plt.savefig('results/LinearGaussian-T={}-Cost={}-tau-both.pdf'.format(soed.n_stage, soed.step_cost), dpi=300)
    plt.show()

