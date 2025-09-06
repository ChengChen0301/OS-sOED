import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.insert(0, parent_dir)


import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LinearLocator

from sOED import SOED
from sOED import PGsOED
from sOED.utils import *
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


conv_diff_nets = (torch.load("conv_diff_net_t0.05.pt"),
                  torch.load("conv_diff_net_t0.1.pt"),
                  torch.load("conv_diff_net_t0.15.pt"),
                  torch.load("conv_diff_net_t0.2.pt"))


def conv_diff_model(stage, theta, d, xp=None):
    """
    Convection diffusion model

    Parameters
    ----------
    stage : int
        The stage index of the experiment.
    theta : np.ndarray of size (n_sample or 1, n_param)
        The value of unknown linear model parameters.
    d : np.ndarray of size (n_sample or 1, n_design)
        The design variable.
    xp : np.ndarray of size (n_sample or 1, n_phys_state),
         optional(default=None)
        The physical state.

    Returns
    -------
    numpy.ndarray of size (n_sample, n_obs)
        The output of the convection diffusion model.
    """
    n_sample = max(len(theta), len(d), len(xp))
    X = torch.zeros(n_sample, 4).double()
    X[:, :2] = torch.from_numpy(theta)
    X[:, 2:] = torch.from_numpy(xp + d)
    return conv_diff_nets[stage](X).detach().numpy()


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
    if stage < n_stage:
        return 0
    else:
        return 0


def phys_state_fun(xp, stage, d, y):
    """
    Physical state transition function.
    x_{k+1,p} = phys_state_fun(x_{k,p}, d_k, y_k).

    Parameters
    ----------
    xp : np.ndarray of size (n_sample or 1, n_phys_state)
        The old physical state before conducting stage-th
        experiement.
    stage : int
        The stage index of the experiment.
    d : np.ndarray of size (n_sample or 1, n_design)
        The design variables at stage-th experiment.
    y : np.ndarray of size (n_sample or 1, n_obs)
        The observations at stage-th expriments.

    Returns
    -------
    A numpy.ndarray of size (n_sample, n_xp)
    """
    return xp + d



n_stage = 4 # Number of stages.
n_param = 2 # Number of parameters.
n_design = 2 # Number of design variables.
n_obs = 1 # Number of observations.
step_cost = -0.8

prior_type = "uniform"  # "normal" for normal dist, "uniform" for uniform dist.
prior_loc = 0 # mean for normal, lower bound for uniform.
prior_scale = 1 # std for normal, range for uniform.
prior_info = [(prior_type, prior_loc, prior_scale),
              (prior_type, prior_loc, prior_scale)]

design_bounds = [(-0.25, 0.25), (-0.25, 0.25)] # lower and upper bounds of design variables.

# Noise if following N(noise_loc, noise_base_scale + noise_ratio_scale * abs(G))
noise_loc = 0
noise_base_scale = 0.05
noise_ratio_scale = 0
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale),]

# Physical state info
n_phys_state = 2
init_phys_state = (0.5, 0.5)
phys_state_info = (n_phys_state, init_phys_state, phys_state_fun)

# Number of grid points on each dimension of parameter space to store PDFs.
n_grid = 50

# Method to sample posterior samples, could be "Rejection" or "MCMC", default
# is "MCMC".
post_rvs_method = "Rejection"


# Random state could be eith an integer or None.
random_state = 2021

def quadratic_cost(designs):
    return - 2 * np.sum(designs**2)


soed = PGsOED(model_fun=conv_diff_model,
              n_stage=n_stage,
              n_param=n_param,
              n_design=n_design,
              n_obs=n_obs,
              step_cost=step_cost,
              prior_info=prior_info,
              design_bounds=design_bounds,
              noise_info=noise_info,
              reward_fun=reward_fun,
              phys_state_info=phys_state_info,
              n_grid=n_grid,
              post_rvs_method=post_rvs_method,
              random_state=random_state,
              actor_dimns=[80, 80],
              critic_dimns=[80, 80],
              double_precision=True,
              stopping_curriculum=True,  # False for Vanilla
              mc_samples=50,
              train_mode=False,  # False for testing
              # cost_function=quadratic_cost,
              stopping_prob_N_final=30)

soed.initialize()
actor_optimizer = optim.Adam(soed.actor_net.parameters(), lr=0.01)
actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.98)

n_critic_update = 100
critic_optimizer = optim.Adam(soed.critic_net.parameters(), lr=0.01)
critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.98)


soed.soed(n_update=300,
          n_traj=1000,
          actor_optimizer=actor_optimizer,
          actor_lr_scheduler=actor_lr_scheduler,
          n_critic_update=n_critic_update,
          critic_optimizer=critic_optimizer,
          critic_lr_scheduler=critic_lr_scheduler,
          design_noise_scale=0.05,
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


filename = 'results/optimal_stopping/soed-cd-T={}-Cost={}-curriculum1.pkl'.format(soed.n_stage, soed.step_cost)
filename = 'results/optimal_stopping/soed-cd-T={}-Cost=2quadratic-curriculum1.pkl'.format(soed.n_stage)
filename = 'results/optimal_stopping/soed-cd-T={}-Cost={}-vanilla.pkl'.format(soed.n_stage, soed.step_cost)
filename = 'results/optimal_stopping/soed-cd-T={}-Cost=2quadratic-vanilla.pkl'.format(soed.n_stage)
filename = 'results/fixed_stage/soed-cd-T={}-fix={}-Cost={}.pkl'.format(soed.n_stage, 3, soed.step_cost)
save_state(soed, filename)
with open(filename, 'rb') as f:
    # state_vanilla = pickle.load(f)
    # state_curriculum = pickle.load(f)
    state_fixed3 = pickle.load(f)
    # state_fixed2 = pickle.load(f)

actor_net = torch.load(filename + '_actor_net.pth')
soed.actor_net.load_state_dict(actor_net)
critic_net = torch.load(filename + '_critic_net.pth')
soed.critic_net.load_state_dict(critic_net)
thetas, dcs_hist, ds_hist, ys_hist, xbs, xps_hist, immediate_rewards_hist = soed.asses(n_traj=1000, return_all=True, store_belief_state=True)

soed.soed(n_update=1,
          n_traj=1000,
          actor_optimizer=actor_optimizer,
          actor_lr_scheduler=actor_lr_scheduler,
          n_critic_update=n_critic_update,
          critic_optimizer=critic_optimizer,
          critic_lr_scheduler=critic_lr_scheduler,
          design_noise_scale=0,
          thetas=thetas)

id = 0
fig = create_probability_plots(soed_curriculum, id)
# plt.show()
plt.savefig('CD-posterior-T={}-Cost={}-id={}-curriculum.pdf'.format(soed.n_stage, soed.step_cost, id), dpi=300)



def create_probability_plots(soed, id):
    """
    Create probability density contour plots

    Parameters:
    -----------
    stars : array-like, shape (1, 2)
        Coordinates of stars for each subplot (x, y)
    nodes : array-like, shape (4, 2)
        Coordinates of nodes for each subplot (x, y)
    posteriors : list of array-like, shape (4, grid_size, grid_size)
        Probability density values for each subplot
    z_values : array-like
        Grid points for x and y axes
    """

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 4, figure=fig)
    label_size = 18
    plt.rcParams.update({'font.size': 18})

    # Titles for each subplot
    titles = [f'$p(\\theta_x, \\theta_y|I_{i + 1})$' for i in range(4)]

    n_plots = int(soed.stages_hist[0, id])
    source_loc = thetas[id]
    grids_x = soed.xbs[id, 1, :, 0].reshape(50, 50)
    grids_y = soed.xbs[id, 1, :, 1].reshape(50, 50)
    designs = soed.dcs_hist[id]
    posteriors = soed.xbs[id, 1:, :, 2].reshape(4, 50, 50)

    former_loc = np.array([0.5, 0.5])
    # Create subplots
    for i in range(n_plots):
        ax = fig.add_subplot(gs[i])

        # Create contour plot
        contour = ax.contourf(grids_x, grids_y, posteriors[i],
                              levels=15, cmap='viridis')

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=label_size)

        # Plot star
        ax.plot(source_loc[0], source_loc[1], '*', color='magenta',
                markersize=15, label='Star')

        # Plot node
        ax.plot(former_loc[0] + designs[i, 0], former_loc[1] + designs[i, 1], 'o', color='red',
                markersize=10, label='Node')

        # Draw line connecting star and node
        ax.plot([former_loc[0], former_loc[0] + designs[i, 0]],
                [former_loc[1], former_loc[1] + designs[i, 1]],
                '-', color='red', alpha=0.7)
        former_loc = former_loc + designs[i]

        # Set title and labels
        ax.set_title(titles[i], fontsize=label_size)
        ax.set_xlabel('$Z_x$', fontsize=label_size)
        if i == 0:
            ax.set_ylabel('$Z_y$', fontsize=label_size)
        else:  # Remove y-ticks for other subplots
            ax.set_yticklabels([])

        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_reward_history(soed):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 28})
    title_size = 30
    label_size = 28
    # n_update = len(soed.averaged_total_reward_hist)
    n_update = 300
    lw = 3.5

    plt.plot(range(n_update), state_vanilla['averaged_total_reward_hist'], label='Vanilla PG', linewidth=lw, color='C0')
    plt.plot(range(n_update), state_curriculum['averaged_total_reward_hist'], label='Curriculum PG', linewidth=lw,color='C2')
    plt.plot(range(n_update), state_fixed2['averaged_total_reward_hist'], label='Stop at $k=2$', linewidth=lw, color='C1')
    plt.plot(range(n_update), state_fixed3['averaged_total_reward_hist'], label='Stop at $k=3$', linewidth=lw, color='C4')
    plt.title('N = {}, Cost = {}'.format(soed.n_stage, soed.step_cost), fontsize=title_size)
    # plt.title(r'N = {}, Cost = - $\|\|\xi_k\|\|^2$'.format(4), fontsize=title_size)
    # plt.xlabel('Gradient ascent step', fontsize=label_size)
    # plt.ylabel('Reward', fontsize=label_size-2)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))
    plt.legend(fontsize=label_size, labelspacing=0.18)
    plt.tight_layout()
    plt.savefig('results2/ConvectionDiffusion-T={}-Cost={}-Reward-both.pdf'.format(soed.n_stage, soed.step_cost), dpi=300)
    # plt.savefig('results/ConvectionDiffusion-T={}-Cost=1quadratic-Reward-both.pdf'.format(soed.n_stage), dpi=300)
    plt.show()


def plot_stopping_history(soed):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({'font.size': 28})
    title_size = 30
    label_size = 28
    lw = 3.5

    n_update = 300
    ax.plot(range(n_update), state_vanilla['averaged_stopping_stage_hist'], label='Vanilla PG', linewidth=lw, color='C0')
    ax.plot(range(n_update), state_curriculum['averaged_stopping_stage_hist'], label='Curriculum PG', linewidth=lw, color='C2')
    ax.set_xlabel('Gradient ascent step', fontsize=label_size)
    # ax.set_ylabel('Stage', fontsize=label_size)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # ax.set_ylim(0.9, 4.1)
    # inset_ax = inset_axes(ax,
    #                       width="70%", height="70%",
    #                       # bbox_to_anchor=(0.25, -0.1, 0.7, 0.7),
    #                       # bbox_to_anchor=(0.25, 0.0, 0.7, 0.7),
    #                       bbox_to_anchor=(0.35, 0.0, 0.6, 0.5),
    #                       bbox_transform=ax.transAxes)
    # inset_ax.plot(range(n_update), state_curriculum['averaged_stopping_stage_hist'], linewidth=lw-1, color='C2')
    # inset_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    # ax.set_title('N = {}, Cost = {}'.format(4, -0.8), fontsize=title_size)
    # ax.set_title(r'N = {}, Cost = - $\|\|\xi_k\|\|^2$'.format(4), fontsize=title_size)
    # ax.legend(fontsize=label_size)
    fig.tight_layout()
    plt.savefig('results2/ConvectionDiffusion-T={}-Cost={}-tau-both.pdf'.format(soed.n_stage, soed.step_cost), dpi=300)
    # plt.savefig('results/ConvectionDiffusion-T={}-Cost=1quadratic-tau-both.pdf'.format(soed.n_stage), dpi=300)
    plt.show()


