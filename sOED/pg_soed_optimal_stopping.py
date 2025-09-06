import torch.nn as nn
import torch.optim as optim
from .soed import SOED
from .utils import *


class PGsOED(SOED):
    def __init__(self, model_fun,
                 n_stage, n_param, n_design, n_obs, step_cost,
                 prior_info, design_bounds, noise_info,
                 reward_fun=None, phys_state_info=None,
                 n_grid=50, post_rvs_method="MCMC", random_state=None,
                 actor_dimns=None, critic_dimns=None,
                 double_precision=False, prob_categorical_output=False,
                 n_categories=None, include_outcome_rewards=0.0,
                 stopping_curriculum=True, stopping_curriculum_decay=0.95,
                 stopping_prob_lambda=None,
                 stopping_prob_target=0.999, stopping_prob_N_final=10,
                 cost_function=None, train_mode=True, mc_samples=50):
        super().__init__(model_fun, n_stage, n_param, n_design, n_obs, step_cost,
                         prior_info, design_bounds, noise_info,
                         reward_fun, phys_state_info,
                         n_grid, post_rvs_method, random_state)

        self.prob_categorical_output = prob_categorical_output
        self.include_outcome_rewards = include_outcome_rewards
        self.train_mode = train_mode
        self.mc_samples = mc_samples
        
        # Curriculum learning parameters for optimal stopping
        self.stopping_curriculum = stopping_curriculum  # Whether to use curriculum learning for stopping
        self.stopping_curriculum_decay = stopping_curriculum_decay  # Decay rate for stopping probability (not used in exp schedule)
        self.stopping_prob_target = stopping_prob_target  # Desired min prob at last N episodes
        self.stopping_prob_N_final = stopping_prob_N_final  # Number of final episodes to reach target
        self.stopping_prob_lambda = stopping_prob_lambda  # Sigmoid schedule parameter (computed if None)
        self.current_stopping_prob = 0.0  # Current stopping probability (starts at 0)
        
        # Cost function configuration
        self.cost_function = cost_function  # Function to compute design-dependent costs
        if self.cost_function is None:
            # Default to constant cost function
            self.cost_function = lambda designs: self.step_cost
        
        if prob_categorical_output:
            assert n_categories is not None, "n_categories must be specified when prob_categorical_output is True"
            self.n_categories = n_categories  # Store number of categories separately
        else:
            self.n_categories = None
        if random_state is None:
            random_state = np.random.randint(1e6)
        torch.manual_seed(random_state)

        assert isinstance(double_precision, bool), (
            "double_precision should either be True or False.")
        if double_precision:
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32

        # Initialize the actor (policy) network and critic network.
        self.actor_input_dimn = (self.n_stage +
                                 (self.n_stage - 1) * (self.n_obs +
                                                       self.n_design))
        self.critic_input_dimn = self.actor_input_dimn + self.n_design
        self.actor_dimns = actor_dimns
        self.critic_dimns = critic_dimns
        self.initialize()

        self.initialize_policy = self.initialize_actor
        self.load_policy = self.load_actor
        self.get_policy = self.get_actor

    def compute_cumulative_cost(self, stage, designs_hist):
        """
        Compute cumulative cost up to a given stage.
        
        Parameters
        ----------
        stage : int
            Current stage (0-indexed)
        designs_hist : numpy.ndarray
            History of designs up to current stage, shape (n_traj, stage, n_design)
            
        Returns
        -------
        numpy.ndarray
            Cumulative costs for each trajectory, shape (n_traj,)
        """
        if stage == 0:
            return np.zeros(designs_hist.shape[0])
        
        cumulative_costs = np.zeros(designs_hist.shape[0])
        for k in range(stage):
            costs = self.cost_function(designs_hist[k, :])
            cumulative_costs += costs
            
        return cumulative_costs

    def initialize(self):
        self.initialize_actor(self.actor_dimns)
        self.initialize_critic(self.critic_dimns)
        self.design_noise_scale = None

    def initialize_actor(self, actor_dimns=None):
        """
        Initialize the actor (policy) network.

        Parameters
        ----------
        actor_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of actor (policy) network.
        """
        NoneType = type(None)
        assert isinstance(actor_dimns, (list, tuple, np.ndarray, NoneType)), (
               "actor_dimns should be a list, tuple or numpy.ndarray of "
               "integers.")
        output_dimn = self.n_design
        if actor_dimns is None:
            actor_dimns = (self.actor_input_dimn * 10,
                           self.actor_input_dimn * 10)
        self.actor_dimns = np.copy(actor_dimns)
        actor_dimns = np.append(np.append(self.actor_input_dimn,
                                          actor_dimns),
                                output_dimn)
        if self.prob_categorical_output:
            self.actor_net = Net_prob_categorical(actor_dimns,
                                           nn.ReLU()).to(self.dtype)
        else:
            self.actor_net = Net(actor_dimns,
                                nn.ReLU(),
                                self.design_bounds).to(self.dtype)
        self.update = 0
        self.actor_optimizer = None
        self.actor_lr_scheduler = None

    def initialize_critic(self, critic_dimns=None):
        """
        Initialize the critic (actor-value function).

        Parameters
        ----------
        critic_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of critic (actor value function)
            network.
        """
        NoneType = type(None)
        assert isinstance(critic_dimns, (list, tuple, np.ndarray, NoneType)), (
               "critic_dimns should be a list, tuple or numpy.ndarray of ",
               "integers.")
        output_dimn = 1
        if critic_dimns is None:
            critic_dimns = (self.critic_input_dimn * 10,
                            self.critic_input_dimn * 10)
        self.critic_dimns = np.copy(critic_dimns)
        critic_dimns = np.append(np.append(self.critic_input_dimn,
                                           critic_dimns),
                                 output_dimn)
        self.critic_net = Net(critic_dimns,
                              nn.ReLU(),
                              np.array([[-np.inf, np.inf]])).to(self.dtype)
        self.update = 0
        self.critic_optimizer = None
        self.critic_lr_scheduler = None

    def load_actor(self, net, optimizer=None):
        """
        Load the actor network with single precision.

        Parameters
        ----------
        net : nn.Module
            A pre-trained PyTorch network with input dimension actor_input_dimn
            and output dimension n_design.
        optimizer : algorithm of torch.optim
            An optimizer corresponding to net.
        """
        try:
            net = net.to(self.dtype)
            output = net(torch.zeros(1, self.actor_input_dimn).to(self.dtype))
            assert output.shape[1] == self.n_design, (
                   "Output dimension should be {}.".format(self.n_design))
            self.actor_net = net
        except:
            print("Actor network should has "
                  "input dimension {}.".format(self.actor_input_dimn))
        self.actor_optimizer = optimizer
        self.update = 0

    def load_critic(self, net, optimizer=None):
        """
        Load the critic network with single precision.

        Parameters
        ----------
        net : nn.Module
            A pre-trained PyTorch network with input dimension critic_input_dimn
            and output dimension 1.
        optimizer : algorithm of torch.optim
            An optimizer corresponding to net.
        """
        try:
            net = net.to(self.dtype)
            output = net(torch.zeros(1, self.critic_input_dimn).to(self.dtype))
            assert output.shape[1] == 1, (
                   "Output dimension should be 1.")
            self.critic_net = net
        except:
            print("Critic network should has "
                  "input dimension {}.".format(self.critic_input_dimn))
        self.critic_optimizer = optimizer
        self.update = 0

    def get_actor(self):
        return self.actor_net

    def get_critic(self):
        return self.critic_net

    def form_actor_input(self, stage, ds_hist, ys_hist):
        """
        A function to form the inputs of actor network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage".

        Returns
        -------
        A torch.Tensor of size (n_traj, dimn_actor_input).
        """
        n_traj = max(len(ds_hist), len(ys_hist))
        X = np.zeros((n_traj, self.actor_input_dimn))  # Inputs: n traj of current state
        X[:, stage] = 1  # Index of experiments.
        # Historical designs.
        begin = self.n_stage
        end = begin + np.prod(ds_hist.shape[1:])
        X[:, begin:end] = ds_hist.reshape(len(ds_hist), end - begin)
        # Historical observations.
        begin = self.n_stage + (self.n_stage - 1) * self.n_design
        end = begin + np.prod(ys_hist.shape[1:])
        X[:, begin:end] = ys_hist.reshape(len(ys_hist), end - begin)
        X = torch.from_numpy(X).to(self.dtype)
        return X

    def form_critic_input(self, stage, ds_hist, ys_hist, ds):
        """
        A function to form the inputs of critic network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage".
        ds : numpy.ndarray of size(n_traj or 1, n_design)
            Designs on which we want to get the Q value.

        Returns
        -------
        A torch.Tensor of size (n_traj, critic_input_dimn).
        """
        n_traj = max(len(ds_hist), len(ys_hist))
        X = torch.zeros(n_traj, self.critic_input_dimn).to(self.dtype)
        X[:, :self.actor_input_dimn] = self.form_actor_input(stage,
                                                             ds_hist,
                                                             ys_hist)
        X[:, -self.n_design:] = torch.from_numpy(ds).to(self.dtype)
        return X

    def get_designs(self, stage=0, ds_hist=None, ys_hist=None):
        """
        A function to get designs by running the policy network.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design),
                  optional(default=None)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs),
                  optional(default=None)
            n_traj sequences of observations before stage "stage".

        Returns
        -------
        A numpy.ndarry of size (n_traj, n_design) which are designs.
        """
        if ds_hist is None:
            ds_hist = np.empty((1, 0, self.n_design))
        if ys_hist is None:
            ys_hist = np.empty((1, 0, self.n_obs))
        # assert ds_hist.shape[1] == stage and ys_hist.shape[1] == stage
        X = self.form_actor_input(stage, ds_hist, ys_hist)
        designs = self.actor_net(X).detach().double().numpy()
        return designs

    def get_design(self, stage=0, d_hist=None, y_hist=None):
        """
        A function to get a single design by running the policy network.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage - 1.
        d_hist : numpy.ndarray of size (stage, n_design),
                 optional(default=None)
            A sequence of designs before stage "stage".
        y_hist : numpy.ndarray of size (stage, n_obs),
                 optional(default=None)
            A sequence of observations before stage "stage".

        Returns
        -------
        A numpy.ndarry of size (n_design) which is the design.
        """
        if d_hist is None:
            d_hist = np.empty((0, self.n_design))
        if y_hist is None:
            y_hist = np.empty((0, self.n_obs))
        return self.get_designs(stage,
                                d_hist.reshape(1, -1, self.n_design),
                                y_hist.reshape(1, -1, self.n_obs)).reshape(-1)

    def get_action_value(self, stage, ds_hist, ys_hist, ds):
        """
        A function to get the Q-value by running the critic network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage".
        ds : numpy.ndarray of size(n_traj or 1, n_design)
            Designs on which we want to get the Q value.

        Returns
        -------
        A numpy.ndarry of size (n_traj) which are Q values.
        """
        # assert ds_hist.shape[1] == stage and ys_hist.shape[1] == stage
        X = self.form_critic_input(stage, ds_hist, ys_hist, ds)
        values = self.critic_net(X).detach().double().numpy()
        return values.reshape(-1)

    def soed(self, n_update=100, n_traj=1000,
             actor_optimizer=None,
             actor_lr_scheduler=None,
             n_critic_update=30,
             critic_optimizer=None,
             critic_lr_scheduler=None,
             design_noise_scale=None, design_noise_decay=0.99,
             on_policy=True, thetas=None):

        if actor_optimizer is None:
            if self.actor_optimizer is None:
                self.actor_optimizer = optim.SGD(self.actor_net.parameters(),
                                                 lr=0.1)
        else:
            self.actor_optimizer = actor_optimizer
        if actor_lr_scheduler is None:
            if self.actor_lr_scheduler is None:
                self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.actor_optimizer, gamma=0.98)
        else:
            self.actor_lr_scheduler = actor_lr_scheduler
        if critic_optimizer is None:
            if self.critic_optimizer is None:
                self.critic_optimizer = optim.Adam(self.critic_net.parameters(),
                                                   lr=0.01)
        else:
            self.critic_optimizer = critic_optimizer
        if critic_lr_scheduler is None:
            if self.critic_lr_scheduler is None:
                self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.critic_optimizer, gamma=0.98)
        else:
            self.critic_lr_scheduler = critic_lr_scheduler
        if design_noise_scale is None:
            if self.design_noise_scale is None:
                self.design_noise_scale = (self.design_bounds[:, 1] -
                                           self.design_bounds[:, 0]) / 20
                self.design_noise_scale[self.design_noise_scale == np.inf] = 5
        elif isinstance(design_noise_scale, (list, tuple)):
            self.design_noise_scale = np.array(design_noise_scale)
        else:
            self.design_noise_scale = design_noise_scale
        assert design_noise_decay > 0 and design_noise_decay <= 1

        self.averaged_total_reward_hist = []  # record the averaged total reward of all trajectories
        self.averaged_stopping_stage_hist = []  # record the averaged stopping stage of all trajectories
        self.part_continue_rewards_hist = [[[] for j in range(self.n_stage)] for i in range(n_update)]  # record the partial continue rewards of all trajectories
        self.part_stop_rewards_hist = [[[] for j in range(self.n_stage)] for i in range(n_update)]  # record the partial stop rewards of all trajectories
        self.averaged_continue_reward_hist = np.zeros((n_update, self.n_stage))  # record the averaged partial continue rewards of all trajectories
        self.averaged_stop_reward_hist = np.zeros((n_update, self.n_stage))  # record the averaged partial stop rewards of all trajectories
        self.stages_hist = np.ones((n_update, n_traj)) # record the stopping stage of all trajectories
        self.rewards_hist = np.zeros((n_update, n_traj))  # record the total rewards of all trajectories
        self.stopping_probs_hist = np.zeros(n_update)  # record the stopping probability of all updates

        for l in range(n_update):
            print('Update Level', self.update)
            
            # Update stopping probability for curriculum learning
            if self.stopping_curriculum and self.train_mode:
                # Compute lambda if not set, to ensure target prob at last N episodes
                if self.stopping_prob_lambda is None:
                    stopping_prob_center = (n_update - self.stopping_prob_N_final) / (2 * n_update)
                    progress_target = (n_update - self.stopping_prob_N_final) / n_update - stopping_prob_center
                    self.stopping_prob_lambda = np.log(self.stopping_prob_target / (1 - self.stopping_prob_target)) / progress_target
                progress = l / n_update
                # Sigmoid schedule: starts slow, accelerates around center, levels off
                self.current_stopping_prob = 1.0 / (1.0 + np.exp(-self.stopping_prob_lambda * (progress - stopping_prob_center)))
                self.stopping_probs_hist[l] = self.current_stopping_prob
            else:
                self.current_stopping_prob = 1.0
                self.stopping_probs_hist[l] = self.current_stopping_prob
            
            # Pass thetas to asses method if provided
            if thetas is not None:
                self.asses(n_traj, self.design_noise_scale, thetas=thetas)
            else:
                self.asses(n_traj, self.design_noise_scale)  # generate n traj based on the current policy for training and testing

            total_reward = 0.0
            stopping_stage_sum = 0

            # Form the inputs and target values of critic network, and form the inputs of the actor network.
            X_critic = torch.zeros(self.n_stage * n_traj, self.critic_input_dimn).to(self.dtype)
            X_actor = torch.zeros(self.n_stage * n_traj, self.actor_input_dimn).to(self.dtype)
            g_critic = torch.zeros(self.n_stage * n_traj, 1).to(self.dtype)  # target value of critic network

            # From the whole trajs, form the input at different stages
            begin = 0
            indexes = np.arange(n_traj)  # current traj index
            continue_rewards = np.zeros(n_traj)

            for k in range(self.n_stage):
                self.indexes = indexes
                if k == 0:  # force to continue
                    end = begin + len(indexes)
                    X = self.form_critic_input(k,
                                               self.ds_hist[:, :k],
                                               self.ys_hist[:, :k],
                                               self.ds_hist[:, k])
                    X_critic[begin:end, :] = X
                    X = self.form_actor_input(k,
                                              self.ds_hist[:, :k],
                                              self.ys_hist[:, :k])
                    X_actor[begin:end] = X

                    # compute the continue rewards with MC samples
                    for j in indexes:
                        # estimate Q(k+1, x_k+1, d_k+1) with MC samples
                        ys = self.ys_mc_hist[:, k, j].reshape(-1, 1) # possible outcomes for current experiments
                        ds = self.get_designs(stage=k+1, ds_hist=self.ds_hist[j, :k+1], ys_hist=ys) # design for next experiment dependent on outcomes
                        continue_rewards[j] = np.mean(self.get_action_value(k+1, self.ds_hist[j, :k+1], ys, ds) + self.immediate_rewards_hist[j, k])

                    if len(indexes) > 0:  # still having trajs to be processed
                        self.averaged_continue_reward_hist[l, k] = np.mean(continue_rewards[indexes])
                        self.averaged_stop_reward_hist[l, k] = 0.0
                        self.part_continue_rewards_hist[l][k] = list(continue_rewards[indexes])
                        self.part_stop_rewards_hist[l][k] = list(np.zeros(len(indexes)))

                    next_values = self.get_action_value(k + 1, self.ds_hist[:, :k + 1], self.ys_hist[:, :k + 1], self.ds_hist[:, k + 1])
                    g_critic[begin:end, 0] = torch.from_numpy(self.immediate_rewards_hist[:, k] + next_values)
                else:
                    if len(indexes) > 0:
                        end = begin + len(indexes)
                        stop_rewards = self.stop_rewards_hist[indexes, k]
                        self.part_stop_rewards_hist[l][k] = list(stop_rewards)

                        # Curriculum learning: apply stopping decision based on current probability
                        if self.stopping_curriculum:
                            continue_ids = np.ones(len(stop_rewards), dtype=bool)  # Default to continue
                            sampled = np.random.random(len(stop_rewards)) < self.current_stopping_prob
                            # Only compute continue rewards for sampled
                            for i, (j, allow_stop) in enumerate(zip(indexes, sampled)):
                                if allow_stop:
                                    ys_mc = self.ys_mc_hist[:, k, j].reshape(-1, 1)
                                    ys_cat = np.concatenate([np.tile(self.ys_hist[j, :k].reshape(1, -1), (self.mc_samples, 1)), ys_mc], axis=1)
                                    if k == self.n_stage - 1:
                                        rs = np.zeros(self.mc_samples)
                                        for ii in range(self.mc_samples):
                                            xb = self.get_xb(d_hist=self.ds_hist[j, :], y_hist=ys_cat[ii, :])
                                            cumulative_cost = self.compute_cumulative_cost(k + 1, self.ds_hist[j, :k+1, :])[0]
                                            rs[ii] = self.get_reward(k, xb, None, None, None) + cumulative_cost
                                        cont_val = np.mean(rs)
                                    else:
                                        ds = self.get_designs(stage=k + 1, ds_hist=self.ds_hist[j, :k + 1].reshape(1, -1), ys_hist=ys_cat)
                                        cont_val = np.mean(
                                            self.get_action_value(k + 1, self.ds_hist[j, :k + 1].reshape(1, -1), ys_cat, ds) + self.immediate_rewards_hist[j, k])
                                    continue_rewards[j] = cont_val
                                    # Now apply stopping decision
                                    eps_i = 0.0
                                    if cont_val + eps_i > stop_rewards[i]:
                                        continue_ids[i] = True
                                    else:
                                        continue_ids[i] = False
                        # vanilla: optimal stopping for all
                        else:
                            for j in indexes:
                                ys_mc = self.ys_mc_hist[:, k, j].reshape(-1, 1)
                                ys_cat = np.concatenate(
                                    [np.tile(self.ys_hist[j, :k].reshape(1, -1), (self.mc_samples, 1)), ys_mc], axis=1)  # possible next states
                                if k == self.n_stage - 1:
                                    rs = np.zeros(self.mc_samples)
                                    for i in range(self.mc_samples):
                                        xb = self.get_xb(d_hist=self.ds_hist[j, :], y_hist=ys_cat[i, :])
                                        rs[i] = self.get_reward(k, xb, None, None, None) + k * self.step_cost
                                    continue_rewards[j] = np.mean(rs)
                                else:
                                    ds = self.get_designs(stage=k + 1,
                                                          ds_hist=self.ds_hist[j, :k + 1].reshape(1, -1),
                                                          ys_hist=ys_cat)
                                    continue_rewards[j] = np.mean(
                                        self.get_action_value(k + 1, self.ds_hist[j, :k + 1].reshape(1, -1), ys_cat,
                                                              ds) + self.immediate_rewards_hist[j, k])

                                self.part_continue_rewards_hist[l][k] = list(continue_rewards[indexes])
                                if len(indexes) > 0:
                                    self.averaged_continue_reward_hist[l, k] = np.mean(continue_rewards[indexes])
                                self.averaged_stop_reward_hist[l, k] = np.mean(stop_rewards)

                            delta = continue_rewards[indexes] - stop_rewards
                            gaps = 0
                            eps = np.zeros(len(stop_rewards)) + gaps
                            continue_ids = np.greater(delta + eps, 0)  # those choose to take actions at current stage

                        continue_indexes = indexes[continue_ids]
                        just_stopped_indexes = indexes[~continue_ids]
                        self.rewards_hist[l, just_stopped_indexes] = stop_rewards[~continue_ids]

                        X = self.form_critic_input(k,
                                                   self.ds_hist[indexes, :k],
                                                   self.ys_hist[indexes, :k],
                                                   self.ds_hist[indexes, k])
                        X_critic[begin:end] = X  # the input of critic network
                        X = self.form_actor_input(k,
                                                   self.ds_hist[indexes, :k],
                                                   self.ys_hist[indexes, :k])
                        X_actor[begin:end] = X  # the input of critic network

                        # update it to the max between stop reward and the value of the next state
                        if k == self.n_stage - 1:  # the next state is forced to stop
                            next_values = self.stop_rewards_hist[indexes, k + 1]
                        else:
                            next_values = self.get_action_value(k+1, self.ds_hist[indexes, :k+1], self.ys_hist[indexes, :k+1], self.ds_hist[indexes, k+1])
                        # next_values = continue_rewards[indexes]
                        g_critic[begin:end, 0] = (torch.from_numpy(next_values).to(self.dtype) + self.immediate_rewards_hist[indexes, k]) * continue_ids + stop_rewards * (~continue_ids)

                        total_reward += np.sum(stop_rewards * ~continue_ids)
                        stopping_stage_sum += k * np.sum(~continue_ids)
                        indexes = continue_indexes
                        self.stages_hist[l, indexes] += 1
                        
                    else:
                        break
                begin = end

            X_critic = X_critic[:end]
            g_critic = g_critic[:end]
            X_actor = X_actor[:end]

            stop_rewards = self.stop_rewards_hist[indexes, self.n_stage]
            total_reward += np.sum(stop_rewards)
            stopping_stage_sum += self.n_stage * np.sum(continue_ids)
            
            # Store rewards for trajectories that completed all stages
            self.rewards_hist[l, indexes] = stop_rewards

            # Train critic.
            if self.train_mode:
                for _ in range(n_critic_update):
                    y_critic = self.critic_net(X_critic)
                    critic_loss = torch.mean((g_critic - y_critic) ** 2)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
                    self.critic_optimizer.step()
                self.critic_lr_scheduler.step()

                # One step update on the actor network.
                # Add negative sign here because we want to do maximization.
                # manual computing with the policy gradient theorem
                X_critic_back = X_critic
                X_critic_back.requires_grad = True
                X_critic_back.grad = None
                output = self.critic_net(X_critic_back).sum()
                output.backward()
                critic_grad = X_critic_back.grad[:, -self.n_design:]  # dQ/da
                output = -(self.actor_net(X_actor) * critic_grad).sum(-1).mean()
                self.actor_optimizer.zero_grad()
                output.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                self.actor_lr_scheduler.step()

            averaged_reward = total_reward / n_traj
            averaged_stopping = stopping_stage_sum / n_traj
            self.averaged_total_reward_hist.append(averaged_reward)
            self.averaged_stopping_stage_hist.append(averaged_stopping)
            
            # Print curriculum learning information
            if self.stopping_curriculum:
                print("Averaged total reward:  {:.4},".format(averaged_reward))
                print("Averaged stopping stage:  {:.4},".format(averaged_stopping))
                print(" Stopping probability: {:.3f}".format(self.current_stopping_prob))
            else:
                print("Averaged total reward:  {:.4},".format(averaged_reward))
                print("Averaged stopping stage:  {:.4},".format(averaged_stopping))

            self.update += 1
            self.design_noise_scale *= design_noise_decay

    def reset_curriculum(self):
        """
        Reset the curriculum learning state.
        """
        self.current_stopping_prob = 0.0
        print("Curriculum learning reset: stopping probability = 0.0")
    
    def get_curriculum_info(self):
        """
        Get information about the current curriculum learning state.
        
        Returns
        -------
        dict
            Dictionary containing curriculum learning parameters and current state.
        """
        return {
            'stopping_curriculum': self.stopping_curriculum,
            'stopping_curriculum_decay': self.stopping_curriculum_decay,
            'stopping_prob_lambda': self.stopping_prob_lambda,
            'stopping_prob_target': self.stopping_prob_target,
            'stopping_prob_N_final': self.stopping_prob_N_final,
            'current_stopping_prob': self.current_stopping_prob,
            'current_update': self.update
        }

    def asses(self, n_traj=10000, design_noise_scale=None,
              return_all=False, store_belief_state=True, thetas=None):
        """
        A function to asses the performance of current policy.

        Parameters
        ----------
        n_traj : int, optional(default=10000)
            Number of trajectories to sample during the assesment.
        design_noise_scale : int, list, tuple or numpy.ndarray of size
                             (n_design), optional(default=None)
            The scale of additive exploration Gaussian noise on each dimension
            of design variable. When it is None, design_noise_scale will be
            set to 0.
        return_all : bool, optional(default=False)
            Return all information or not.
            If False, only return the averaged totoal reward.
            If True, return a tuple of all information generated during the
            assesment, including
            * averaged_reward (averaged total reward), float
            * thetas (parameters), numpy.ndarray of size (n_traj, n_param).
            * dcs_hist (clean designs), numpy.ndarray of size (n_traj,
                                                               n_stage,
                                                               n_design)
            * ds_hist (noisy designs), numpy.ndarray of size (n_traj,
                                                              n_stage,
                                                              n_design).
            * ys_hist (observations), numpy.ndarray of size (n_traj,
                                                             n_stage,
                                                             n_obs) .
            * xbs (terminal belief states), could either be None or
                numpy.ndarray of size (n_traj,
                                       n_grid ** n_param,
                                       n_param + 1),
                controlled by store_belief_state.
            * xps_hist (physical states), numpy.ndarray of size (n_traj,
                                                                 n_stage + 1,
                                                                 n_phys_state).
            * rewards_hist (rewards), numpy.ndarray of size (n_traj,
                                                             n_stage + 1).
        store_belief_state : bool, optional(default=False)
            Whether store the belief states.
        thetas : numpy.ndarray of size (n_traj, n_param), optional(default=None)
            Specific theta values to use for assessment. If None, samples from prior.

        Returns
        -------
        A float which is the averaged total reward.
        (optionally) other assesment results.
        """
        # Generate prior samples
        if design_noise_scale is None:
            design_noise_scale = np.zeros(self.n_design)
        elif isinstance(design_noise_scale, (int, float)):
            design_noise_scale = np.ones(self.n_design) * design_noise_scale
        elif isinstance(design_noise_scale, (list, tuple, np.ndarray)):
            assert (isinstance(design_noise_scale, (list, tuple, np.ndarray))
                    and len(design_noise_scale) == self.n_design)

        # Generate or use provided theta values
        if thetas is None:
            thetas = self.prior_rvs(n_traj)  # generate theta from prior
        else:
            # Validate provided thetas
            assert thetas.shape == (n_traj, self.n_param), (
                f"thetas should have shape ({n_traj}, {self.n_param}), got {thetas.shape}")
            # Ensure we have the right number of trajectories
            if thetas.shape[0] != n_traj:
                # print(f"Warning: thetas has {thetas.shape[0]} samples but n_traj={n_traj}. "
                #       f"Using first {n_traj} samples.")
                thetas = thetas[:n_traj]
        dcs_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        ds_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        ys_hist = np.zeros((n_traj, self.n_stage, self.n_obs))
        ycs_hist = np.zeros((n_traj, self.n_stage, self.n_obs))
        ys_mc_hist = np.zeros((self.mc_samples, self.n_stage, n_traj))
        if store_belief_state:
            # We only store the terminal belief state.
            xbs = np.zeros((n_traj, self.n_stage + 1, *self.init_xb.shape))
            # print(xbs.shape, self.init_xb.shape)
        else:
            xbs = None
        # Store n_stage + 1 physical states.
        xps_hist = np.zeros((n_traj, self.n_stage + 1, self.n_xp))
        xps_hist[:, 0] = self.init_xp
        xpcs_hist = np.zeros((n_traj, self.n_stage + 1, self.n_xp))
        xpcs_hist[:, 0] = self.init_xp
        immediate_rewards_hist = np.zeros((n_traj, self.n_stage + 1))
        stop_rewards_hist = np.zeros((n_traj, self.n_stage + 1))
        immediate_rewards_clean_hist = np.zeros((n_traj, self.n_stage + 1))
        stop_rewards_clean_hist = np.zeros((n_traj, self.n_stage + 1))
        progress_points = np.rint(np.linspace(0, n_traj - 1, 30))

        for k in range(self.n_stage + 1):
            if k < self.n_stage:
                # Get clean designs.
                dcs = self.get_designs(stage=k,
                                       ds_hist=ds_hist[:, :k],
                                       ys_hist=ys_hist[:, :k])
                dcs_hist[:, k, :] = dcs
                # Add design noise for exploration.
                ds = np.random.normal(loc=dcs, scale=design_noise_scale)
                ds = np.maximum(ds, self.design_bounds[:, 0])
                ds = np.minimum(ds, self.design_bounds[:, 1])
                ds_hist[:, k, :] = ds
                # Run the forward model to get observations.
                Gs = self.m_f(k,
                              thetas,
                              ds,
                              xps_hist[:, k, :])
                Gcs = self.m_f(k,
                              thetas,
                              dcs,
                              xps_hist[:, k, :])
                ys = np.random.normal(Gs + self.noise_loc,
                                      self.noise_b_s
                                      + self.noise_r_s * np.abs(Gs))
                ys_mc = np.random.normal((Gs + self.noise_loc).reshape(1, -1),
                                         (self.noise_b_s
                                      + self.noise_r_s * np.abs(Gs)).reshape(1, -1), (self.mc_samples, n_traj))
                ycs = np.random.normal(Gcs + self.noise_loc,
                                      self.noise_b_s
                                      + self.noise_r_s * np.abs(Gcs))
                ys_hist[:, k, :] = ys
                ycs_hist[:, k, :] = ycs
                ys_mc_hist[:, k, :] = ys_mc
                # Get rewards.
                for i in range(n_traj):
                    immediate_rewards_hist[i, k] = self.get_reward(k,
                                                                   None,
                                                                   None,
                                                                   None,
                                                                   None)
                    
                    # compute the stop reward
                    if k == 0:
                        stop_rewards_hist[i, k] = 0.0
                        stop_rewards_clean_hist[i, k] = 0.0
                    else:
                        xb = self.get_xb(d_hist=ds_hist[i, :k], y_hist=ys_hist[i, :k])
                        cumulative_cost = self.compute_cumulative_cost(k, ds_hist[i, :k, :])[0]
                        stop_rewards_hist[i, k] = self.get_reward(k,
                                                                  xb,
                                                                  None,
                                                                  None,
                                                                  None) + cumulative_cost
                        xb_clean = self.get_xb(d_hist=dcs_hist[i, :k], y_hist=ycs_hist[i, :k])
                        if store_belief_state:
                            xbs[i, k] = xb_clean
                        
                # Update physical state.
                xps = self.xp_f(xps_hist[:, k], k, ds, ys)
                xps_hist[:, k + 1] = xps
            else:
                for i in range(n_traj):
                    # Get terminal belief state.
                    xb = self.get_xb(d_hist=ds_hist[i, :], y_hist=ys_hist[i, :])
                    # Get reward.
                    cumulative_cost = self.compute_cumulative_cost(k, ds_hist[i, :, :])[0]
                    immediate_rewards_hist[i, k] = self.get_reward(k,
                                                                   xb,
                                                                   None,
                                                                   None,
                                                                   None) + cumulative_cost
                    stop_rewards_hist[i, k] = immediate_rewards_hist[i, k]

                    xb_clean = self.get_xb(d_hist=dcs_hist[i, :], y_hist=ycs_hist[i, :])
                    if store_belief_state:
                        xbs[i, k] = xb_clean
                    stop_rewards_clean_hist[i, k] = immediate_rewards_clean_hist[i, k]

                    print('*' * (progress_points == i).sum(), end='')
                print('\n')
        self.dcs_hist = dcs_hist
        self.ds_hist = ds_hist
        self.ys_hist = ys_hist
        self.ycs_hist = ycs_hist
        self.ys_mc_hist = ys_mc_hist
        self.xbs = xbs
        self.xps_hist = xps_hist
        self.xpcs_hist = xps_hist
        self.immediate_rewards_hist = immediate_rewards_hist
        self.immediate_rewards_clean_hist = immediate_rewards_clean_hist
        self.stop_rewards_hist = stop_rewards_hist
        self.stop_rewards_clean_hist = stop_rewards_clean_hist
        if return_all:
            return (thetas,
                    dcs_hist, ds_hist, ys_hist,
                    xbs, xps_hist,
                    immediate_rewards_hist)
        else:
            return immediate_rewards_hist

