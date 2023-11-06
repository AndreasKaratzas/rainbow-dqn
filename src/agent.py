
import os
import torch
import shutil
import numpy as np
import gymnasium as gym
import torch.optim as optim

from pathlib import Path
from torch.nn.utils import clip_grad_norm_

from src.model import RainbowDQN
from src.replay import ReplayBuffer
from src.priority import PrioritizedReplayBuffer


class Agent:
    """Rainbow Agent [1]_.

    ...

    Attributes
    ----------
    env : gym.Env
        The agent's environment.
    batch_size : int
        Number of samples to batchify.
    target_sync : int
        Number of steps to sync target network.
    gamma : float
        Discount factor.
    num_of_steps_to_checkpoint : int
        Number of steps to save the model.
    beta : float
        Beta parameter for prioritized replay buffer.
    prior_eps : float
        Epsilon parameter for prioritized replay buffer.
    mem_capacity : int
        Number of experiences to hold.
    alpha : float
        Alpha parameter for prioritized replay buffer.
    n_step : int
        `n` parameter of  multi-step learning algorithm.
    v_min : float
        Minimum value of the support.
    v_max : float
        Maximum value of the support.
    n_atoms : int
        Number of atoms in the support.
    learning_rate : float
        Learning rate for the optimizer.
    episodes : int
        Number of episodes to train the agent.
    model_save_dir : str
        Path to save the model's checkpoints.
    memory_save_dir : str 
        Path to save the agent's memory.
    model_checkpoint : str
        Path to retrieve a previous model checkpoint.
    mem_checkpoint : str
        Path to retrieve a previous agent's memory checkpoint.
    clip_grad_norm : float
        Maximum norm of the gradients.
    topk : int
        Number of top model checkpoints to keep.
    verbose : bool
        If `True`, prints some more information.
    demo : bool
        If `True`, runs the agent in test mode.
    learning_starts : int
        Number of steps before starting to train the agent.
    num_hiddens : int
        Number of hidden units in the model.
    device : str
        Device to use.
    enable_base_model : bool
        If `True`, enables the base model.
    activation : str
        Activation function to use.

    .. [1] Matteo Hessel, Joseph Modayil, Hadovan Hasselt,
        Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan,
        Bilal Piot, Mohammad Azar and David Silver. Rainbow:
        Combining improvements in deep reinforcement learning, 2017.
    """
    def __init__(
        self, 
        env: gym.Env,
        batch_size: int = 128,
        target_sync: int = int(1e3),
        gamma: float = 0.9,
        num_of_steps_to_checkpoint_model: int = int(1e4),
        num_of_steps_to_checkpoint_memory: int = int(1e6),
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        mem_capacity: int = int(2e4),
        alpha: float = 0.4,
        n_step: int = 3,
        v_min: float = 0,
        v_max: float = 200,
        n_atoms: int = 51,
        learning_rate: float = 0.00025,
        episodes: int = int(1e3),
        model_save_dir: str = 'model',
        memory_save_dir: str = 'memory',
        model_checkpoint: str = None,
        mem_checkpoint: str = None,
        clip_grad_norm: float = 5.0,
        topk: int = 5,
        verbose: bool = False,
        demo: bool = False,
        learning_starts: int = int(1e3),
        num_hiddens: int = 128,
        device: str = 'cpu',
        activation: str = 'relu',
        enable_base_model: bool = False,
    ):
        self.batch_size = batch_size
        self.target_update = target_sync
        self.gamma = gamma
        self.episodes = episodes
        self.clip_grad_norm = clip_grad_norm
        self.topk = topk
        self.learning_starts = learning_starts
        self.bak_dir = os.path.join(model_save_dir, 'bak')
        self.last_model_chkpt = 0
        self.last_memory_chkpt = 0
        self.demo = demo
        self.is_training = False
        
        # Create directories
        if not os.path.exists(self.bak_dir):
            os.makedirs(self.bak_dir)

        # Agent checkpoint helper variables
        self.counter = 0
        self.curr_step = 0
        self.curr_episode = 0
        self.save_dir = model_save_dir
        self.memory_interval = num_of_steps_to_checkpoint_memory
        self.model_interval = num_of_steps_to_checkpoint_model
        self.curr_best_mean_reward = -np.inf
        self.ep_rewards = np.zeros(
            (self.model_interval, 1),
            dtype=np.float)

        # Device: cpu / gpu
        self.device = self.get_device(device=device)
        print(f"Agent using: {self.device}")

        # PER
        # Memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.priority_replay = PrioritizedReplayBuffer(
            obs_dim=env.observation_space.shape, size=mem_capacity, 
            save_dir=memory_save_dir, batch_size=batch_size,
            alpha=alpha)

        # Memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.random_replay = ReplayBuffer(
                obs_dim=env.observation_space.shape, size=mem_capacity, 
                save_dir=memory_save_dir, batch_size=batch_size,
                n_step=n_step, gamma=gamma)

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = n_atoms
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms)\
            .to(self.device)

        # Networks: online, target
        self.online = RainbowDQN(in_dim=env.observation_space.shape, out_dim=env.action_space.n, 
                                 atom_size=self.atoms, support=self.support, activation=activation, 
                                 num_hiddens=num_hiddens, enable_base_model=enable_base_model, verbose=verbose).to(self.device)
        self.target = RainbowDQN(in_dim=env.observation_space.shape, out_dim=env.action_space.n, 
                                 atom_size=self.atoms, support=self.support, activation=activation, 
                                 num_hiddens=num_hiddens, enable_base_model=enable_base_model, verbose=verbose).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.online.train()
        self.target.eval()

        # Print networks
        if verbose:
            print(self.online)

        # Optimizer
        self.optimizer = optim.Adam(self.online.parameters(), lr=learning_rate)

        # Checkpoint loaders
        if model_checkpoint:
            self.load(model_checkpoint / "model.pth")

        if mem_checkpoint and not self.demo:
            self.priority_replay.load(
                mem_checkpoint / "priority-replay.npz",
                mem_checkpoint / "priority-replay-misc.pkl",
                mem_checkpoint / "sum-tree.pkl",
                mem_checkpoint / "min-tree.pkl",
                mem_checkpoint / "miscellaneous.pkl")
            self.random_replay.load(
                mem_checkpoint / "random-replay.npz",
                mem_checkpoint / "random-replay-misc.pkl")
        
        if self.demo:
            self.online.eval()
            self.target.eval()

    def get_device(self, device: str = 'cpu'):
        """Get device for torch.

        Parameters
        ----------
        device : str, optional
            Device to use, by default 'cpu'

        Returns
        -------
        torch.device
            Device to use.
        """
        gpu_candidates = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if device in gpu_candidates + ['gpu', 'cuda']:
            if device in gpu_candidates:
                if torch.cuda.is_available():
                    return torch.device(device)
            else:
                if torch.cuda.is_available():
                    return torch.device('cuda:0')
        return torch.device('cpu')

    def act(self, state):
        """Given a state, choose an action and update value of step.

        Parameters
        ----------
        state : numpy.ndarray
            A single observation of the current state.

        Returns
        -------
        int
            An integer representing which action the drone will perform.
        """

        # NoisyNet: no epsilon greedy action selection
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        action_values = self.online(state)
        action_idx = torch.argmax(action_values, axis=1).item()

        if self.learning_starts > self.curr_step and not self.demo:
            action_idx = np.random.randint(0, self.env.action_space.n)

        # Increment step
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Stores the experience replay and priority buffers.

        Parameters
        ----------
        state : numpy.ndarray
            The state of the agent at a time step `t`.
        next_state : numpy.ndarray
            The state of the agent at the next time step `t + 1`.
        action : int
            The action selected by the agent at a time step `t`.
        reward : float
            The reward accumulated by the agent at a time step `t`.
        done : bool
            The terminal indicator at a time step `t`.
        """
        Transition = [state, action, reward, next_state, done]

        # N-step transition
        if self.use_n_step:
            one_step_transition = self.random_replay.store(*Transition)
        # 1-step transition
        else:
            one_step_transition = Transition

        # Add a single step transition
        if one_step_transition:
            self.priority_replay.store(*one_step_transition)

        # Update mean reward array
        self.ep_rewards[(self.curr_step - 1) %
                        int(self.model_interval)] = reward

    def recall_priority(self):
        """Retrieve a batch of experiences from the priority experience replay.

        Returns
        -------
        Tuple
            A batch of experiences fetched by the priority experience replay.
        """
        # PER needs beta to calculate weights
        samples = self.priority_replay.sample_batch(self.beta)

        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)

        indices = samples["indices"]
        return state, action, reward, next_state, done, weights, indices

    def recall_random(self, indices):
        """Retrieve a batch of experiences from the random experience replay.

        Returns
        -------
        Tuple
            A batch of experiences fetched by the random experience replay.
        """
        samples = self.random_replay.sample_batch_from_idxs(indices)

        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        return state, action, reward, next_state, done

    def update_model(self, loss):
        """Executes backpropagation algorithm on the online model.

        Parameters
        ----------
        loss : float
            The loss of the model which is to be backpropagated.
        """
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients by L_2 norm
        clip_grad_norm_(self.online.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    @torch.no_grad()
    def projection_distribution(self, next_state, reward, done, gamma):
        """Use the Categorical DQN algorithm to compute the distribution.

        Parameters
        ----------
        next_state : numpy.ndarray
            The state corresponding to time step `t + 1`.
        reward : float
            The reward corresponding to time step `t`.
        done : bool
            The terminal flag corresponding to time step `t`.
        gamma : float
            The gamma constant described in the Rainbow DQN algorithm.

        Returns
        -------
        torch.Tensor
            The discrete probability distribution
            of the Bellman operator T applied to z.
        """
        # Computes support
        delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)

        """Calculate n^{th} next state probabilities.
        """
        # Probabilities p(s_{t + n}, · ; theta_{online})
        # an then compute the distribution
        # d_{t + n} = (z, p(s_{t + n}, · ; theta_{online}))
        # Finally, perform argmax action selection
        # using online network:
        #   argmax_{a}[(z, p(s_{t + n}, a; theta_{online}))]
        next_action = self.online(next_state).argmax(1)
        # Probabilities p(s_{t + n}, · ; theta_{target})
        next_dist = self.target.dist(next_state)
        # Double-DQN probabilities
        #   p(s_{t + n}, argmax_{a}[(z, p(s_{t + n}, a; theta_{online}))]; theta_{target})
        next_dist = next_dist[range(self.batch_size), next_action]

        """Compute T_{z} (Bellman operator T applied to z).
        """
        # T_{z} = R^n + (gamma^n)z (accounting for terminal states)
        t_z = reward + (1 - done) * gamma * self.support
        # Clamp unsupported values
        t_z = t_z.clamp(min=self.v_min, max=self.v_max)

        """Compute L_2 projection of Tz onto fixed support z.
        """
        # b = (T_{z} - V_{min}) / Delta_{z}
        b = (t_z - self.v_min) / delta_z
        # Type cast variables
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        """Distribute probability of T_{z}.
        """
        # Compute offset
        offset = (torch.linspace(0, (self.batch_size - 1) * self.atoms,
                                 self.batch_size).long().unsqueeze(1).expand(
            self.batch_size, self.atoms).to(self.device))

        # Initialize distribution
        proj_dist = torch.zeros(next_dist.size(), device=self.device)
        # proj_dist_{l} = proj_dist_{l} + p(s_{t + n}, a*)(u - b)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (
            next_dist * (u.float() - b)).view(-1))
        # proj_dist_{u} = proj_dist_{u} + p(s_{t + n}, a*)(b - l)
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (
            next_dist * (b - l.float())).view(-1))

        return proj_dist

    def compute_td_loss(self, state, action, proj_dist):
        """Computes Cross-entropy loss. This minimizes 
        the Kullback-Leibler divergence (m | p(s_{t}, a_{t})).

        Parameters
        ----------
        state : numpy.ndarray
            The state of the agent at a past time step `t`.
        action : int
            The action selected by the agent at a past time step `t`.
        proj_dist : torch.Tensor
            The discrete probability distribution
            of the Bellman operator T applied to z.

        Returns
        -------
        float
            The cross-entropy loss to be backpropagated.
        """
        dist = self.online.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        # Backpropagate importance-weighted minibatch loss
        td_loss = -(proj_dist * log_p).sum(1)
        return td_loss

    def learn(self, episode):
        """Main Rainbow agent training algorithm.

        Parameters
        ----------
        episode : int
            Current episode number.

        Returns
        -------
        float
            Loss of model at that time step.
        """
        if self.curr_step % self.target_update == 0:
            self.sync_Q_target()

        # Re-calculate number of agent training episodes
        if self.learning_starts < self.curr_step and not self.is_training:
            self.episodes -= episode
            self.is_training = True

        # If learning has started
        if self.learning_starts < self.curr_step:
            # If number of steps since last update is more than the interval
            if self.last_model_chkpt > self.model_interval:
                # If current model is better than last checkpoint
                if np.mean(self.ep_rewards) > self.curr_best_mean_reward:
                    # Update best mean reward metric
                    self.curr_best_mean_reward = np.mean(self.ep_rewards)
                    # Update model checkpoint
                    self.save()
                    # Reset checkpoint counter
                    self.last_model_chkpt = 0
                else:
                    if self.curr_episode != episode:
                        # Save the model as the most recent checkpoint  
                        self.save(recent=True)
                        # Update current episode
                        self.curr_episode = episode
            else:
                # Update checkpoint counter
                self.last_model_chkpt += 1
        
        if self.last_memory_chkpt > self.memory_interval:
            # Update memory checkpoint data
            self.priority_replay.save()
            self.random_replay.save()
            # Reset checkpoint counter
            self.last_memory_chkpt = 0
        else:
            # Update checkpoint counter
            self.last_memory_chkpt += 1

        if len(self.priority_replay) < self.batch_size:
            return None
        
        if self.learning_starts > self.curr_step:
            return None

        # Sample from memory
        state, action, reward, next_state, done, weights, indices = \
            self.recall_priority()

        # Get categorical dqn loss
        proj_dist = self.projection_distribution(
            next_state, reward, done, self.gamma)

        # Get cross entropy
        td_loss = self.compute_td_loss(state, action, proj_dist)

        # PER: importance sampling before average
        loss = torch.mean(td_loss * weights)

        # N-step Learning loss
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            state, action, reward, next_state, done = self.recall_random(
                indices)
            n_step_proj_dist = self.projection_distribution(
                next_state, reward, done, gamma)
            n_step_td_loss = self.compute_td_loss(
                state, action, n_step_proj_dist)
            td_loss += n_step_td_loss

            # PER: importance sampling before average
            loss = torch.mean(td_loss * weights)

        self.update_model(loss)

        # PER: update priorities
        loss_for_prior = td_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.priority_replay.update_priorities(indices, new_priorities)

        # PER: increase beta
        fraction = min(episode / int(self.episodes), 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        # NoisyNet: reset noise
        self.online.reset_noise()
        self.target.reset_noise()

        return loss.item()

    def sync_Q_target(self):
        """Hard target model sync with online model.
        """
        self.target.load_state_dict(self.online.state_dict())

    def backup_curr_topk(self):
        """Backup current top-k model checkpoints.
        """
        if self.counter > self.topk:
            # Delete last model checkpoint
            os.remove(Path(self.bak_dir) / "model_1.pth")

            # Rename all model checkpoints
            for i in range(1, self.topk):
                os.rename(Path(self.bak_dir) / f"model_{i + 1}.pth",
                        Path(self.bak_dir) / f"model_{i}.pth")
            
            # Copy current model checkpoint to top-k directory
            shutil.copy(Path(self.save_dir) / "model.pth", Path(self.bak_dir) / f"model_{self.topk}.pth")
        else:
            # Copy current model checkpoint to top-k directory
            shutil.copy(Path(self.save_dir) / "model.pth",
                        Path(self.bak_dir) / f"model_{self.counter}.pth")
        
    def save(self, recent: bool = False):
        """Agent Q model checkpoint builder.
        """
        save_path = Path(self.save_dir) / f"model.pth"
        if recent:
            parent = Path(self.save_dir).parent / "recent"
            if not parent.exists():
                parent.mkdir()
            save_path = parent / "model.pth"

        torch.save(
            dict(
                online=self.online.state_dict(),
                target=self.target.state_dict(),
                optim=self.optimizer.state_dict(),
                mean_rew=self.curr_best_mean_reward
            ),
            save_path)
        print(f"[{'model #' + str(self.counter) + ']':>15} "
              f"Agent saved to {save_path}")

        if not recent:
            self.counter += 1
            self.backup_curr_topk()

    def load(self, agent_chkpt_path):
        """Agent Q model and optimizer checkpoint loader.

        Parameters
        ----------
        agent_chkpt_path : str
            Path to retrieve a previous model checkpoint.

        Raises
        ------
        ValueError
            If path to previous model checkpoint does not exist.
        """
        if not Path(agent_chkpt_path).exists():
            raise ValueError(f"{agent_chkpt_path} does not exist")

        ckp = torch.load(agent_chkpt_path, map_location=self.device)

        online_state_dict = ckp.get('online')
        target_state_dict = ckp.get('target')
        optim_state = ckp.get('optim')
        self.curr_best_mean_reward = ckp.get('mean_rew')

        print(f"Loading model at {agent_chkpt_path}")

        self.online.load_state_dict(online_state_dict, strict=False)
        self.target.load_state_dict(target_state_dict, strict=False)
        self.optimizer.load_state_dict(optim_state)

        print(f"Loaded pretrained weights with mean reward {self.curr_best_mean_reward}.")
