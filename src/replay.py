
import numpy
import pickle

from collections import deque


class ReplayBuffer:
    """Random experience replay buffer.

    Attributes
    ----------
    obs_dim : Tuple
        Dimensions of state of agent.
    size : int
        Number of experiences to hold.
    save_dir : str
        Path to save the experience replay.
    batch_size: int
        Number of samples to batchify.
    n_step : int
        `n` parameter of  multi-step learning algorithm.
    gamma : float
        Gamma parameter of multi-step learning algorithm.
    """

    def __init__(self, obs_dim, size, save_dir, batch_size, n_step, gamma):
        self.obs_buf = numpy.zeros([size, *obs_dim], dtype=numpy.float32)
        self.next_obs_buf = numpy.zeros([size, *obs_dim], dtype=numpy.float32)
        self.acts_buf = numpy.zeros([size], dtype=numpy.float32)
        self.rews_buf = numpy.zeros([size], dtype=numpy.float32)
        self.done_buf = numpy.zeros(size, dtype=numpy.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        self.save_dir = save_dir
        self.r_chkpt_cnt = 1

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs, act, rew, next_obs, done):
        """Stores experience.
        
        Attributes
        ----------
        obs : numpy.ndarray
            State of agent at a time step `t`.
        act : int
            Selected action by the agent at a time step `t`.
        rew : float
            Accumulated reward by the agent after an action 
            given its state at a time step `t`.
        next_obs : numpy.ndarray
            State of the agent at a time step `t + 1`.
        done : bool
            Terminal flag after the selected action at a time step `t`.
        """

        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self):
        """Sample a batch of experiences.
        
        Returns
        -------
        Dict
            A dictionary with a number of past experiences equal to the batch number
            fetched by the random replay algorithm.
        """

        idxs = numpy.random.choice(
            self.size, size=self.batch_size, replace=False)

        return dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs], rews=self.rews_buf[idxs], done=self.done_buf[idxs],
                    # for N-step Learning
                    indices=idxs,
                    )

    def sample_batch_from_idxs(self, idxs):
        """Builds a batch using an index list.

        Parameters
        ----------
        idxs : List[int]
            The index list used to fetch experiences 
            from the random replay buffer.

        Returns
        -------
        Dict
            An experience batch.
        """

        # for N-step Learning
        return dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs], rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs], )

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Returns n-step information on reward,
        next state, and terminal indicator.
        
        Parameters
        ----------
        n_step_buffer : deque
            An n-step memory deque data structure.    
        gamma : float
            Gamma parameter of multi-step learning algorithm.

        Returns
        -------
        Dict
            An experience sample of reward, next state, and terminal indicator.
        """

        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def save(self, attribute="random"):
        """Random experience replay checkpoint builder.

        Parameters
        ----------
        attribute : str, optional
            Filename used to define the
            checkpoint, by default "random"
        """
        save_path = self.save_dir / f"{attribute}-replay.npz"
        numpy.savez_compressed(save_path, obs=self.obs_buf,
                               next_obs=self.next_obs_buf, acts=self.acts_buf,
                               rews=self.rews_buf, done=self.done_buf)

        misc = dict(ptr=self.ptr, max_size=self.max_size)
        repl_misc_save_path = self.save_dir / f"{attribute}-replay-misc.pkl"
        with open(repl_misc_save_path, "wb") as fp:
            pickle.dump(misc, fp)

        print(f"[{' ' + attribute[0].upper() + '. replay #' + str(self.r_chkpt_cnt) + ']':>15} "
              f"Random Memory Buffer of size {self.ptr} and total "
              f"capacity {self.max_size} saved to {save_path}")

        self.r_chkpt_cnt += 1

    def load(self, np_chkpt_path, misc_repl_path):
        """Random experience replay checkpoint loader.

        Parameters
        ----------
        np_chkpt_path : str
            Path to retrieve the random experience replay checkpoint.
        misc_repl_path : str
            Path to retrieve the settings used in
            the random experience replay checkpoint.
            
        Raises
        ------
        ValueError
            If path to random experience replay
            checkpoint does not exist.
        ValueError
            If path to settings used in the random
            experience replay checkpoint does not exist.
        """
        if not np_chkpt_path.exists():
            raise ValueError(f"{np_chkpt_path} does not exist")

        if not misc_repl_path.exists():
            raise ValueError(f"{misc_repl_path} does not exist")

        chkpt = numpy.load(np_chkpt_path)

        self.obs_buf = chkpt['obs']
        self.acts_buf = chkpt['acts']
        self.rews_buf = chkpt['rews']
        self.done_buf = chkpt['done']
        self.next_obs_buf = chkpt['next_obs']

        with open(misc_repl_path, "rb") as fh:
            misc = pickle.load(fh)

            self.ptr = misc["ptr"]
            self.max_size = misc["max_size"] if misc["max_size"] > self.max_size else self.max_size

        print(f"Loading replay of size {self.ptr} and total "
              f"capacity {self.max_size} from {np_chkpt_path}")

    def __len__(self):
        return self.size
