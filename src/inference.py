
import torch
import numpy as np
import gymnasium as gym

from pathlib import Path

from src.model import RainbowDQN


class Inference:
    """Rainbow Agent [1]_.

    ...

    Attributes
    ----------
    env : gym.Env
        The agent's environment.
    v_min : float
        Minimum value of the support.
    v_max : float
        Maximum value of the support.
    n_atoms : int
        Number of atoms in the support.
    model_checkpoint : str
        Path to retrieve a previous model checkpoint.
    verbose : bool
        If `True`, prints some more information.
    num_hiddens : int
        Number of hidden units in the model.
    device : str
        Device to use.
    activation : str
        Activation function to use.
    enable_base_model : bool
        If `True`, enables the base model.

    .. [1] Matteo Hessel, Joseph Modayil, Hadovan Hasselt,
        Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan,
        Bilal Piot, Mohammad Azar and David Silver. Rainbow:
        Combining improvements in deep reinforcement learning, 2017.
    """
    def __init__(
        self, 
        env: gym.Env,
        v_min: float = -21.0,
        v_max: float = 20.0,
        n_atoms: int = 51,
        model_checkpoint: str = None,
        verbose: bool = False,
        num_hiddens: int = 512,
        device: str = 'cpu',
        activation: str = 'gelu',
        enable_base_model: bool = False,
    ):
        # Device: cpu / gpu
        self.device = self.get_device(device=device)
        print(f"Agent using: {self.device}")

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
        
        # Checkpoint loaders
        self.load(model_checkpoint / "model.pth")
        
        # Set to eval mode
        self.online.eval()

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
        rst_sampler : bool, optional
            Whether to reset the sampler, by default False
        
        Returns
        -------
        int
            An integer representing which action the drone will perform.
        """
        
        state = np.array(state)
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        action_values = self.online(state)
        action_idx = torch.argmax(action_values, axis=1).item()

        return action_idx

    def load(self, agent_chkpt_path):
        """Agent Q model checkpoint loader.

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

        self.online.load_state_dict(online_state_dict, strict=False)
        print(f"Loaded model at {agent_chkpt_path}")
