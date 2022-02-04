
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes
    ----------
    in_features : int
        Input size of linear module.
    out_features : int 
        Output size of linear module.
    std_init : float 
        Initial std value.
    """

    def __init__(self, in_features, out_features, std_init):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Resets trainable network parameters (factorized Gaussian noise).
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Makes new noise distribution.
        """
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Forward method implementation.

        Returns
        -------
        torch.Tensor
            NoisyNet result after forward propagation algorithm execution.
        """
        return F.linear(
            x,
            self.weight_mu +
            self.weight_sigma *
            self.weight_epsilon,
            self.bias_mu +
            self.bias_sigma *
            self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size):
        """Sets scale to make noise (factorized Gaussian noise).

        Parameters
        ----------
        size : int
            Scale of noise.

        Returns
        -------
        torch.Tensor
            Factorized Gaussian noise.
        """
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
