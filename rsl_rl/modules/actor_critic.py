# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

class ScalarFiLM(nn.Module):
    def __init__(self, cond_dim, film_hidden_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, film_hidden_dims[0]),
            nn.GELU(),
            nn.Linear(film_hidden_dims[0], film_hidden_dims[1]),
            nn.GELU(),
            nn.Linear(film_hidden_dims[1], 2)
        )

    def forward(self, x, cond):
        gamma_beta = self.mlp(cond)
        gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * x + beta
    
class FiLMActor(nn.Module):
    def __init__(self, mlp_input_dim, actor_hidden_dims, num_actions, cond_dim, film_hidden_dims, activation):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(mlp_input_dim, actor_hidden_dims[0])
        self.film = ScalarFiLM(cond_dim, film_hidden_dims)
        self.cond_dim = cond_dim

        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(actor_hidden_dims)):
            self.hidden_layers.append(nn.Linear(actor_hidden_dims[i - 1], actor_hidden_dims[i]))

        self.output_layer = nn.Linear(actor_hidden_dims[-1], num_actions)
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.activation(self.fc1(obs))
        x = self.film(x, obs[:, -self.cond_dim:])

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.tanh(self.output_layer(x))
        return x

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256],
        film_hidden_dims=[3, 3],
        critic_hidden_dims=[256, 256],
        cond_dim=2,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        min_std=0.2,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        self.actor = FiLMActor(
            mlp_input_dim=mlp_input_dim_a,
            actor_hidden_dims=actor_hidden_dims,
            num_actions=num_actions,
            cond_dim=cond_dim,
            film_hidden_dims=film_hidden_dims,
            activation=activation
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        self.min_std = min_std

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = torch.max(torch.tensor(self.min_std), self.std).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
