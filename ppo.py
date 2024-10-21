import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def map_numpy_torch(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(array)


class Actor(nn.Module):

    def __init__(self, obs_space, action_space, device) -> None:
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 1024)),
            nn.ReLU(),
            # nn.LayerNorm(1024),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
            # nn.LayerNorm(512),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            # nn.LayerNorm(256),
            layer_init(nn.Linear(256, 64)),
            nn.ReLU(),
            # nn.LayerNorm(64),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
            # nn.LayerNorm(action_space.n),
        )

        self.tpdv = dict(dtype=torch.float32, device=device)

    def get_action(self, x, action=None, deterministic=False):
        x = map_numpy_torch(x).to(**self.tpdv)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = probs.mode
            else:
                action = probs.sample()
        else:
            action = map_numpy_torch(action).to(**self.tpdv)
        return action, probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    def __init__(self, cent_obs_space, device) -> None:
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(cent_obs_space).prod(), 2048)),
            nn.ReLU(),
            # nn.LayerNorm(2048),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            # nn.LayerNorm(1024),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
            # nn.LayerNorm(512),
            layer_init(nn.Linear(512, 64)),
            nn.ReLU(),
            # nn.LayerNorm(64),
            layer_init(nn.Linear(64, 1), std=1),
        )

        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x):
        x = map_numpy_torch(x).to(**self.tpdv)
        return self.critic(x)


class Policy(nn.Module):
    def __init__(self, cent_obs_space, obs_space, action_space, device) -> None:
        super(Policy, self).__init__()

        self.critic = Critic(cent_obs_space=cent_obs_space, device=device)
        self.actor = Actor(
            obs_space=obs_space, action_space=action_space, device=device
        )

    def get_action_and_value(self, cent_obs, obs):

        actions, action_log_probs, _ = self.actor.get_action(obs)

        critic_values = self.critic(cent_obs)

        return actions, action_log_probs, critic_values

    def act(self, obs, deterministic=False):
        actions, _, _ = self.actor.get_action(obs, deterministic=deterministic)
        return actions

    def evaluate_action(self, cent_obs, obs, actions):

        _, action_log_probs, entropy = self.actor.get_action(obs, action=actions)

        value = self.critic(cent_obs)

        return action_log_probs, entropy, value

    def evaluate_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()


class Buffer:

    def __init__(self, cent_obs_space, obs_space, args):
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.minibatch_size = args.minibatch_size
        self.num_minibatches = args.num_minibatches

        self.length = args.length
        self.num_head = args.num_head
        self.num_agents = args.num_agents

        self.cent_obs = np.zeros(
            (self.length, self.num_head, self.num_agents, cent_obs_space),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (
                self.length,
                self.num_head,
                self.num_agents,
                np.array(obs_space.shape).prod(),
            ),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.length, self.num_head, self.num_agents, 1), dtype=np.float32
        )
        self.logprobs = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.logprobs)
        self.done = np.zeros_like(self.rewards)
        self.value = np.zeros_like(self.done)

        self.step = 0

    def insert(self, data, env_index):
        cent_obs, obs, actions, logprobs, rewards, done, value = data

        self.cent_obs[self.step, env_index] = cent_obs
        self.obs[self.step, env_index] = obs
        self.actions[self.step, env_index] = actions
        self.logprobs[self.step, env_index] = logprobs
        self.rewards[self.step, env_index] = rewards
        self.done[self.step, env_index] = done
        self.value[self.step, env_index] = value

        self.step = (1 + self.step) % self.length

    def compute_return_and_advantage(self, next_value, next_done):
        self.advantages = np.zeros_like(self.rewards)
        lastgaelam = 0
        for t in reversed(range(self.length)):
            if t == self.length - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.done[t + 1]
                nextvalues = self.value[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * nextvalues * nextnonterminal
                - self.value[t]
            )
            self.advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )
        self.returns = self.advantages + self.value

    def feed_forward_generator(self):
        ## Flatten data
        b_cent_obs = self.cent_obs.reshape((-1,) + self.cent_obs.shape[3:])
        b_obs = self.obs.reshape((-1,) + self.obs.shape[3:])
        b_actions = self.actions.reshape((-1,) + self.actions.shape[3:])
        b_logprobs = self.logprobs.reshape((-1,) + self.logprobs.shape[3:])
        # b_rewards = self.rewards.reshape((-1,) + self.rewards.shape[3:])
        # b_done = self.done.reshape((-1,) + self.done.shape[3:])
        b_value = self.value.reshape((-1,) + self.value.shape[3:])
        b_advantages = self.advantages.reshape((-1,) + self.advantages.shape[3:])
        b_returns = self.returns.reshape((-1,) + self.returns.shape[3:])

        rand = torch.randperm(self.length * self.num_head).numpy()
        sampler = [
            rand[i * self.minibatch_size : (i + 1) * self.minibatch_size]
            for i in range(self.num_minibatches)
        ]

        for indices in sampler:
            share_obs_samples = b_cent_obs[indices, :]
            obs_samples = b_obs[indices, :]
            actions_samples = b_actions[indices, :]
            logprobs_samples = b_logprobs[indices, :]
            value_samples = b_value[indices, :]
            advantages_samples = b_advantages[indices, :]
            returns_samples = b_returns[indices, :]

            yield (
                share_obs_samples,
                obs_samples,
                actions_samples,
                logprobs_samples,
                value_samples,
                advantages_samples,
                returns_samples,
            )


class StudentPolicy(NNPolicy):
    """Generate policy"""

    def __init__(self, policy, base_env, action_space):
        super(StudentPolicy, self).__init__()
        self.policy = policy
        self.base_env = base_env
        self.action_space = action_space

    def state_policy(self, state, agent_index):
        """
        This method should be used to generate the poiicy vector corresponding to
        the state and agent_index provided as input.  If you're using a neural
        network-based solution, the specifics depend on the algorithm you are using.
        Below are two commented examples, the first for a policy gradient algorithm
        and the second for a value-based algorithm.  In policy gradient algorithms,
        the neural networks output a policy directly.  In value-based algorithms,
        the policy must be derived from the Q value outputs of the networks.  The
        uncommented code below is a placeholder that generates a random policy.
        """
        featurized_state = self.base_env.featurize_state_mdp(state)
        input_obs = featurized_state[agent_index]

        # Example for policy NNs named "PNN0" and "PNN1"
        with torch.no_grad():
            action = self.policy.act(np.array(input_obs), deterministic=True)

        action_probs = np.zeros(self.action_space.n)
        action_probs[action] = 1
        return action_probs

    def multi_state_policy(self, states, agent_indices):
        """Generate a policy for a list of states and agent indices"""
        res = [
            self.state_policy(state, agent_index)
            for state, agent_index in zip(states, agent_indices)
        ]
        return res


class StudentAgent(AgentFromPolicy):
    """Create an agent using the policy created by the class above"""

    def __init__(self, policy):
        super(StudentAgent, self).__init__(policy)
