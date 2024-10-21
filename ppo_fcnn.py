import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair

reward_shaping_dict = {
    "useful_onion_pickup": 0.5,
    "useful_dish_pickup": 0.5,
    "soup_drop": -0.25,
    "soup_pickup": 1,
}


def helper_additional_reward_shaping(current_step, game_stats):
    # Initialize reward counters for each agent
    reward_agent1 = 0
    reward_agent2 = 0

    # Iterate through each event in the events dictionary
    for event, agents_steps in game_stats.items():
        if event in EVENT_TYPES:
            # Check if the step is in each agent's list
            if current_step in agents_steps[0]:
                reward_agent1 += reward_shaping_dict.get((event), 0)
            if current_step in agents_steps[1]:
                reward_agent2 += reward_shaping_dict.get((event), 0)

    return [reward_agent1, reward_agent2]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def map_numpy_torch(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(array)


def helper_func_obs(obs):
    return (
        torch.from_numpy(
            np.vstack(
                (
                    np.concatenate((obs[0], obs[1][-4:])),
                    np.concatenate((obs[1], obs[0][-4:])),
                )
            )
        )
    ).to()


# class Actor(nn.Module):

#     def __init__(self, obs_space, action_space, device) -> None:
#         super(Actor, self).__init__()

#         self.actor = nn.Sequential(
#             layer_init(nn.Linear(np.array(obs_space.shape).prod(), 1024)),
#             nn.ReLU(),
#             nn.LayerNorm(1024),
#             layer_init(nn.Linear(1024, 512)),
#             nn.ReLU(),
#             nn.LayerNorm(512),
#             layer_init(nn.Linear(512, 256)),
#             nn.ReLU(),
#             nn.LayerNorm(256),
#             layer_init(nn.Linear(256, 64)),
#             nn.ReLU(),
#             nn.LayerNorm(64),
#             layer_init(nn.Linear(64, action_space.n), std=0.01),
#             nn.LayerNorm(action_space.n),
#         )

#         self.tpdv = dict(dtype=torch.float32, device=device)

#     def get_action(self, x, action=None, deterministic=False):
#         x = map_numpy_torch(x).to(**self.tpdv)
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             if deterministic:
#                 action = probs.mode
#             else:
#                 action = probs.sample()
#         else:
#             action = map_numpy_torch(action).to(**self.tpdv)
#         return action, probs.log_prob(action), probs.entropy()


# class Critic(nn.Module):
#     def __init__(self, cent_obs_space, device) -> None:
#         super(Critic, self).__init__()

#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(np.array(cent_obs_space).prod(), 2048)),
#             nn.ReLU(),
#             nn.LayerNorm(2048),
#             layer_init(nn.Linear(2048, 1024)),
#             nn.ReLU(),
#             nn.LayerNorm(1024),
#             layer_init(nn.Linear(1024, 512)),
#             nn.ReLU(),
#             nn.LayerNorm(512),
#             layer_init(nn.Linear(512, 64)),
#             nn.ReLU(),
#             nn.LayerNorm(64),
#             layer_init(nn.Linear(64, 1), std=1),
#         )

#         self.tpdv = dict(dtype=torch.float32, device=device)

#     def forward(self, x):
#         x = map_numpy_torch(x).to(**self.tpdv)
#         return self.critic(x)


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, device) -> None:
        super().__init__()

        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(obs_space, 1024)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LeakyReLU(),
        )

        self.actor = layer_init(nn.Linear(128, action_space), std=0.01)

        self.critic = layer_init(nn.Linear(128, 1), std=1)

        self.tpdv = dict(dtype=torch.float32, device=device)

    def get_action_and_value(self, obs, action=None, deterministic=False):
        obs = map_numpy_torch(obs).to(**self.tpdv)
        hidden = self.shared_net(obs)

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = probs.mode
            else:
                action = probs.sample()
        else:
            action = map_numpy_torch(action).to(**self.tpdv)

        value = self.critic(hidden)

        return value, action, probs.log_prob(action), probs.entropy()

    def act(self, obs, deterministic=False):
        obs = map_numpy_torch(obs).to(**self.tpdv)
        hidden = self.shared_net(obs)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if deterministic:
            action = probs.mode
        else:
            action = probs.sample()
        return action

    def evaluate_mode(self):
        self.shared_net.eval()
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.shared_net.train()
        self.actor.train()
        self.critic.train()

    def save_model(self, path):
        torch.save(
            {
                "shared_net": self.shared_net.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        params_dict = torch.load(path)
        self.shared_net.load_state_dict(params_dict["shared_net"])
        self.actor.load_state_dict(params_dict["actor_state_dict"])
        self.critic.load_state_dict(params_dict["critic_state_dict"])


class Buffer:

    def __init__(self, obs_space: tuple, args):
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.minibatch_size = args.minibatch_size
        self.num_minibatches = args.num_minibatches

        self.length = args.length
        self.num_head = args.num_head
        self.num_agents = args.num_agents

        self.obs = torch.zeros(
            (self.length, self.num_head, self.num_agents) + obs_space,
            dtype=torch.float32,
        )
        self.actions = torch.zeros(
            (self.length, self.num_head, self.num_agents, 1), dtype=torch.float32
        )
        self.logprobs = torch.zeros_like(self.actions)
        self.done = torch.zeros_like(self.logprobs)
        self.rewards = torch.zeros_like(self.logprobs)
        self.value = torch.zeros_like(self.logprobs)  # torch.zeros_like(self.done)
        self.step = 0

    def insert(self, data, env_index):
        obs, actions, logprobs, rewards, done, value = data

        self.obs[self.step, env_index] = obs
        self.actions[self.step, env_index] = actions
        self.logprobs[self.step, env_index] = logprobs
        self.rewards[self.step, env_index] = rewards
        self.done[self.step, env_index] = done
        self.value[self.step, env_index] = value

        self.step = (1 + self.step) % self.length

    def compute_return_and_advantage(self, next_value, next_done):
        self.advantages = torch.zeros_like(self.rewards)
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
        b_obs = self.obs.reshape((-1,) + self.obs.shape[3:])
        b_actions = self.actions.reshape((-1,) + self.actions.shape[3:])
        b_logprobs = self.logprobs.reshape((-1,) + self.logprobs.shape[3:])
        b_value = self.value.reshape((-1,) + self.value.shape[3:])
        b_advantages = self.advantages.reshape((-1,) + self.advantages.shape[3:])
        b_returns = self.returns.reshape((-1,) + self.returns.shape[3:])

        rand = torch.randperm(self.length * self.num_head).numpy()
        sampler = [
            rand[i * self.minibatch_size : (i + 1) * self.minibatch_size]
            for i in range(self.num_minibatches)
        ]

        for indices in sampler:
            obs_samples = b_obs[indices, :]
            actions_samples = b_actions[indices, :]
            logprobs_samples = b_logprobs[indices, :]
            value_samples = b_value[indices, :]
            advantages_samples = b_advantages[indices, :]
            returns_samples = b_returns[indices, :]

            yield (
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
        super().__init__()
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
        # featurized_state = self.base_env.featurize_state_mdp(state)
        featurized_state = self.base_env.featurize_state_mdp(state)
        reshaped_obs = helper_func_obs(featurized_state)

        # Example for policy NNs named "PNN0" and "PNN1"
        with torch.no_grad():
            _, actions, _, _ = self.policy.get_action_and_value(
                reshaped_obs,
                deterministic=False,
            )

        action_probs = np.zeros(self.action_space.n)
        action_probs[actions[agent_index]] = 1
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
        super().__init__(policy)


class StudentAgent(AgentFromPolicy):
    """Create an agent using the policy created by the class above"""

    def __init__(self, policy):
        super(StudentAgent, self).__init__(policy)
