import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import EVENT_TYPES

reward_shaping_dict = {
    "useful_onion_pickup": 0.5,
    "useful_dish_pickup": 0.5,
    "soup_drop": -0.5,
    "soup_pickup": 0.5,
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


def helper_func_obs(env, overcookstate):
    temp_lossless_state = np.array(env.mdp.lossless_state_encoding(overcookstate))
    return torch.permute(torch.from_numpy(temp_lossless_state), (0, 3, 1, 2))


class Policy(nn.Module):
    def __init__(self, cent_obs_space: tuple, action_space: int, device):
        super().__init__()

        self.cnn_net = nn.Sequential(
            nn.Conv2d(26, 25, kernel_size=(5, 5), padding="same", stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(25, 25, kernel_size=(3, 3), padding="same", stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(25, 25, kernel_size=(3, 3), padding="valid", stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.fcc_net = nn.Sequential(
            nn.Linear(150, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )

        self.critic = layer_init(nn.Linear(64, 1), std=1)

        self.actor = layer_init(nn.Linear(64, 6), std=0.01)

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.cent_obs_space = cent_obs_space
        self.action_space = action_space

    def forward_cnn_fcc_net(self, x):
        new_x = map_numpy_torch(x).to(**self.tpdv)
        new_x = self.cnn_net(new_x).view(new_x.shape[0], -1)
        return self.fcc_net(new_x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        x : in [batch_size, num_agent(2), H, W],
        lstm_state: in ((batch_size, 256), (batch_size, 256))
        done: in (batch_size, 1)
        action: in (batch_size, num_agent(2))

        return:
        value: [batch_size, 1]
        action : [batch_size, num_agent(2)]
        log_prob : [batch_size, num_agent(2)]
        entropy: [batch_size, num_agent(2)]
        lstm_state: ((batch_size, 256), (batch_size, 256))
        """
        hidden = self.forward_cnn_fcc_net(x)
        value = self.critic(hidden)
        # (batch_size, 256) -> (batch_size, num_agent(2), 256)
        # hidden = hidden.unsqueeze(1).repeat(1, 2, 1)
        # logits in (batch_size, 2, action_space)
        logits = self.actor(hidden)
        # probs in (batch_size, 2)
        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = probs.mode
            else:
                action = probs.sample()
        else:
            action = map_numpy_torch(action).to(**self.tpdv)

        return value, action, probs.log_prob(action), probs.entropy()

    def evaluate_mode(self):
        self.cnn_net.eval()
        self.fcc_net.eval()
        self.critic.eval()
        self.actor.eval()

    def train_mode(self):
        self.cnn_net.train()
        self.fcc_net.train()
        self.critic.train()
        self.actor.train()

    def save_model(self, path):
        torch.save(
            {
                "cnn_net_state_dict": self.cnn_net.state_dict(),
                "fcc_net_state_dict": self.fcc_net.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        params_dict = torch.load(path)
        self.cnn_net.load_state_dict(params_dict["cnn_net_state_dict"])
        self.fcc_net.load_state_dict(params_dict["fcc_net_state_dict"])
        self.critic.load_state_dict(params_dict["critic_state_dict"])
        self.actor.load_state_dict(params_dict["actor_state_dict"])


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
        self.base_env.agent_idx = agent_index
        reshaped_obs = helper_func_obs(self.base_env, state)

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
