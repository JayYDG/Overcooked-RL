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


def helper_func_obs(both_agent_obs):
    first_36 = both_agent_obs[:, :46]
    # Select the last 4 elements from axis 1
    last_4 = both_agent_obs[:, -4:]
    # Concatenate the selected parts along axis 1
    selected_parts = np.concatenate([first_36, last_4], axis=1)
    return selected_parts.reshape(2, 5, 10)


class ActorLayer(nn.Module):
    def __init__(self, action_space, num_agent=2):
        super().__init__()
        self.actor_layers = [
            layer_init(nn.Linear(256, action_space), std=0.01) for _ in range(num_agent)
        ]
        self.num_agent = num_agent
        self.action_space = action_space

    def forward(self, x):
        assert x.shape[1] == self.num_agent
        res = []
        for agent_index in range(self.num_agent):
            res += [self.actor_layers[agent_index](x[:, agent_index, :])]
        return torch.stack(res, dim=1)


class Policy(nn.Module):
    def __init__(self, cent_obs_space: tuple, action_space: int, device):
        super().__init__()

        self.cnn_net = nn.Sequential(
            nn.Conv2d(2, 25, kernel_size=(5, 5), padding="same", stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(25, 25, kernel_size=(3, 3), padding="valid", stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(25, 25, kernel_size=(3, 3), padding="valid", stride=(1, 1)),
            nn.LeakyReLU(),
        )

        self.fcc_net = nn.Sequential(
            nn.Linear(150, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        )

        self.lstm_net = nn.LSTM(64, 256, batch_first=True)

        self.critic = layer_init(nn.Linear(256, 1), std=1)

        self.actor = ActorLayer(action_space)

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.cent_obs_space = cent_obs_space
        self.action_space = action_space

    def forward_cnn_fcc_net(self, x):
        new_x = map_numpy_torch(x).to(**self.tpdv)
        new_x = self.cnn_net(new_x).view(new_x.shape[0], -1)
        return self.fcc_net(new_x)

    def forward_lstm_net(self, hidden, lstm_state, done):
        """
        return:
            (batch_size, 256),
            ((batch_size, 256), (batch_size, 256))
        """
        hidden = map_numpy_torch(hidden).to(**self.tpdv).unsqueeze(1)
        done = map_numpy_torch(done).to(**self.tpdv)
        prev_h, prev_c = lstm_state
        prev_h = map_numpy_torch(prev_h).to(**self.tpdv)
        prev_c = map_numpy_torch(prev_c).to(**self.tpdv)
        prev_h = ((1.0 - done) * prev_h).unsqueeze(0)
        prev_c = ((1.0 - done) * prev_c).unsqueeze(0)
        hidden, (next_h, next_c) = self.lstm_net(hidden, (prev_h, prev_c))
        return hidden.squeeze(dim=1), (next_h.squeeze(dim=0), next_c.squeeze(dim=0))

    def get_action_and_value(
        self, x, lstm_state, done, action=None, deterministic=False
    ):
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
        hidden, lstm_state = self.forward_lstm_net(hidden, lstm_state, done)
        value = self.critic(hidden)
        # (batch_size, 256) -> (batch_size, num_agent(2), 256)
        hidden = hidden.unsqueeze(1).repeat(1, 2, 1)
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

        return value, action, probs.log_prob(action), probs.entropy(), lstm_state

    def act(self, x, lstm_state, done, deterministic=False, actor_index=None):
        hidden = self.forward_cnn_fcc_net(x)
        hidden, lstm_state = self.forward_lstm_net(hidden, lstm_state, done)
        # (batch_size, 256) -> (batch_size, num_agent(2), 256)
        hidden = hidden.unsqueeze(1).repeat(1, 2, 1)
        # logits in (batch_size, 2, action_space)
        logits = self.actor(hidden)
        # probs in (batch_size, 2)
        probs = Categorical(logits=logits)

        if deterministic:
            action = probs.mode
        else:
            action = probs.sample()

        if actor_index is None:
            return action
        return action[0, actor_index], lstm_state

    def evaluate_mode(self):
        self.cnn_net.eval()
        self.fcc_net.eval()
        self.lstm_net.eval()
        self.critic.eval()
        self.actor.eval()

    def train_mode(self):
        self.cnn_net.train()
        self.fcc_net.train()
        self.lstm_net.train()
        self.critic.train()
        self.actor.train()


class Buffer:

    def __init__(self, obs_space: tuple, args):
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.minibatch_size = args.minibatch_size
        self.num_minibatches = args.num_minibatches

        self.length = args.length
        self.num_head = args.num_head
        self.num_agents = args.num_agents
        self.lstm_n = args.lstm_n

        self.obs = torch.zeros(
            (self.length, self.num_head, self.num_agents) + obs_space,
            dtype=torch.float32,
        )
        self.actions = torch.zeros(
            (self.length, self.num_head, self.num_agents), dtype=torch.float32
        )
        self.logprobs = torch.zeros_like(self.actions)
        self.rewards = torch.zeros_like(self.logprobs)
        self.done = torch.zeros((self.length, self.num_head, 1), dtype=torch.float32)
        self.value = torch.zeros_like(self.done)
        self.lstm_hn = torch.zeros(
            (self.length, self.num_head, self.lstm_n), dtype=torch.float32
        )
        self.lstm_cn = torch.zeros(
            (self.length, self.num_head, self.lstm_n), dtype=torch.float32
        )

        self.step = 0

    def insert(self, data, env_index):
        obs, actions, logprobs, rewards, done, value, lstm_hn, lstm_cn = data

        self.obs[self.step, env_index] = obs
        self.actions[self.step, env_index] = actions
        self.logprobs[self.step, env_index] = logprobs
        self.rewards[self.step, env_index] = rewards
        self.done[self.step, env_index] = done
        self.value[self.step, env_index] = value
        self.lstm_hn[self.step, env_index] = lstm_hn
        self.lstm_cn[self.step, env_index] = lstm_cn

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
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        b_logprobs = self.logprobs.reshape((-1,) + self.logprobs.shape[2:])
        b_value = self.value.reshape((-1,) + self.value.shape[2:])
        b_advantages = self.advantages.reshape((-1,) + self.advantages.shape[2:])
        b_returns = self.returns.reshape((-1,) + self.returns.shape[2:])
        b_lstm_hn = self.lstm_hn.reshape((-1,) + self.lstm_hn.shape[2:])
        b_lstm_cn = self.lstm_cn.reshape((-1,) + self.lstm_cn.shape[2:])
        b_done = self.done.reshape((-1,) + self.value.shape[2:])

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
            lstm_hn_samples = b_lstm_hn[indices, :]
            lstm_cn_samples = b_lstm_cn[indices, :]
            done_samples = b_done[indices, :]

            yield (
                obs_samples,
                actions_samples,
                logprobs_samples,
                value_samples,
                advantages_samples,
                returns_samples,
                lstm_hn_samples,
                lstm_cn_samples,
                done_samples,
            )


class StudentPolicy(NNPolicy):
    """Generate policy"""

    def __init__(self, policy, base_env, action_space):
        super().__init__()
        self.policy = policy
        self.base_env = base_env
        self.action_space = action_space
        self.lstm_prev_h = torch.zeros(1, 256)
        self.lstm_prev_c = torch.zeros(1, 256)
        self.done = torch.zeros((1, 1))

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
        if agent_index:
            both_agent_obs = np.array(featurized_state)
            both_agent_obs[[1, 0], :] = both_agent_obs[[0, 1], :]
        else:
            both_agent_obs = np.array(featurized_state)
        reshaped_obs = torch.from_numpy(helper_func_obs(both_agent_obs)).unsqueeze(0)

        # Example for policy NNs named "PNN0" and "PNN1"
        with torch.no_grad():
            action, lstm_state = self.policy.act(
                reshaped_obs,
                (self.lstm_prev_h, self.lstm_prev_c),
                self.done,
                deterministic=True,
                actor_index=agent_index,
            )

        self.lstm_prev_h = lstm_state[0]
        self.lstm_prev_c = lstm_state[1]
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
        super().__init__(policy)
