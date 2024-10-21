import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from sacred import Experiment
from ppo_fcnn_v2 import (
    Policy,
    Buffer,
    StudentPolicy,
    StudentAgent,
    map_numpy_torch,
    helper_func_obs,
    helper_additional_reward_shaping,
    reward_shaping_dict,
)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from PIL import Image
import os
from IPython.display import display, Image as IPImage
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("PPO_Overcooked")


def check_status_change(prev_shaped_obs, next_prev_shaped_obs, punish=-0.01):
    prev_shaped_obs = prev_shaped_obs.view(2, -1)
    next_prev_shaped_obs = next_prev_shaped_obs.view(2, -1)
    c1 = torch.all((prev_shaped_obs[:, :8] == next_prev_shaped_obs[:, :8]), axis=1)
    c2 = torch.all((prev_shaped_obs[:, -2:] == next_prev_shaped_obs[:, -2:]), axis=1)
    return (c1 & c2) * punish


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=time.time(), help="Experiment Name")
    parser.add_argument("--layout-name", type=str, default='cramped_room', help="Layout Name")
    parser.add_argument("--learning-rate-shared", type=float, default=5e-4,
        help="the learning rate of the shared optimizer")
    parser.add_argument("--learning-rate-actor", type=float, default=5e-4,
        help="the learning rate of the actor optimizer")
    parser.add_argument("--learning-rate-critic", type=float, default=5e-4,
        help="the learning rate of the critic optimizer")
    parser.add_argument("--num-envs", type=int, default=10, help="Number of parallel enviorment")
    parser.add_argument("--num-minibatches", type=int, default=2, help="Number of minibatches")
    parser.add_argument("--epoch", type=int, default=8, help="Number of epoch")
    parser.add_argument("--num-updates", type=int, default=500, help="Number of update")
    parser.add_argument("--extra-reward", type=bool, default=False, help="With Extra Reward")
    parser.add_argument("--time-punish", type=float, default=0.0, help="Time punish")
    return parser.parse_args()


@torch.no_grad()
def evaluate_ppo(ppo_policy, env, num_episode=10):

    num_soups_made = np.zeros(num_episode)

    for i in range(num_episode):

        next_done = False

        next_obs = env.reset()

        reshaped_obs = helper_func_obs(next_obs["both_agent_obs"])
        first = True
        tpdv = dict(dtype=torch.float32, device=device)
        while not next_done:

            with torch.no_grad():
                _, actions, _, _ = ppo_policy.get_action_and_value(
                    reshaped_obs,
                    deterministic=False,
                )
                if first:
                    tem = ppo_policy.shared_net(reshaped_obs.to(**tpdv))
                    logits = ppo_policy.actor(tem)
                    print(env.base_env)
                    print(Categorical(logits=logits).probs)
                    first = False

            next_obs, R, next_done, info = env.step(actions.view(-1).tolist())
            reshaped_obs = helper_func_obs(next_obs["both_agent_obs"])
            num_soups_made[i] += int(R / 20)

    return num_soups_made


@torch.no_grad()
def evaluate_policy_img_gif(
    policy, base_env, action_space, layout, horizon, update_num, exp_path
):
    # Instantiate the policies for both agents
    policy0 = StudentPolicy(policy, base_env, action_space)
    policy1 = StudentPolicy(policy, base_env, action_space)

    # Instantiate both agents
    agent0 = StudentAgent(policy0)
    agent1 = StudentAgent(policy1)
    agent_pair = AgentPair(agent0, agent1)

    ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
    trajs = ae.evaluate_agent_pair(agent_pair, num_games=100)
    print("\nlen(trajs):", len(trajs))

    img_dir = f"{exp_path}/imgs/{update_num}/"
    ipython_display = True
    gif_path = f"{exp_path}/imgs/{update_num}/imgs.gif"

    StateVisualizer().display_rendered_trajectory(
        trajs, img_directory_path=img_dir, ipython_display=ipython_display
    )

    img_list = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    img_list = sorted(img_list, key=lambda x: int(x.split(".")[0]))
    images = [Image.open(img_dir + img).convert("RGBA") for img in img_list]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=250,
        loop=0,
    )
    with open(gif_path, "rb") as f:
        display(IPImage(data=f.read(), format="png"))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def default_config():

    training_params = {
        # num of parallel envs
        "num_envs": 10,
        # the number of steps to run in each environment per policy rollout
        "num_steps": 400,
        # number of minibatch
        "num_minibatches": 2,
        # epoch to update policy per training
        "epoch": 8,
        # initial shared learning rate
        "lr_shared": 5e-5,
        # initial shared learning rate
        "lr_actor": 5e-5,
        # initial shared learning rate
        "lr_critic": 5e-5,
        # learning rate scheudler
        "lr_scheduler": None,  # torch.optim.lr_scheduler.PolynomialLR,
        # Discount factor
        "gamma": 0.99,
        # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
        # Reference: https://arxiv.org/pdf/1506.02438.pdf
        "gae_lambda": 0.98,
        # normalized advantage per batch
        "norm_adv": True,
        # PPO clipping factor, if None no clipping
        "clip_param": 0.05,
        # the maximum norm for the gradient clipping, if None no clip
        "grad_clip": 0.1,
        # Target KL coeff
        "target_kl": 0.01,
        # clip value function loss or not,  if true, use the clip_param to clip
        "vf_clip": True,
        # How much the loss of the value network is weighted in overall loss
        "vf_loss_coeff": 1e-4,
        # entropy coeff schedule
        "entropy_coeff_schedule": [
            (0, 0.2),
            (3e5, 0.1),
        ],
        # evaluate policy at # training
        "evaluation_interval": 200,
        # use cuda
        "use_cuda": False,
        "extra_reward": {},
    }

    env_param = {
        "reward_shaping": {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        },
        "horizon": 400,
    }

    buffer_args = {
        "gamma": training_params["gamma"],
        "gae_lambda": training_params["gae_lambda"],
        "minibatch_size": int(
            training_params["num_envs"]
            * training_params["num_steps"]
            // training_params["num_minibatches"]
        ),
        "num_minibatches": training_params["num_minibatches"],
        "train_epoch": training_params["epoch"],
        "length": training_params["num_steps"],
        "num_head": training_params["num_envs"],
        "num_agents": 2,
        "lstm_n": 256,
    }

    param = {
        "training_param": training_params,
        "env_param": env_param,
        "buffer_param": buffer_args,
    }

    return param


param = default_config()
args = parse_args()

param["training_param"]["lr_shared"] = args.learning_rate_shared
param["training_param"]["lr_actor"] = args.learning_rate_actor
param["training_param"]["lr_critic"] = args.learning_rate_critic
param["training_param"]["num_envs"] = args.num_envs
param["training_param"]["num_minibatches"] = args.num_minibatches
param["training_param"]["epoch"] = args.epoch

param["buffer_param"]["num_head"] = args.num_envs
param["buffer_param"]["minibatch_size"] = int(
    param["training_param"]["num_envs"]
    * param["training_param"]["num_steps"]
    // param["training_param"]["num_minibatches"]
)

if args.extra_reward:
    param["training_param"]["extra_reward"] = reward_shaping_dict

param["training_param"]["time_punish"] = args.time_punish


os.mkdir(f"./runs/fcnn_v2/{args.layout_name}/{args.exp_name}")
exp_path = rf"./runs/fcnn_v2/{args.layout_name}/{args.exp_name}"


# entropy_coeff_lambda
def entropy_coeff_func(curr_update):
    entropy_coef_param = param["training_param"]["entropy_coeff_schedule"]
    slop = (entropy_coef_param[1][1] - entropy_coef_param[0][1]) / (
        entropy_coef_param[1][0] - entropy_coef_param[0][0]
    )
    curr = (
        (curr_update - 1)
        * param["training_param"]["num_envs"]
        * param["training_param"]["num_steps"]
    )
    cal_coef = slop * curr + entropy_coef_param[0][1]
    return max(entropy_coef_param[1][1], cal_coef)


# visulization
writer = SummaryWriter(exp_path)
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % (
        "\n".join(
            [f"|{key}|{value}|" for key, value in param["training_param"].items()]
        )
    ),
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# create envs
mdp = OvercookedGridworld.from_layout_name(
    args.layout_name,
    rew_shaping_params=param["env_param"]["reward_shaping"],
    # reward_shaping_horizon=param["env_param"]["reward_shaping_horizon"],
)
base_env = OvercookedEnv.from_mdp(
    mdp, horizon=param["env_param"]["horizon"], info_level=0
)
envs_list = [
    gym.make(
        "Overcooked-v0",
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp,
    )
    for _ in range(param["training_param"]["num_envs"])
]

num_updates = args.num_updates

device = torch.device(
    "cuda"
    if torch.cuda.is_available() and param["training_param"]["use_cuda"]
    else "cpu"
)
tpdv = dict(dtype=torch.float32, device=device)

policy = Policy(100, 6, device)
buffer = Buffer((100,), dotdict(param["buffer_param"]))

optimizer_shared = optim.Adam(
    list(policy.shared_net.parameters()),
    lr=param["training_param"]["lr_shared"],
)
optimizer_actor = optim.Adam(
    policy.actor.parameters(), lr=param["training_param"]["lr_actor"]
)
optimizer_critic = optim.Adam(
    policy.critic.parameters(), lr=param["training_param"]["lr_critic"]
)

optimizers = [optimizer_shared, optimizer_actor, optimizer_critic]

if param["training_param"]["lr_scheduler"]:
    shared_lr_scheduler = param["training_param"]["lr_scheduler"](
        optimizer_shared,
        total_iters=param["training_param"]["num_updates"] * 10,
        power=1.0,
        last_epoch=-1,
    )

    actor_lr_scheduler = param["training_param"]["lr_scheduler"](
        optimizer_actor,
        total_iters=param["training_param"]["num_updates"] * 10,
        power=1.0,
        last_epoch=-1,
    )

    critic_lr_scheduler = param["training_param"]["lr_scheduler"](
        optimizer_critic,
        total_iters=param["training_param"]["num_updates"] * 10,
        power=1.0,
        last_epoch=-1,
    )

    lr_schedulers = [shared_lr_scheduler, actor_lr_scheduler, critic_lr_scheduler]


for update in tqdm(range(1, num_updates + 1), total=num_updates):

    end_done = np.zeros(
        (
            1,
            param["training_param"]["num_envs"],
            2,
            1,
        )
    )
    end_value = np.zeros(
        (
            1,
            param["training_param"]["num_envs"],
            2,
            1,
        )
    )

    # collect training data
    for index, env in enumerate(envs_list):
        next_obs = env.reset()

        reshaped_obs = helper_func_obs(next_obs["both_agent_obs"])

        prev_done = torch.zeros((2, 1))

        for step in range(param["training_param"]["num_steps"]):

            with torch.no_grad():
                value, actions, action_log_probs, _ = policy.get_action_and_value(
                    reshaped_obs
                )
            next_obs, R, next_done, info = env.step(actions.view(-1).tolist())
            r_shaped = info["shaped_r_by_agent"]
            r_shaped2 = helper_additional_reward_shaping(
                step,
                env.base_env.game_stats,
                reward_dict=param["training_param"]["extra_reward"],
            )
            if env.agent_idx:
                r_shaped_0 = r_shaped[1] + r_shaped2[1]
                r_shaped_1 = r_shaped[0] + r_shaped2[0]
            else:
                r_shaped_0 = r_shaped[0] + r_shaped2[0]
                r_shaped_1 = r_shaped[1] + r_shaped2[1]

            next_reshaped_obs = helper_func_obs(next_obs["both_agent_obs"])

            actor_R = torch.Tensor(
                [r_shaped_0 + R - args.time_punish, r_shaped_1 + R - args.time_punish]
            ).reshape(-1)

            data = (
                reshaped_obs,
                actions.view(-1, 1),
                action_log_probs.view(-1, 1),
                actor_R.view(-1, 1),
                prev_done,
                value,
            )

            buffer.insert(data, env_index=index)

            reshaped_obs = next_reshaped_obs
            prev_done = torch.ones((2, 1)) * next_done

        end_done[:, index] = prev_done
        with torch.no_grad():
            value, _, _, _ = policy.get_action_and_value(reshaped_obs)
            end_value[:, index] = value

    buffer.compute_return_and_advantage(end_value, end_done)

    # training begin
    clipfracs = []
    total_loss_log = []
    v_loss_log = []
    pg_loss_log = []
    entropy_loss_log = []
    old_approx_kl_log = []
    approx_kl_log = []
    for epoch in range(param["training_param"]["epoch"]):

        data_generator = buffer.feed_forward_generator()

        for batch_data in data_generator:

            batch_data = [map_numpy_torch(d).to(**tpdv) for d in batch_data]

            (
                obs_samples,
                actions_samples,
                logprobs_samples,
                value_samples,
                advantages_samples,
                returns_samples,
            ) = batch_data

            if obs_samples.shape[0] == 0:
                break

            new_value = torch.zeros_like(value_samples)
            new_logprobs = torch.zeros_like(logprobs_samples)
            new_entropy = torch.zeros_like(logprobs_samples)

            for i in range(new_value.shape[0]):
                value_temp, _, new_logprobs_temp, new_entropy_temp = (
                    policy.get_action_and_value(
                        obs_samples[i],
                        action=actions_samples[i].reshape(-1),
                    )
                )
                new_value[i] = value_temp
                new_logprobs[i] = new_logprobs_temp.reshape(-1, 1)
                new_entropy[i] = new_entropy_temp.reshape(-1, 1)

            logratio = new_logprobs - logprobs_samples

            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > param["training_param"]["clip_param"])
                    .float()
                    .mean()
                    .item()
                ]

            if param["training_param"]["norm_adv"]:
                new_advantages_samples = (
                    advantages_samples - advantages_samples.mean()
                ) / (advantages_samples.std() + 1e-8)
            else:
                new_advantages_samples = advantages_samples

            # Policy loss
            pg_loss1 = -new_advantages_samples * ratio
            pg_loss2 = -new_advantages_samples * torch.clamp(
                ratio,
                1 - param["training_param"]["clip_param"],
                1 + param["training_param"]["clip_param"],
            )
            # max or min?
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if param["training_param"]["vf_clip"]:
                v_loss_unclipped = (new_value - returns_samples) ** 2
                v_clipped = value_samples + torch.clamp(
                    new_value - value_samples,
                    -param["training_param"]["clip_param"],
                    param["training_param"]["clip_param"],
                )
                v_loss_clipped = (v_clipped - returns_samples) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_value - returns_samples) ** 2).mean()

            entropy_loss = new_entropy.mean()

            # calculate entropy coef
            ent_coef = entropy_coeff_func(update)

            loss = (
                pg_loss
                - ent_coef * entropy_loss
                + v_loss * param["training_param"]["vf_loss_coeff"]
            )

            [optimizer.zero_grad() for optimizer in optimizers]

            loss.backward()
            if param["training_param"]["grad_clip"]:
                nn.utils.clip_grad_norm_(
                    policy.parameters(), param["training_param"]["grad_clip"]
                )
            [optimizer.step() for optimizer in optimizers]

            total_loss_log += [loss.item()]
            v_loss_log += [v_loss.item()]
            pg_loss_log += [pg_loss.item()]
            entropy_loss_log += [entropy_loss.item()]
            old_approx_kl_log += [old_approx_kl.item()]
            approx_kl_log += [approx_kl.item()]

            if param["training_param"]["target_kl"] is not None:
                if approx_kl > param["training_param"]["target_kl"]:
                    break

    if param["training_param"]["lr_scheduler"]:
        [lr_scheduler.step() for lr_scheduler in lr_schedulers]

    if update % param["training_param"]["evaluation_interval"] == 0:
        num_soups_made = evaluate_ppo(policy, env, num_episode=20)
        print(f"At update {update} the evaluation result is {num_soups_made}")

        evaluate_policy_img_gif(
            policy,
            base_env,
            envs_list[0].action_space,
            args.layout_name,
            param["env_param"]["horizon"],
            update,
            exp_path,
        )

        writer.add_scalar("charts/mean_num_soups_made", np.mean(num_soups_made), update)

        policy.save_model(os.path.join(exp_path, f"{update}-model.pt"))

    y_pred, y_true = buffer.value.numpy().squeeze(), buffer.returns.numpy().squeeze()
    y_pred = y_pred.sum(axis=-1).reshape(-1)
    y_true = y_true.sum(axis=-1).reshape(-1)
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    writer.add_scalar(
        "charts/learning_rate_shared", optimizer_shared.param_groups[0]["lr"], update
    )
    writer.add_scalar(
        "charts/train_reward", buffer.rewards.sum(axis=[0, 2]).mean(), update
    )
    writer.add_scalar(
        "charts/train_soup",
        (buffer.rewards.sum(axis=2) > 15).sum(axis=0).float().mean(),
        update,
    )
    writer.add_scalar("charts/entropy_coef", ent_coef, update)
    writer.add_scalar("losses/total_loss", np.mean(total_loss_log), update)
    writer.add_scalar("losses/value_loss", np.mean(v_loss_log), update)
    writer.add_scalar("losses/policy_loss", np.mean(pg_loss_log), update)
    writer.add_scalar("losses/entropy", np.mean(entropy_loss_log), update)
    writer.add_scalar("losses/old_approx_kl", np.mean(old_approx_kl_log), update)
    writer.add_scalar("losses/approx_kl", np.mean(approx_kl_log), update)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), update)
    writer.add_scalar("losses/explained_variance", explained_var, update)

writer.close()
