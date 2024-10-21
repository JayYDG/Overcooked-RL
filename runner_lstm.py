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
from ppo_lstm import (
    Policy,
    Buffer,
    StudentPolicy,
    StudentAgent,
    map_numpy_torch,
    helper_func_obs,
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
    parser.add_argument("--layout-name", type=str, default='cramped_room', help="Layout Name")
    return parser.parse_args()


@torch.no_grad()
def evaluate_ppo(ppo_policy, env, num_episode=10):

    res_R = np.zeros(num_episode)

    for i in range(num_episode):

        next_done = False

        obs = env.reset()

        if env.agent_idx:
            both_agent_obs = np.array(obs["both_agent_obs"])
            both_agent_obs[[1, 0], :] = both_agent_obs[[0, 1], :]
        else:
            both_agent_obs = np.array(obs["both_agent_obs"])

        reshaped_obs = torch.from_numpy(helper_func_obs(both_agent_obs)).unsqueeze(0)
        prev_done = torch.zeros((1, 1))
        lstm_prev_h = torch.zeros(1, 256)
        lstm_prev_c = torch.zeros(1, 256)

        while not next_done:

            with torch.no_grad():
                _, actions, _, _, lstm_next = ppo_policy.get_action_and_value(
                    reshaped_obs,
                    (lstm_prev_h, lstm_prev_c),
                    prev_done,
                    deterministic=False,
                )

            next_obs, R, next_done, info = env.step(actions.view(-1).tolist())

            if env.agent_idx:
                both_agent_obs = np.array(next_obs["both_agent_obs"])
                both_agent_obs[[1, 0], :] = both_agent_obs[[0, 1], :]
            else:
                both_agent_obs = np.array(next_obs["both_agent_obs"])

            reshaped_obs = torch.from_numpy(helper_func_obs(both_agent_obs)).unsqueeze(
                0
            )
            prev_done = torch.ones((1, 1)) * next_done
            lstm_prev_h = lstm_next[0]
            lstm_prev_c = lstm_next[1]

        res_R[i] = R
        return res_R


@torch.no_grad()
def evaluate_policy_img_gif(
    policy, base_env, action_space, layout, horizon, update_num
):
    # Instantiate the policies for both agents
    policy0 = StudentPolicy(policy, base_env, action_space)
    policy1 = StudentPolicy(policy, base_env, action_space)

    # Instantiate both agents
    agent0 = StudentAgent(policy0)
    agent1 = StudentAgent(policy1)
    agent_pair = AgentPair(agent0, agent1)

    ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
    trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
    print("\nlen(trajs):", len(trajs))

    img_dir = f"runs/{layout}/imgs/{update_num}/"
    ipython_display = True
    gif_path = f"runs/{layout}/imgs/{update_num}/imgs.gif"

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


@ex.config
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
        # initial learning rate
        "lr": 5e-4,
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
        "evaluation_interval": 50,
        # number of total update
        "num_updates": 500,
        # use cuda
        "use_cuda": False,
        # punishment for no valid action
        "punish": -0.01,
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


@ex.automain
def run(param):
    args = parse_args()

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
    writer = SummaryWriter(f"runs/{args.layout_name} + {time.time()}")
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

    for env in envs_list:
        env.agent_idx = 0

    num_updates = param["training_param"]["num_updates"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and param["training_param"]["use_cuda"]
        else "cpu"
    )
    tpdv = dict(dtype=torch.float32, device=device)

    policy = Policy((5, 10), 6, device)

    optimizer = optim.Adam(
        policy.parameters(), lr=param["training_param"]["lr"], eps=1e-5
    )
    # optimizer = optim.SGD(policy.parameters(), lr=param["training_param"]["lr"])

    if param["training_param"]["lr_scheduler"]:
        lr_scheduler = param["training_param"]["lr_scheduler"](
            optimizer,
            total_iters=param["training_param"]["num_updates"] * 10,
            power=1.0,
            last_epoch=-1,
        )

    buffer = Buffer((5, 10), dotdict(param["buffer_param"]))

    for update in tqdm(range(1, num_updates + 1), total=num_updates):

        end_done = np.zeros(
            (
                1,
                param["training_param"]["num_envs"],
                1,
            )
        )
        end_value = np.zeros(
            (
                1,
                param["training_param"]["num_envs"],
                1,
            )
        )

        # collect training data
        for index, env in enumerate(envs_list):
            obs = env.reset()

            if env.agent_idx:
                both_agent_obs = np.array(obs["both_agent_obs"])
                both_agent_obs[[1, 0], :] = both_agent_obs[[0, 1], :]
            else:
                both_agent_obs = np.array(obs["both_agent_obs"])

            reshaped_obs = torch.from_numpy(helper_func_obs(both_agent_obs)).unsqueeze(
                0
            )
            prev_done = torch.zeros((1, 1))
            lstm_prev_h = torch.zeros((1, 256))
            lstm_prev_c = torch.zeros((1, 256))

            for step in range(param["training_param"]["num_steps"]):

                with torch.no_grad():
                    value, actions, action_log_probs, _, lstm_next = (
                        policy.get_action_and_value(
                            reshaped_obs, (lstm_prev_h, lstm_prev_c), prev_done
                        )
                    )
                next_obs, R, next_done, info = env.step(actions.view(-1).tolist())
                r_shaped = info["shaped_r_by_agent"]
                if env.agent_idx:
                    r_shaped_0 = r_shaped[1]
                    r_shaped_1 = r_shaped[0]
                    both_agent_obs = np.array(next_obs["both_agent_obs"])
                    both_agent_obs[[1, 0], :] = both_agent_obs[[0, 1], :]
                else:
                    r_shaped_0 = r_shaped[0]
                    r_shaped_1 = r_shaped[1]
                    both_agent_obs = np.array(next_obs["both_agent_obs"])

                # Encourage moving
                next_reshaped_obs = torch.from_numpy(
                    helper_func_obs(both_agent_obs)
                ).unsqueeze(0)
                bouns = check_status_change(
                    reshaped_obs, next_reshaped_obs, punish=-0.01
                )

                actor_R = (
                    torch.Tensor([r_shaped_0 + R / 2, r_shaped_1 + R / 2]).reshape(-1)
                    + bouns
                )

                data = (
                    reshaped_obs,
                    actions,
                    action_log_probs,
                    actor_R,
                    prev_done,
                    value,
                    lstm_next[0],
                    lstm_next[1],
                )

                buffer.insert(data, env_index=index)

                reshaped_obs = next_reshaped_obs
                prev_done = torch.ones((1, 1)) * next_done
                lstm_prev_h = lstm_next[0]
                lstm_prev_c = lstm_next[1]

            end_done[:, index] = prev_done
            with torch.no_grad():
                value, _, _, _, _ = policy.get_action_and_value(
                    reshaped_obs, (lstm_prev_h, lstm_prev_c), prev_done
                )
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
                    lstm_hn_samples,
                    lstm_cn_samples,
                    done_samples,
                ) = batch_data

                new_value, _, new_logprobs, new_entropy, _ = (
                    policy.get_action_and_value(
                        obs_samples,
                        (lstm_hn_samples, lstm_cn_samples),
                        done_samples,
                        action=actions_samples,
                    )
                )

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

                optimizer.zero_grad()
                loss.backward()
                if param["training_param"]["grad_clip"]:
                    nn.utils.clip_grad_norm_(
                        policy.parameters(), param["training_param"]["grad_clip"]
                    )
                optimizer.step()

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
            lr_scheduler.step()

        if update % param["training_param"]["evaluation_interval"] == 0:
            eval_res = evaluate_ppo(policy, env, num_episode=10)
            print(f"At update {update} the validate result is {eval_res}")

            evaluate_policy_img_gif(
                policy,
                base_env,
                envs_list[0].action_space,
                args.layout_name,
                param["env_param"]["horizon"],
                update,
            )

            writer.add_scalar("charts/validate_reward", np.mean(eval_res), update)

        y_pred, y_true = buffer.value.numpy(), buffer.returns.numpy()
        var_y = np.var(y_true.sum(axis=-1))
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], update
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
