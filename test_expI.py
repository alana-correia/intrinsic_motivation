import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import deque
import gym
import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from ppo_atari_lstm import Agent as AgentLstmBaseline
from ppo_atari_lstm import make_env as make_env_baseline
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="XXXX/XXXXX",
                        help="the name of this experiment")
    parser.add_argument("--agent_type" , type=str, default="cnn_lstm_mlp_mlp_extrinsic_reward",
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--num_games", type=int, default=100,
                        help="number of games")
    parser.add_argument("--frame_stack", type=int, default=4,
                        help="frame stack num")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    parser.add_argument("--emb_size", type=int, default=512,
                        help="embedding size")

    parser.add_argument("--lstm_output", type=int, default=128,
                        help="embedding size")

    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")

    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    args = parser.parse_args()
    return args



def evaluate_env_I(args, device, run_name="BreakoutNoFrameskip-v4__cnn_lstm_mlp_mlp_extrinsic_reward__1__1657749065"):
    scores = []
    lives = []
    lenght = []
    times = []
    for idx in range(args.num_games):
        envs = gym.vector.SyncVectorEnv(
            [make_env_baseline(args.gym_id, args.seed + i, i, args.frame_stack, args.capture_video, run_name, split='test') for i in
             range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        # print(env_I.single_action_space.n)
        agent = AgentLstmBaseline(args.frame_stack, args.emb_size, args.lstm_output).to(device)
        path_load = os.path.join(os.getcwd(),
                                 "checkpoints")
        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        )
        score = 0
        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state,
                                                                                        next_done)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                score += reward[-1]
                next_obs = torch.Tensor(next_obs).to(device)
        scores.append(info[-1]['episode']['r'])
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])



    data = [[np.mean(scores)], [np.mean(lives)], [np.mean(lenght)], [np.mean(times)]]
    columns = ["scores", "lives", "lenght", "times"]
    print(data)

    # or initialize from existing data
    #my_table = wandb.Table(dataframe=df)
    # log the Table directly to a project workspace
    #wandb.run.log({"env_I_stats": my_table})


def evaluate_env_II():
    pass



def main():
    args = parse_args()
    run_name = "BreakoutNoFrameskip-v4__cnn_lstm_mlp_mlp_extrinsic_reward__1__1657749065"



    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    evaluate_env_I(args, device)

#wandb.log({"train-video": wandb.Video(f"videos/train_{run_name}.mp4", fps=4, format="gif")})
if __name__ == "__main__":
    main()