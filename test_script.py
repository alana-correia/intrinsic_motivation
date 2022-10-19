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
import json
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from ppo_atari_lstm import Agent as AgentLstmBaseline
from ppo_atari_brims import AgentBrims
from ppo_atari_brims_curiosity import AgentCuriosity
from ppo_atari_lstm import make_env as make_env_baseline
from collections import defaultdict

def test_I(args, agent, run_name, device):
    scores = []
    lives = []
    lenght = []
    times = []
    for idx in range(100):
        envs = gym.vector.SyncVectorEnv(
            [make_env_baseline(args.gym_id, args.seed + i, i, args.frame_stack, args.capture_video, args.run_name, split='test') for i in
             range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
       
        path_load = os.path.join(os.getcwd(),
                                 "checkpoints")
        checkpoint = torch.load(os.path.join(path_load, f"{args.run_name}_model.pth"))
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
    print(f'run_name {run_name}')
    print(f'scores - lives - lenght - times')
    print(data)

    # or initialize from existing data
    #my_table = wandb.Table(dataframe=df)
    # log the Table directly to a project workspace
    #wandb.run.log({"env_I_stats": my_table})


def evaluate_env_II():
    pass


def function_with_args_and_default_kwargs(optional_args=None, **kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args(optional_args)
    return args


def main():
    #args = parse_args()
    run_name = "BreakoutNoFrameskip-v4__cnn_lstm_mlp_mlp_extrinsic_reward__1__1657749065"
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "BreakoutNoFrameskip-v4__cnn_lstm_mlp_mlp_extrinsic_reward__1__1657749065_args.json")
    print(checkpoint_path)
    #args = json.load(checkpoint_path)

    #f = open(checkpoint_path)
    #args = json.load(f)
    f = open(checkpoint_path, "r")
    args = json.loads(f.read())

    args = function_with_args_and_default_kwargs(**args)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if("brims" in run_name and "extrinsic" in run_name):
        agent = AgentBrims(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks, args.topk,
                   args.use_inactive, args.blocked_grad).to(device)
    elif("brims" in run_name and "hibrid" in run_name):
        agent = AgentCuriosity(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks,
                               args.topk,
                               args.use_inactive, args.blocked_grad).to(device)
    else:
        agent = AgentLstmBaseline(args.frame_stack, args.emb_size, args.lstm_output).to(device)
    # cenário de teste I - mesmo ambiente de treinamento do agente
    test_I(args, agent, run_name, device)
    # cenário de teste II - ambiente de teste do agente com estilo diferente

    # cenário de teste III - ambiente de teste do agente mais difícil

    # cenário de teste IV - ambiente de teste do agente pulando frames

    # cenário de teste V - ambiente de teste do agente com oclusões não vistas



#wandb.log({"train-video": wandb.Video(f"videos/train_{run_name}.mp4", fps=4, format="gif")})
if __name__ == "__main__":
    main()