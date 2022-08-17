import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import deque
import gym
import matplotlib
import numpy as np
import pandas as pd
import torch
import json
import wandb
import glob
import sys
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from ppo_atari_lstm import Agent as AgentLstmBaseline
from ppo_atari_brims import AgentBrims
from ppo_atari_brims_curiosity import AgentCuriosity
from ppo_atari_brims_curiosity import make_env as make_env_curiosity
from collections import defaultdict

import os
import moviepy.video.io.ImageSequenceClip

def test_I(args, agent, run_name, path_load, path_save, test_name, num_games, num_envs, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + i, i, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_best_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        #next_done = torch.zeros(num_envs).to(device)

        #next_lstm_state = agent.init_hidden(1)
        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []
        im_reward_game = []
        total_reward_game = []


        while not done:
            with torch.no_grad():
                #action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)

                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)


                #print(im_reward)
                if action.cpu().numpy()[-1] == 1:
                    print('FIRE')

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]
                total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('game {} - reward (sum) {} - im_reward {} - done {} - action {} - lives {}'.format(idx, r, im_reward.data.cpu().numpy(), done, action.cpu().numpy(), info[-1]['lives']), end='\n')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()
                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])
                    im_reward_game.append(im_reward.data.cpu().numpy()[-1])
                    total_reward_game.append(total_reward)
                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1

        if idx == 0:

            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')
            plt.show()
            #plt.get_figure().savefig(os.path.join(path_save, 'actions.pdf'), format='pdf')

            plt.plot(ex_reward_game, 'b')
            plt.plot(im_reward_game, 'r')
            plt.plot(total_reward_game, 'g')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.show()

            #plt.get_figure().savefig(os.path.join(path_save, 'lives.pdf'), format='pdf')

            #plt.plot(dones)
            #plt.xlabel('Step')
            #plt.ylabel('Dones')
            #plt.show()
            #plt.get_figure().savefig(os.path.join(path_save, 'dones.pdf'), format='pdf')

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video),
                                 key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}.mp4"))


        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"]
    df_stats = pd.DataFrame([stats], columns=columns)

    data_dict = {'scores': pd.Series(scores),
                 'rewards': pd.Series(rewards),
                 'lives': pd.Series(lives),
                 'lenght': pd.Series(lenght),
                 'times': pd.Series(times)}

    data_df = pd.DataFrame(data_dict)
    print(f'{test_name} stats:')
    print(df_stats)
    print(f'{test_name} results:')
    print(data_df)

    plot = data_df['scores'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_II(args, agent, run_name, path_load, path_save, test_name, num_games, num_envs, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + i, i, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_best_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        r = 0
        k = 0
        while not done:
            with torch.no_grad():
                # action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)

                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                #next_obs, reward, done, info = envs.step(int(action.cpu().numpy()))
                next_obs = 255 - next_obs
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()
                matplotlib.image.imsave(os.path.join(path_save, f'{k}.png'), obs_np, cmap='gray')
                k += 1
        if idx == 0:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/*.png'
            image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"]
    df_stats = pd.DataFrame([stats], columns=columns)

    data_dict = {'scores': pd.Series(scores),
                 'rewards': pd.Series(rewards),
                 'lives': pd.Series(lives),
                 'lenght': pd.Series(lenght),
                 'times': pd.Series(times)}

    data_df = pd.DataFrame(data_dict)
    print(f'{test_name} stats:')
    print(df_stats)
    print(f'{test_name} results:')
    print(data_df)

    plot = data_df['scores'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_III(args, agent, run_name, path_load, path_save, test_name, mode, num_games, num_envs, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + i, i, args.frame_stack, False, run_name, mode=mode, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_best_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        r = 0
        k = 0
        while not done:
            with torch.no_grad():
                # action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)

                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                #next_obs, reward, done, info = envs.step(int(action.cpu().numpy()))
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()
                matplotlib.image.imsave(os.path.join(path_save, f'{k}.png'), obs_np, cmap='gray')
                k += 1
        if idx == 0:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/*.png'
            image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"]
    df_stats = pd.DataFrame([stats], columns=columns)

    data_dict = {'scores': pd.Series(scores),
                 'rewards': pd.Series(rewards),
                 'lives': pd.Series(lives),
                 'lenght': pd.Series(lenght),
                 'times': pd.Series(times)}

    data_df = pd.DataFrame(data_dict)
    print(f'{test_name} stats:')
    print(df_stats)
    print(f'{test_name} results:')
    print(data_df)

    plot = data_df['scores'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_IV(args, agent, run_name, path_load, path_save, test_name, mode, difficulty, num_games, num_envs, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + i, i, args.frame_stack, False, run_name, mode=mode, difficulty=difficulty, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_best_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        r = 0
        k = 0
        while not done:
            with torch.no_grad():
                # action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)

                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                #next_obs, reward, done, info = envs.step(int(action.cpu().numpy()))
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()
                matplotlib.image.imsave(os.path.join(path_save, f'{k}.png'), obs_np, cmap='gray')
                k += 1
        if idx == 0:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/*.png'
            image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"]
    df_stats = pd.DataFrame([stats], columns=columns)

    data_dict = {'scores': pd.Series(scores),
                 'rewards': pd.Series(rewards),
                 'lives': pd.Series(lives),
                 'lenght': pd.Series(lenght),
                 'times': pd.Series(times)}

    data_df = pd.DataFrame(data_dict)
    print(f'{test_name} stats:')
    print(df_stats)
    print(f'{test_name} results:')
    print(data_df)

    plot = data_df['scores'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_V(args, agent, run_name, path_load, path_save, test_name, skip_frames, num_games, num_envs, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + i, i, args.frame_stack, False, run_name, skip=skip_frames, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_best_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        r = 0
        k = 0
        while not done:
            with torch.no_grad():
                # action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)

                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                #next_obs, reward, done, info = envs.step(int(action.cpu().numpy()))
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()
                matplotlib.image.imsave(os.path.join(path_save, f'{k}.png'), obs_np, cmap='gray')
                k += 1
        if idx == 0:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/*.png'
            image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"]
    df_stats = pd.DataFrame([stats], columns=columns)

    data_dict = {'scores': pd.Series(scores),
                 'rewards': pd.Series(rewards),
                 'lives': pd.Series(lives),
                 'lenght': pd.Series(lenght),
                 'times': pd.Series(times)}

    data_df = pd.DataFrame(data_dict)
    print(f'{test_name} stats:')
    print(df_stats)
    print(f'{test_name} results:')
    print(data_df)

    plot = data_df['scores'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_VI(args, agent, run_name, path_load, path_save, test_name, wd, num_games, num_envs, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + i, i, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_best_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        r = 0
        k = 0
        while not done:
            with torch.no_grad():
                # action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)

                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                #next_obs, reward, done, info = envs.step(int(action.cpu().numpy()))
                cx = random.randint(wd + 1, 84 - wd)
                cy = random.randint(wd + 1, 84 - wd)
                next_obs[:, :, cx:cx + wd, cy:cy + wd] = 255
                next_obs[:, :, cx - wd:cx, cy - wd:cy] = 255
                next_obs[:, :, cx - wd:cx, cy:cy + wd] = 255
                next_obs[:, :, cx:cx + wd:, cy - wd:cy] = 255

                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()
                matplotlib.image.imsave(os.path.join(path_save, f'{k}.png'), obs_np, cmap='gray')
                k += 1
        if idx == 0:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/*.png'
            image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"]
    df_stats = pd.DataFrame([stats], columns=columns)

    data_dict = {'scores': pd.Series(scores),
                 'rewards': pd.Series(rewards),
                 'lives': pd.Series(lives),
                 'lenght': pd.Series(lenght),
                 'times': pd.Series(times)}

    data_df = pd.DataFrame(data_dict)
    print(f'{test_name} stats:')
    print(df_stats)
    print(f'{test_name} results:')
    print(data_df)

    plot = data_df['scores'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plt.show()
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


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
    run_name = "BreakoutNoFrameskip-v4__cnn_brims_mlp_mlp_hibrid_reward__1__1658017692"
    checkpoint_path = os.path.join("/home/brain/alana/checkpoints/modelos_hibridos_v2", f"{run_name}_args.json")
    print(checkpoint_path)
    path_load = "/home/brain/alana/checkpoints/modelos_hibridos_v2"
    path_save = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    num_games = 1
    num_envs = 1
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

    #agent = AgentBrims(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks, args.topk,
    #           args.use_inactive, args.blocked_grad).to(device)

    agent = AgentCuriosity(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks,
                           args.topk,
                           args.use_inactive, args.blocked_grad).to(device)

    # cenário de teste I - mesmo ambiente de treinamento do agente
    stats_I = test_I(args, agent, run_name, path_load, os.path.join(path_save, "test_I"), "test_I" , num_games, num_envs, device)
    print(stats_I)
    # cenário de teste II - ambiente de teste do agente com estilo diferente
    ''' 
    stats_II = test_II(args, agent, run_name, path_load, os.path.join(path_save, "test_II"), "test_II", num_games, num_envs,
                     device)
    # cenário de teste III - ambiente de teste do agente mais difícil
    stats_III = test_III(args, agent, run_name, path_load, os.path.join(path_save, "test_III"), "test_III", 4, num_games,
                       num_envs,
                       device)
    # cenário de teste IV - ambiente de teste do agente mais difícil
    stats_IV = test_IV(args, agent, run_name, path_load, os.path.join(path_save, "test_IV"), "test_IV", 4, 1,
                         num_games,
                         num_envs,
                         device)
    # cenário de teste V - ambiente de teste do agente pulando frames - 10 frames
    stats_V = test_V(args, agent, run_name, path_load, os.path.join(path_save, "test_V"), "test_V", 10 , num_games,
                       num_envs,
                       device)
    # cenário de teste V - ambiente de teste do agente com oclusões não vistas
    stats_VI = test_VI(args, agent, run_name, path_load, os.path.join(path_save, "test_VI"), "test_VI", 10, num_games,
                       num_envs,
                       device)

    dict_data = {'test_I': stats_I, 'test_II': stats_II, 'test_III': stats_III, 'test_IV': stats_IV, 'test_V': stats_V, 'test_VI': stats_VI}
    all_results = pd.DataFrame.from_dict(dict_data, orient='index', columns=["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lives", "lenght", "times"])
    all_results.to_csv(os.path.join(path_save, 'all_results.csv'))'''

if __name__ == "__main__":
    main()