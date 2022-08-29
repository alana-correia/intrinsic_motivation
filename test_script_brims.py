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
from ppo_atari_brims import AgentBrims
from ppo_atari_brims import make_env as make_env_brims
from collections import defaultdict
from scipy.stats import norm

import os
import moviepy.video.io.ImageSequenceClip

def test_I(args, agent, run_name, path_load, path_save, test_name, num_games, num_envs, type_model, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_brims(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state = agent.init_hidden(num_envs)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save,f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        if idx == 0:
            fig_att = plt.gcf()
            counts, bins = np.histogram(actions_game)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'histogram_actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(actions_game, 'b')
            plt.legend(["Actions"], loc="best")
            plt.xlabel('Step')

            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            actions_game = np.asarray(actions_game)
            actions_game = np.where(actions_game == 1, 1, 0)
            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')

            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_versus_lives_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(ex_reward_game, 'b')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'rewards_game0.pdf'), format='pdf')
            plt.close()

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}_game{idx}.mp4"))

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
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')
    plt.close()

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')
    plt.close()

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_II(args, agent, run_name, path_load, path_save, test_name, num_games, num_envs, type_model, device, pertub):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    dones_game = []
    actions_game = []
    lives_game = []
    ex_reward_game = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_brims(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state = agent.init_hidden(num_envs)

        r = 0
        k = 0



        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())

                next_obs = next_obs - pertub
                next_obs = np.clip(next_obs, a_min=0, a_max=255)

                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        if idx == 0:
            fig_att = plt.gcf()
            counts, bins = np.histogram(actions_game)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'histogram_actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(actions_game, 'b')
            plt.legend(["Actions"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            actions_game = np.asarray(actions_game)
            actions_game = np.where(actions_game == 1, 1, 0)
            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_versus_lives_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(ex_reward_game, 'b')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'rewards_game0.pdf'), format='pdf')
            plt.close()

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video),
                                 key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}_game{idx}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght),
             np.mean(times)]
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
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')
    plt.close()

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')
    plt.close()

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats



def test_III(args, agent, run_name, path_load, path_save, test_name, mode, num_games, num_envs, type_model, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_brims(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, mode=mode, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state = agent.init_hidden(num_envs)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        if idx == 0:
            fig_att = plt.gcf()
            counts, bins = np.histogram(actions_game)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'histogram_actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(actions_game, 'b')
            plt.legend(["Actions"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            actions_game = np.asarray(actions_game)
            actions_game = np.where(actions_game == 1, 1, 0)
            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_versus_lives_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(ex_reward_game, 'b')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'rewards_game0.pdf'), format='pdf')
            plt.close()

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video),
                                 key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save,  f"{test_name}_{run_name}_game{idx}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght),
             np.mean(times)]
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
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')
    plt.close()

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')
    plt.close()

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats

def test_IV(args, agent, run_name, path_load, path_save, test_name, mode, difficulty, num_games, num_envs, type_model, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_brims(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, mode=mode, difficulty=difficulty, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state = agent.init_hidden(num_envs)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        if idx == 0:
            fig_att = plt.gcf()
            counts, bins = np.histogram(actions_game)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'histogram_actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(actions_game, 'b')
            plt.legend(["Actions"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            actions_game = np.asarray(actions_game)
            actions_game = np.where(actions_game == 1, 1, 0)
            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_versus_lives_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(ex_reward_game, 'b')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'rewards_game0.pdf'), format='pdf')
            plt.close()

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video),
                                 key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save,  f"{test_name}_{run_name}_game{idx}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght),
             np.mean(times)]
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
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')
    plt.close()

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')
    plt.close()

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_V(args, agent, run_name, path_load, path_save, test_name, skip_frames, num_games, num_envs, type_model, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_brims(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, skip=skip_frames, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state = agent.init_hidden(num_envs)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        if idx == 0:
            fig_att = plt.gcf()
            counts, bins = np.histogram(actions_game)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'histogram_actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(actions_game, 'b')
            plt.legend(["Actions"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            actions_game = np.asarray(actions_game)
            actions_game = np.where(actions_game == 1, 1, 0)
            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_versus_lives_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(ex_reward_game, 'b')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'rewards_game0.pdf'), format='pdf')
            plt.close()

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video),
                                 key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save,  f"{test_name}_{run_name}_game{idx}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght),
             np.mean(times)]
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
    plt.draw()
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')
    plt.close()

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')
    plt.close()

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_VI(args, agent, run_name, path_load, path_save, test_name, wd, num_games, num_envs, type_model, device):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_brims(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state = agent.init_hidden(num_envs)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state)
                next_obs, reward, done, info = envs.step(action.cpu().numpy())

                cx = random.randint(wd + 1, 84 - wd)
                cy = random.randint(wd + 1, 84 - wd)
                next_obs[:, :, cx:cx + wd, cy:cy + wd] = 255
                next_obs[:, :, cx - wd:cx, cy - wd:cy] = 255
                next_obs[:, :, cx - wd:cx, cy:cy + wd] = 255
                next_obs[:, :, cx:cx + wd:, cy - wd:cy] = 255

                done = done[-1]
                r += reward[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0:
                    dones_game.append(done)
                    actions_game.append(action.cpu().numpy()[-1])
                    lives_game.append(info[-1]['lives'])
                    ex_reward_game.append(reward[-1])

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        if idx == 0:
            fig_att = plt.gcf()
            counts, bins = np.histogram(actions_game)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'histogram_actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(actions_game, 'b')
            plt.legend(["Actions"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            actions_game = np.asarray(actions_game)
            actions_game = np.where(actions_game == 1, 1, 0)
            plt.plot(actions_game, 'b')
            plt.plot(lives_game, 'r')
            plt.plot(dones_game, 'g')
            plt.legend(["Actions", "Lives", "Dones"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'actions_versus_lives_game0.pdf'), format='pdf')
            plt.close()

            fig_att = plt.gcf()
            plt.plot(ex_reward_game, 'b')
            plt.legend(["Extrinsic Reward", "Intrinsic Reward", "Total Reward"], loc="best")
            plt.xlabel('Step')
            plt.draw()
            fig_att.savefig(os.path.join(path_save, 'rewards_game0.pdf'), format='pdf')
            plt.close()

        if idx == 0 or idx == 1:
            fps = 10
            print('\nstart video ...')
            path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
            image_files = sorted(glob.glob(path_video),
                                 key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(os.path.join(path_save,  f"{test_name}_{run_name}_game{idx}.mp4"))

        scores.append(info[-1]['episode']['r'])
        rewards.append(r)
        lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lives), np.mean(lenght),
             np.mean(times)]
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
    plot.get_figure().savefig(os.path.join(path_save, 'scores.pdf'), format='pdf')
    plt.close()

    plot = data_df['rewards'].plot.line()
    plt.xlabel('Game')
    plt.ylabel('Reward')
    plot.get_figure().savefig(os.path.join(path_save, 'rewards.pdf'), format='pdf')
    plt.close()

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def function_with_args_and_default_kwargs(unk, **kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    #args = parser.parse_args(optional_args)
    args, unknown = parser.parse_known_args(unk)
    test_name = unknown[1]
    return args, test_name

if __name__ == "__main__":
    #args = parse_args()
    run_name = "BreakoutNoFrameskip-v4__cnn_brims_mlp_mlp_extrinsic_reward__1__1657895196"
    checkpoint_path = os.path.join("/home/brain/alana/checkpoints/final_checkpoints", f"{run_name}_args.json")
    print(checkpoint_path)
    path_load = "/home/brain/alana/checkpoints/final_checkpoints"
    path_save = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}'
    type_model = "model"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    num_games = 10
    num_envs = 1

    f = open(checkpoint_path, "r")
    kwargs = json.loads(f.read())

    args, test_name = function_with_args_and_default_kwargs(sys.argv, **kwargs)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    agent = AgentBrims(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks, args.topk,
               args.use_inactive, args.blocked_grad).to(device)

    if test_name == "test_I":
        test_I(args, agent, run_name, path_load, os.path.join(path_save, "test_I"), "test_I" , num_games, num_envs, type_model, device)
    elif test_name == "test_II_v1":
    # cenário de teste II - ambiente de teste do agente com estilo diferente
        test_II(args, agent, run_name, path_load, os.path.join(path_save, "test_II_v1"), "test_II_v1", num_games, num_envs, type_model,
                     device, 10.0)
    elif test_name == "test_II_v2":
        test_II(args, agent, run_name, path_load, os.path.join(path_save, "test_II_v2"), "test_II_v2", num_games,
                       num_envs, type_model,
                       device, 30.0)
    elif test_name == "test_II_v3":
        test_II(args, agent, run_name, path_load, os.path.join(path_save, "test_II_v3"), "test_II_v3", num_games,
                       num_envs, type_model,
                       device, 50.0)

    elif test_name == "test_III":
    # cenário de teste III - ambiente de teste do agente mais difícil
        test_III(args, agent, run_name, path_load, os.path.join(path_save, "test_III"), "test_III", 4, num_games,
                       num_envs, type_model,
                       device)
    elif test_name == "test_IV":
    # cenário de teste IV - ambiente de teste do agente mais difícil
        test_IV(args, agent, run_name, path_load, os.path.join(path_save, "test_IV"), "test_IV", 4, 1,
                         num_games,
                         num_envs, type_model,
                         device)
    elif test_name == "test_V":
    # cenário de teste V - ambiente de teste do agente pulando frames - 10 frames
        test_V(args, agent, run_name, path_load, os.path.join(path_save, "test_V"), "test_V", 10 , num_games,
                       num_envs, type_model,
                       device)
    else:
    # cenário de teste V - ambiente de teste do agente com oclusões não vistas
        test_VI(args, agent, run_name, path_load, os.path.join(path_save, "test_VI"), "test_VI", 2, num_games,
                       num_envs, type_model,
                       device)





