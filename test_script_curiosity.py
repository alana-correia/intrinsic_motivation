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
import glob
import sys

from intrinsic_baseline_cvpr_all import AgentCuriosity
from intrinsic_baseline_cvpr_all import make_env as make_env_curiosity
from intrinsic_baseline_cvpr_all import parse_args as parse_args_curiosity


import os
#import moviepy.video.io.ImageSequenceClip

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
            [make_env_curiosity(args.gym_id, args.seed, args.frame_stack, args.full_space_actions, mode=args.mode) for idx in
             range(num_envs)]
        )



        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'], map_location = torch.device(device))
        print('Loading Agent ...')
        agent.eval()
        done = False
        print(type(envs.reset()))
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]

                #total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()), end='')
                next_obs = torch.Tensor(next_obs).to(device)
                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                #if idx == 0:
                #    path_img = os.path.join(path_save,f'imgs_video_{idx}')
                #    if not os.path.exists(path_img):
                #        os.makedirs(path_img)
                #    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                #    k += 1

        envs.close()
        #if idx == 0:
        #    fps = 10
        #    print('\nstart video ...')
        #    path_video = f'/home/brain/alana/checkpoints/videos_and_results/{run_name}/{test_name}/imgs_video_{idx}/*.png'
        #    image_files = sorted(glob.glob(path_video), key=lambda x: int(os.path.basename(x).split('/')[-1].split('.')[0]))
        #    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        #    clip.write_videofile(os.path.join(path_save, f"{test_name}_{run_name}_game{idx}.mp4"))

        scores.append(info[-1]['episode']['r'])
        #scores.append(r)
        rewards.append(r)
        #lives.append(info[-1]['lives'])
        lenght.append(info[-1]['episode']['l'])
        times.append(info[-1]['episode']['t'])

    stats = [np.mean(scores), np.std(scores), np.mean(rewards), np.std(rewards), np.mean(lenght), np.mean(times)]
    columns = ["scores_mean", "scores_std", "rewards_mean", "rewards_std", "lenght", "times"]
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

    df_stats.to_csv(os.path.join(path_save, f'{test_name}_stats.csv'), index=False)
    data_df.to_csv(os.path.join(path_save, f'{test_name}_results.csv'), index=False)

    return stats


def test_II(args, agent, run_name, path_load, path_save, test_name, num_games, num_envs, type_model, device, pertub):
    scores = []
    lives = []
    lenght = []
    times = []
    rewards = []


    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx in range(num_games):

        envs = gym.vector.SyncVectorEnv(
            [make_env_curiosity(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []


        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]

                total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = next_obs - pertub
                next_obs = np.clip(next_obs, a_min=0, a_max=255)
                next_obs = torch.Tensor(next_obs).to(device)

                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        envs.close()
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
            [make_env_curiosity(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, mode=mode, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)



        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []


        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]

                total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)

                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        envs.close()
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
            [make_env_curiosity(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, mode=mode, difficulty=difficulty, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)



        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []


        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]

                total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)

                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        envs.close()
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
            [make_env_curiosity(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, skip=skip_frames, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)



        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []

        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]

                total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)

                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()

                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        envs.close()
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
            [make_env_curiosity(args.gym_id, args.seed + idx, idx, args.frame_stack, False, run_name, split='test') for i in
             range(num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        checkpoint = torch.load(os.path.join(path_load, f"{run_name}_{type_model}.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        done = False
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(num_envs).to(device)

        next_lstm_state_p = agent.init_hidden_p(1)
        next_lstm_state_f = agent.init_hidden_f(1)

        exp_hidden = torch.zeros((1, 128)).to(device)

        r = 0
        k = 0
        dones_game = []
        actions_game = []
        lives_game = []
        ex_reward_game = []


        while not done:
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(
                    next_obs, next_lstm_state_p)
                # hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden,
                                                                                          exp_hidden, next_lstm_state_f)

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                done = done[-1]
                r += reward[-1]

                total_reward = args.em_weight * reward[-1] + args.im_weight * im_reward.data.cpu().numpy()[-1]
                print('\rgame {} - reward (sum) {} - done {} - action {}'.format(idx, r, done, action.cpu().numpy()),
                      end='')
                next_obs = torch.Tensor(next_obs).to(device)

                obs_np = next_obs[0, 3, :, :].data.cpu().numpy()


                if idx == 0 or idx == 1:
                    path_img = os.path.join(path_save, f'imgs_video_{idx}')
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    matplotlib.image.imsave(os.path.join(path_img, f'{k}.png'), obs_np, cmap='gray')
                    k += 1
        envs.close()
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
    #test_name = "test_I"
    return args, test_name

if __name__ == "__main__":
    args = parse_args_curiosity()
    #run_name = "MsPacman-v4__my_baseline_intrinsic__1__1664396744"
    checkpoint_path = os.path.join("checkpoints", f"{args.run_name}_args.json")
    print(checkpoint_path)
    path_load = "checkpoints/"
    path_save = f'checkpoints/videos_and_results/{args.run_name}'
    type_model = args.type_model



    if not os.path.exists(path_save):
        os.makedirs(path_save)

    num_games = 10
    num_envs = 1

    #f = open(checkpoint_path, "r")
    #kwargs = json.loads(f.read())

    #kwargs = json.loads(checkpoint_path)
    #with open(checkpoint_path, "r") as read_file:
    #    data = json.load(read_file)

    #args, test_name = function_with_args_and_default_kwargs(sys.argv, **kwargs)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.cuda.set_device(args.device_num)


    #agent = AgentCuriosity(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks, args.topk,
    #           args.use_inactive, args.blocked_grad).to(device)

    agent = AgentCuriosity(args.num_actions, args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout,
                           args.num_blocks,
                           args.topk,
                           args.use_inactive, args.blocked_grad).to(device)

    if args.test_name == "testI":
        #mesmo ambiente de treinamento
        test_I(args, agent, args.run_name, path_load, os.path.join(path_save, "test_I"), "test_I" , num_games, num_envs, type_model, device)
    ''' 
    elif args.test_name == "testII":
        #pequenas mudanças de dinamica do ambiente - mudança de nível do jogo
        #change_(args, agent, args.run_name, path_load, os.path.join(path_save, "test_II_v1"), "test_II_v1", num_games, num_envs, type_model,
        #             device, 5.0)
        #change_difficulty(args, agent, args.mode, args.run_name, path_load, os.path.join(path_save, "test_III"), "test_III", 4, num_games, num_envs, type_model, device)
    elif args.test_name == "testIII":
    # cenário de teste III - ambiente de teste do agente mais difícil
        test_III(args, agent, args.run_name, path_load, os.path.join(path_save, "test_III"), "test_III", 4, num_games,
                       num_envs, type_model,
                       device)
    elif args.test_name == "testIV":
    # cenário de teste IV - ambiente de teste do agente mais difícil
        test_IV(args, agent, args.run_name, path_load, os.path.join(path_save, "test_IV"), "test_IV", 4, 1,
                         num_games,
                         num_envs, type_model,
                         device)
    elif args.test_name == "testV":
    # cenário de teste V - ambiente de teste do agente pulando frames - 10 frames
        test_V(args, agent, args.run_name, path_load, os.path.join(path_save, "test_V"), "test_V", 10 , num_games,
                       num_envs, type_model,
                       device)
    else:
    # cenário de teste V - ambiente de teste do agente com oclusões não vistas
        test_VI(args, agent, args.run_name, path_load, os.path.join(path_save, "test_VI"), "test_VI", 2, num_games,
                       num_envs, type_model,
                       device)'''





