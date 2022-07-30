import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from brims.blocks import Blocks
import wandb
import json
import torch.nn.functional as F

'''modificações:
- normalizar o estado com normalizacão z
- normalizar a reward
'''



from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="cnn_brims_mlp_mlp_intrinsic_reward",
        help="the name of this experiment")
    parser.add_argument("--run_name", type=str, default="BreakoutNoFrameskip-v4__cnn_brims_mlp_mlp_intrinsic_reward__1__1659217285",
                        help="experiment name")
    parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--frame_stack", type=int, default=4,
                        help="frame stack num")

    parser.add_argument("--total-timesteps", type=int, default=210000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Breakout-experiment_I",
        help="the wandb's project name")
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")


    #Brims parameters
    parser.add_argument("--nlayers", type=int, default=1, help="number of layers")
    parser.add_argument('--nhid', nargs='+', type=int, default=[128])
    parser.add_argument('--topk', nargs='+', type=int, default=[2])
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4])
    parser.add_argument("--ninp", type=int, default=128, help="embedding input")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--use_inactive", type=bool, default=True)
    parser.add_argument("--blocked_grad", type=bool, default=True)

    parser.add_argument("--norm_obs", type=bool, default=True)
    parser.add_argument("--obs_norm_step", type=int, default=10000)

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--device_num", type=int, default=0,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, frame_stack, capture_video, run_name, split='train'):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{split}_{run_name}.mp4")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if split == 'train':
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, frame_stack)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        #self.mean = torch.zeros(shape).to(device)
        #self.var = torch.ones(shape).to(device)
        self.count = epsilon


    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class AgentCuriosity(nn.Module):
    def __init__(self, framestack, ninp=512, nhid=[512], nlayers=1, dropout=0.5, num_blocks=[4], topk=[2], use_inactive=False, blocked_grad=False):
        super(AgentCuriosity, self).__init__()

        self.nhid = nhid
        self.topk = topk
        print('Top k Blocks: ', topk)
        self.drop = nn.Dropout(dropout)
        self.num_blocks = num_blocks
        self.ninp = ninp
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        self.nlayers = nlayers
        print("Dropout rate", dropout)
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, ninp)),
            nn.ReLU(),
        )

        self.brims_p = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)
        self.brims_f = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)
        self.actor = layer_init(nn.Linear(self.nhid[-1], 4), std=0.01)
        self.critic = layer_init(nn.Linear(self.nhid[-1], 1), std=1)

    def init_hidden_p(self, bsz):
        hx, cx = [],[]
        weight = next(self.brims_p.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))
        return (hx,cx)

    def init_hidden_f(self, bsz):
        hx, cx = [],[]
        weight = next(self.brims_f.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))
        return (hx,cx)

    def brims_p_blockify_params(self):
        self.brims_p.blockify_params()

    def brims_f_blockify_params(self):
        self.brims_f.blockify_params()

    def get_states(self, x, lstm_state):
        #embs = self.encoder(x / 255.0)
        embs = self.encoder(x)
        batch_size = lstm_state[0][0].shape[0]
        input_size = lstm_state[0][0].shape[1]
        embs = embs.reshape((-1, batch_size, input_size))
        new_hidden = []
        self.brims_p_blockify_params()
        #print(f'embs (fora do for): {embs.shape}')
        for emb in embs:
            #print(f'emb (dentro do for): {emb.shape}')
            lstm_state = self.brims_p(emb, lstm_state)
            #new_hidden += [lstm_state[0][-1]]

            new_hidden.append(lstm_state[0][-1])
        #print(f'saiu da chamada da lstm new hidden shape: {len(new_hidden)}')
        #new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        new_hidden = torch.stack(new_hidden)
        new_hidden = new_hidden.view(embs.shape[0]*batch_size, self.nhid[-1])
        #print(f'após a chamada da lstm new hidden shape: {new_hidden.shape}')
        #exit()
        return embs, new_hidden, lstm_state

    def get_value(self, x, lstm_state):
        _, hidden, lstm_state = self.get_states(x, lstm_state)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, action=None):
        embs, hidden, lstm_state = self.get_states(x, lstm_state)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state, embs, hidden

    def compute_intrinsic_reward(self, enc_ts, actual_brims_p, exp_hidden_f, lstm_state):
        batch_size = lstm_state[0][0].shape[0]
        input_size = lstm_state[0][0].shape[1]
        enc_ts = enc_ts.reshape((-1, batch_size, input_size))
        new_hidden_f = []
        self.brims_f_blockify_params()
        # print(f'embs (fora do for): {embs.shape}')
        for enc_t in enc_ts:
            # print(f'emb (dentro do for): {emb.shape}')
            lstm_state = self.brims_f(enc_t, lstm_state)
            # new_hidden += [lstm_state[0][-1]]

            new_hidden_f.append(lstm_state[0][-1])
        # print(f'saiu da chamada da lstm new hidden shape: {len(new_hidden)}')
        # new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        new_hidden_f = torch.stack(new_hidden_f)
        new_hidden_f = new_hidden_f.view(enc_ts.shape[0] * batch_size, self.nhid[-1])
        # print(f'após a chamada da lstm new hidden shape: {new_hidden.shape}')
        # exit()
        eta = 0.01
        intrinsic_reward = eta*F.mse_loss(actual_brims_p, exp_hidden_f, reduction='none').mean(-1)
        return intrinsic_reward.detach().data.cpu().numpy(), lstm_state, new_hidden_f


if __name__ == "__main__":
    args = parse_args()
    if args.run_name is None:
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.run_name

    print(f'RUN NAME: {run_name}')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device_num >= 0:
        torch.cuda.set_device(args.device_num)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print("CUDA is not used")
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.frame_stack, args.capture_video, run_name) for i in
         range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = AgentCuriosity(args.frame_stack, args.ninp, args.nhid, args.nlayers, args.dropout, args.num_blocks, args.topk,
                       args.use_inactive, args.blocked_grad).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    json.dump(vars(args), open(os.path.join(checkpoint_path, f"{run_name}_args.json"), 'w'))

    run = wandb.init(project=args.wandb_project_name,
                     entity=args.wandb_entity,
                     config=vars(args),
                     name=run_name,
                     monitor_gym=True,
                     save_code=True,
                     id=run_name,
                     resume=True)

    if os.path.exists(os.path.join(checkpoint_path, f"{run_name}_model.pth")):
        print(f'loading model ... {run_name}')
        wandb.restore(os.path.join(checkpoint_path, f"{run_name}_model.pth"))
        checkpoint = torch.load(os.path.join(checkpoint_path, f"{run_name}_model.pth"))
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        update_init = checkpoint['update']
        global_step = checkpoint['global_step']
        print(f'load model OK ... update_init {update_init} | global_step {global_step}')
    else:
        print(f'new model ... {run_name}')
        update_init = 1
        global_step = 0


    print(f'new model ... {run_name}')
    update_init = 1
    global_step = 0

    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print("Model Built with Total Number of Trainable Parameters: " + str(total_params))

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    #rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)


    next_lstm_state_p = agent.init_hidden_p(args.num_envs)
    next_lstm_state_f = agent.init_hidden_f(args.num_envs)

    exp_hidden = torch.zeros((args.num_envs, args.num_steps)).to(device)

    num_updates = args.total_timesteps // args.batch_size

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 4, 84, 84))
    discounted_reward = RewardForwardFilter(args.gamma)

    sample_i_rall_im = 0
    sample_i_rall_ex = 0

    if args.norm_obs:
        # normalize obs
        print('Start to initailize observation normalization parameter.....')
        next_obs_norm = []
        steps_norm = 0
        while steps_norm < args.obs_norm_step:
            steps_norm += args.num_envs
            actions_rand = np.random.randint(0, 3, size=(args.num_envs,))
            s, _ , _ , _ = envs.step(actions_rand)
            next_obs_norm.append(s)

        next_obs_norm = np.stack(next_obs_norm)
        next_obs_norm = np.reshape(next_obs_norm, (args.obs_norm_step, next_obs_norm.shape[2], next_obs_norm.shape[3], next_obs_norm.shape[4]))
        obs_rms.update(next_obs_norm)

        #print('End to initalize...')
        #print(f'obs_mean: {obs_rms.mean.shape} obs_var: {obs_rms.var.shape}')

        #print(np.unique(obs_rms.mean))
        #print(np.unique(obs_rms.var))

    for update in range(update_init, num_updates + 1):
        initial_lstm_state_p = (next_lstm_state_p[0], next_lstm_state_p[1])
        initial_lstm_state_f = (next_lstm_state_f[0], next_lstm_state_f[1])
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow


            #agent.brims_blockify_params()
        total_int_reward = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                #next_obs = next_obs.reshape(1, 8, 1, 84, 84)
                action, logprob, _, value, next_lstm_state_p, embs_t, actual_hidden = agent.get_action_and_value(((next_obs - torch.Tensor(obs_rms.mean).to(device)) / torch.Tensor(np.sqrt(obs_rms.var)).to(device)), next_lstm_state_p)
                #hidden = repackage_hidden(hidden)

                im_reward, next_lstm_state_f, exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden, exp_hidden, next_lstm_state_f)


                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob


            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, ex_reward, done, info = envs.step(action.cpu().numpy())
            #ex_reward = torch.tensor(ex_reward).to(device).view(-1)
            #print(f'ex_reward: {ex_reward}')
            #total_reward = im_reward
            #print(f'reward externa: {ex_reward} shape: {ex_reward.shape}')
            #print(f'reward interna: {im_reward} shape: {im_reward.shape}')
            #rewards[step] = im_reward
            total_int_reward.append(im_reward)
            sample_i_rall_im += im_reward[0].item()
            sample_i_rall_ex += ex_reward[0]

            #print(f'im_reward: {im_reward} -- reward/step: {rewards[step]}')
            #exit()
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            #exit()

            #print(type(info))
            #print(len(info))
            #exit()
            for id, item in enumerate(info):
                if "episode" in item.keys():
                    print(f"update={update}/total_updates={num_updates}, global_step={global_step}, episodic_return={item['episode']['r']}")
                    avg_returns.append(item['episode']['r'])
                    wandb.log({
                    "charts/average_20_last_score_episodes": np.average(avg_returns),
                    "charts/episodic_return": item["episode"]["r"],
                    "charts/sample_i_rall - intrinsic": sample_i_rall_im,
                    "charts/sample_i_rall - score game (extrinsic)": sample_i_rall_ex,
                    "charts/episodic_length": item["episode"]["l"]
                    }, step=global_step)

                    sample_i_rall_im = 0
                    sample_i_rall_ex = 0
                    break

        # bootstrap value if not done
        with torch.no_grad():

            next_value = agent.get_value(((next_obs - torch.Tensor(obs_rms.mean).to(device)) / torch.Tensor(np.sqrt(obs_rms.var)).to(device)), next_lstm_state_p).reshape(1, -1)

            #normalizar a reward intrinseca
            #for it in range(args.num_steps):
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                             total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)

            rewards = torch.from_numpy(total_int_reward.T).to(device)






            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        mean_advantages = b_advantages.mean()

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        #print(flatinds.shape)
        clipfracs = []

        intrinsic_mse = nn.MSELoss()
        #print(f'COMEÇOU A TREINARRRRR')
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                hx_mb_p = []
                cx_mb_p = []
                [hx_mb_p.append(initial_lstm_state_p[0][i][mbenvinds]) for i in range(args.nlayers)]
                [cx_mb_p.append(initial_lstm_state_p[1][i][mbenvinds]) for i in range(args.nlayers)]

                _, newlogprob, entropy, newvalue, _ , embs, h_current = agent.get_action_and_value(
                    ((b_obs[mb_inds] - torch.Tensor(obs_rms.mean).to(device)) / torch.Tensor(np.sqrt(obs_rms.var)).to(device)),
                    (hx_mb_p, cx_mb_p),
                    b_actions.long()[mb_inds],
                )

                hx_mb_f = []
                cx_mb_f = []
                [hx_mb_f.append(initial_lstm_state_f[0][i][mbenvinds]) for i in range(args.nlayers)]
                [cx_mb_f.append(initial_lstm_state_f[1][i][mbenvinds]) for i in range(args.nlayers)]

                _, _ , exp_hidden = agent.compute_intrinsic_reward(embs_t, actual_hidden, exp_hidden, (hx_mb_f, cx_mb_f))



                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                loss_intrinsic = intrinsic_mse(actual_hidden.detach(), exp_hidden.detach())
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef) + loss_intrinsic


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "losses/intrinsic_loss": loss_intrinsic.item(),
            "charts/SPS": int(global_step / (time.time() - start_time))
        }, step=global_step)


        print("SPS:", int(global_step / (time.time() - start_time)))
        torch.save({'update': update,
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()}, os.path.join(checkpoint_path, f"{run_name}_model.pth"))
        wandb.save(os.path.join(checkpoint_path, f"{run_name}_model.pth"))

    envs.close()