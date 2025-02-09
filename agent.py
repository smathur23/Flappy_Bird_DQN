import flappy_bird_gymnasium
import gymnasium
from model import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os

DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:

    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameters_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]
            # print(hyperparameters)
        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.enable_dueling = hyperparameters['enable_dueling_dqn']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):

        if is_training:
            start_time = datetime.now()
            last_graph_update = start_time

            log_msg = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_msg)
            with open(self.LOG_FILE, "w") as f:
                f.write(log_msg + '\n')

        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_ep = []
        
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling).to(device)

        if is_training:
            epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []
            
            steps=0

            best_reward = -99999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            ep_reward = 0.0

            while (not terminated and ep_reward < self.stop_on_reward):
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                ep_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    steps += 1
                
                state = new_state

            rewards_per_ep.append(ep_reward)

            if is_training:
                if ep_reward > best_reward:
                    log_msg = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {ep_reward:0.1f} ({(ep_reward - best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_msg)
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = ep_reward
                
                cur_time = datetime.now()
                if cur_time - last_graph_update > timedelta(seconds=10):
                    self.save_graph(rewards_per_ep, epsilon_history)
                    last_graph_update=cur_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps=0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        
        cur_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(cur_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_ep, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_ep))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_ep[max(0, x-99):x+1])
        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)