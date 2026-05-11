import random
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse


# ================= DEVICE SELECTION =================
# Check which hardware is available (Apple GPU / CUDA / CPU)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Folder to store trained models & logs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent():
    def __init__(self, param_set):
        self.param_set = param_set

        # ================= LOAD HYPERPARAMETERS =================
        # Load YAML file and pick parameter set (like flappybirdv0)
        with open("parameters.yaml", 'r') as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        # Exploration parameters (epsilon-greedy)
        self.epsilon_init = params['epsilon_init']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']

        # Replay memory settings
        self.replay_memory_size = params['replay_memory_size']
        self.mini_batch_size = params['mini_batch_size']

        # Target network update frequency
        self.network_sync_rate = params['netwok_sync_rate']

        # Learning parameters
        self.alpha = params['alpha']   # learning rate
        self.gamma = params['gamma']   # discount factor

        # Stop episode if reward exceeds this
        self.reward_threshold = params['reward_threshold']

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Optimizer will be initialized later
        self.optimizer = None

        # File paths
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training=True, render=False):

        # ================= CREATE ENVIRONMENT =================
        # Flappy Bird Gym environment
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        # Get state and action dimensions
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Create main DQN (policy network)
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            # Experience replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Initial exploration probability
            epsilon = self.epsilon_init

            # Target network (used for stable training)
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Step counter (for syncing target network)
            steps = 0

            # Optimizer (Adam)
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = float('-inf')

        else:
            # Load trained model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # ================= TRAINING LOOP =================
        for episode in itertools.count():

            # Reset environment
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            # Run one episode
            while not terminated and episode_reward < self.reward_threshold:

                # ================= ACTION SELECTION =================
                # Epsilon-greedy strategy
                if is_training and random.random() < epsilon:
                    # Random action (exploration)
                    action = torch.tensor(env.action_space.sample(), dtype=torch.long, device=device)
                else:
                    # Best action from model (exploitation)
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                # ================= ENVIRONMENT STEP =================
                # Take action and observe result
                next_state, reward, terminated, _, _ = env.step(action.item())

                # Add reward
                episode_reward += reward

                # Convert to tensor
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Store experience in replay memory
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                # Move to next state
                state = next_state

            print(f"Episode {episode+1} | Reward: {episode_reward} | Epsilon: {epsilon:.4f}")

            if is_training:
                # ================= EPSILON DECAY =================
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

                # Save best model
                if episode_reward > best_reward:
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(f"Best reward: {episode_reward} at episode {episode+1}\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

            # ================= TRAIN MODEL =================
            if is_training and len(memory) >= self.mini_batch_size:

                # Sample batch from memory
                mini_batch = memory.sample(self.mini_batch_size)

                # Train step
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Sync target network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0


    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Unpack batch
        states, actions, next_states, rewards, terminateds = zip(*mini_batch)

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)

        # Convert termination flags (True/False → 1/0)
        terminations = torch.tensor(terminateds, dtype=torch.float, device=device)

        # ================= TARGET Q VALUE =================
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(1)[0]

        # ================= CURRENT Q VALUE =================
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # ================= LOSS =================
        loss = self.loss_fn(current_q, target_q)

        # ================= BACKPROP =================
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ================= ENTRY POINT =================
if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Train or test DQN model")

    # Required: parameter set name (from YAML)
    parser.add_argument('hyperparameters', help='Parameter set name (e.g., flappybirdv0)')

    # Optional flag: training mode
    parser.add_argument('--train', help='Training model', action='store_true')

    args = parser.parse_args()

    # Create agent
    dql = Agent(param_set=args.hyperparameters)

    # Run training or testing
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)