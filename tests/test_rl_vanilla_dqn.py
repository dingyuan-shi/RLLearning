import torch
from rllib.memories import NaiveReplayMemory
from rllib.agents import VanillaDQNAgent
from .test_rl_common import seed_all, NaiveQNet
import gym
import torch.nn as nn
from rllib.trainers import RawTrainer


def test_vanilla_dqn():
    env = gym.make("CartPole-v0")
    env.seed(0)
    gamma = 0.9
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n
    agent = VanillaDQNAgent(gamma, state_dim, n_action, device,
                            NaiveQNet(state_dim, n_action), (0.5, 0.01), lr=0.001, target_update_freq=20, finite=True)
    trainer = RawTrainer(env, agent, NaiveReplayMemory(buffer_size=5000), episode=500,
                         batch_size=32, learn_start=100, learn_freq=1, render=False, verbose=1)
    trainer.train()
    trainer.evaluation()


if __name__ == "__main__":
    seed_all()
    test_vanilla_dqn()
