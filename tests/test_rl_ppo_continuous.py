import gym
from rllib.agents import PPOContinuousAgent
from rllib.memories import NaiveMemory
from rllib.trainers import RawTrainer
from .test_rl_common import seed_all, PendulumEnvWrapper, ContinuousPolicy, Critic
import torch



def test_ppo_continuous():

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    env = PendulumEnvWrapper(gym.make("Pendulum-v1"))
    env.seed(20)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    gamma = 0.9
    eps = 0.2
    agent = PPOContinuousAgent(gamma, eps, 0.0001, state_dim, action_dim, device,
                               ContinuousPolicy(state_dim, action_dim, action_range),
                               Critic(state_dim), action_range, finite=False)
    trainer = RawTrainer(env, agent, NaiveMemory(), episode=1000, batch_size=32, update_time=10, render=True, verbose=1)
    trainer.train()
    trainer.evaluation(env.unwrapper())


if __name__ == "__main__":
    seed_all()
    test_ppo_continuous()
