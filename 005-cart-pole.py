import gym
import numpy as np

"""
Basic Implementation
"""
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print "Episode is finished after {} timesteps"
#             break

"""
Hill Climbing: initialize weights randomly, use memory to save good weights
"""
def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        # init random weights
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')

    episodes_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 - 1
    bestreward = 0

    for _ in range(2000):
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = run_episode(env, newparams)
        print "reward %d best %d" % (reward, bestreward)
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == 200:
                break

r = train(submit=False)
print r
