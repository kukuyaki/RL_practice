'''
先來看一下常用的強化學習函式庫
https://stable-baselines3.readthedocs.io/en/master/
https://gymnasium.farama.org/
https://pettingzoo.farama.org/
https://isaac-sim.github.io/IsaacLab/main/index.html


stable_baselines3
特點： 基於 PyTorch，易用性極高，程式碼結構清晰。它是 RL Baselines 的現代標準，非常適合新手入門以及快速建立基準模型。
gymnasium
特點： 這是 OpenAI Gym 的維護版本，也是目前的業界標準介面。幾乎所有的 RL 函式庫都支援 Gymnasium 的 API 格式。

PettingZoo
特點： 針對「多代理人強化學習」（Multi-Agent RL, MARL）開發的環境庫，API 設計仿照 Gymnasium。
Isaac Lab (NVIDIA)
特點： 基於 NVIDIA Isaac Sim，利用 GPU 加速物理模擬。如果你在做機器人相關的 RL，這目前是效能的天花板。

pandas
numpy
matplotlib
seaborn
pygame

'''


import gymnasium as gym
from stable_baselines3 import A2C, PPO

env = gym.make("CartPole-v0", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()


'''
雖然這些函式庫很好用
但為了有良好的基礎
我們先拋棄這些現成工具
從0開始打造最簡單的強化學習Q learning ，加強對強化學習的理解

Q1.py 我們將實作Q learning
'''