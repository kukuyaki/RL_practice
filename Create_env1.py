'''
練習時可以用gymnasium給的環境
但當我們想作專題研究時，多半會需要根據需求創建自己的環境

gymnasium有提供創建新環境的方法
https://gymnasium.farama.org/introduction/create_custom_env/


重點函式有四個
1.init
2.reset
3.step
4.render

只要能寫出這四個就可以用了

讓我們延伸CartPole v1 ，製作一個輸出包括動力大小的連續數值output
'''

import gymnasium as env_gym
from gymnasium import spaces
import numpy as np

class CatchEnvironment(env_gym.Env):
    # 告訴 Gymnasium 支援的渲染模式
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CatchEnvironment, self).__init__()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.agent_pos = 5.0
        self.target_pos = 3.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 5.0
        self.target_pos = float(np.random.randint(0, 11))
        
        state = np.array([self.agent_pos, self.target_pos], dtype=np.float32)
        return state, {}

    def step(self, action):
        if action == 0:  # 向左
            self.agent_pos = max(0.0, self.agent_pos - 1.0)
        elif action == 2:  # 向右
            self.agent_pos = min(10.0, self.agent_pos + 1.0)
        terminated = False
        reward = 0.0

        if abs(self.agent_pos - self.target_pos) < 0.1:
            reward = 10.0
            terminated = True
        else:
            reward = -0.1  # 沒抓到之前，每走一步扣一點點分數，逼它快點找到目標

        truncated = False
        state = np.array([self.agent_pos, self.target_pos], dtype=np.float32)

        return state, reward, terminated, truncated, {}

    def render(self):
        print(f"Agent 位置: {self.agent_pos:.1f} | 目標位置: {self.target_pos:.1f}")