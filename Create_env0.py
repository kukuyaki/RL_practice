'''
這是gymnasium的CartPoleEnv環境的程式碼
Create_env1.py將教學如何自創環境


這個範例我們使用CartPole-v1環境
    State — 遊戲輸出
        車子的位置 — 值域：-4.8 ～ 4.8
        車子的加速度 — 值域：-Inf ~ Inf
        柱子對車子的角度 — 值域：-24 deg~ 24 deg
        柱子倒下的加速度 — 值域：-Inf ~ Inf
    Action— 遊戲輸入
        0：向左推動
        1：向右推動
    Rewards — 每個 step 評分方式
        結束：0 分
        活著：1 分
    終止條件
        柱子對車子的角度 > 12 deg
        柱子對車子的角度 < -12 deg
        車子位置 > 2.4
        車子位置 < -2.4
        執行 500 次
'''

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils

class CartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None):
        # 1. 物理常數設定
        self.gravity = 9.8          # 重力加速度
        self.masscart = 1.0         # 小車質量
        self.masspole = 0.1         # 桿子質量
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5           # 桿子長度的一半 (Half-length)
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0       # 施加力量的大小 (固定值 10 牛頓)
        self.tau = 0.02             # 每一個時間步 (Time step) 的時間間隔 (秒)
        self.kinematics_integrator = 'euler' # 物理積分方法 (歐拉法)

        # 2. 邊界條件設定 (超出即遊戲結束)
        # 桿子傾斜角度限制：大於 ±24度 (0.418 rad) 就一定結束，但通常設定 ±12度 就判定失敗
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # 小車位置限制：超出 ±2.4 軌道邊界就算失敗
        self.x_threshold = 2.4

        # 3. 定義動作與觀察空間
        # 動作空間：0 (向左推), 1 (向右推)
        self.action_space = spaces.Discrete(2)
        
        # 觀察空間：[車子位置, 車子速度, 桿子角度, 桿子角速度]
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.state = None
        self.steps_beyond_terminated = None

    def step(self, action):
        # 確保傳進來的 action 是合法的 (0 或 1)
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert self.state is not None, "Call reset() before using step() method."
        
        x, x_dot, theta, theta_dot = self.state

        # 根據動作決定力量方向：0 往左 (-10), 1 往右 (+10)
        force = self.force_mag if action == 1 else -10.0
        
        # --- 以下為核心物理公式 (牛頓力學與運動方程式) ---
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # 暫時變數計算
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sin_theta
        ) / self.total_mass
        
        # 計算角加速度 (thetaacc) 與 加速度 (xacc)
        thetaacc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * cos_theta / self.total_mass

        # 使用歐拉法 (Euler's method) 更新下一幀的速度與位置
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # 更新環境狀態
        self.state = (x, x_dot, theta, theta_dot)

        # 4. 判定遊戲是否結束 (Termination 條件)
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # 5. 給予獎勵 (Reward)
        if not terminated:
            reward = 1.0  # 只要還活著，每一步給 +1 分
        elif self.steps_beyond_terminated is None:
            # 剛好倒下的那一步
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            # 如果已經倒了卻還繼續呼叫 step，就不給獎勵
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        # 回傳 Gymnasium 標準的 5 個回傳值
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # 重置時，給予一個很小的隨機初始狀態 (-0.05 到 0.05 之間)
        low, high = -0.05, 0.05
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), {}