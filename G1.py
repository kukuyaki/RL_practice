import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as env_gym
import time
from gymnasium.envs.registration import register

register(
    id="game_YO-v0",
    entry_point="G0:Game_A",  # 請確保 Create_env1.py 內有 CartPoleEnv_YO 類別
    max_episode_steps=500,
)

class ReplayBuffer:
    def __init__(self, capacity):
        # 使用 deque 實作固定大小的佇列，滿了會自動從頭刪除舊資料
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 隨機抽樣 batch 數量的經驗
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. 定義 DQN 神經網路 (Q-Network)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 簡單的全連接網路：輸入狀態，輸出各個動作的 Q 值
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ==========================================
# 3. DQN 演算法主體
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 參數設定
        self.gamma = 0.99          # 折扣因子 (Discount factor)
        self.lr = 0.001            # 學習率
        self.batch_size = 64       # 批次大小
        self.epsilon = 1.0         # 探索率 (Epsilon-greedy)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 建立兩個網路：Q網路 與 穩定訓練用的 Target網路
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.update_target_net()   # 初始化時讓兩者權重一樣

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(10000)

    def update_target_net(self):
        # 將 Q 網路的權重複製給 Target 網路
        self.target_net.load_state_dict(self.q_net.state_dict())

    def choose_action(self, state,evaluate=False):
        # Epsilon-greedy 策略：有機率亂走（探索），其餘時間選 Q 值最高的路（利用）
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  # 資料還夠多時先不訓練

        # 從記憶體中隨機抽樣一個 batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 轉成 PyTorch Tensor
        state_t = torch.FloatTensor(states)
        action_t = torch.LongTensor(actions)
        reward_t = torch.FloatTensor(rewards)
        next_state_t = torch.FloatTensor(next_states)
        done_t = torch.FloatTensor(dones)

        # 1. 計算目前的 Q 值: Q(s, a)
        q_values = self.q_net(state_t)
        state_action_values = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # 2. 計算目標 Q 值 (Target Q)：利用 Target 網路預測下一步的最大 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_t).max(1)[0]
            # 貝爾曼方程式: Reward + Gamma * max(Q(s', a')) * (1 - done)
            target_values = reward_t + self.gamma * next_q_values * (1 - done_t)

        # 3. 計算 Loss（MSE 均方誤差），這就是深度學習的核心！
        loss = nn.MSELoss()(state_action_values, target_values)

        # 4. 倒傳遞更新神經網路權重 (W 和 b)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == "__main__":
    env = env_gym.make("game_YO-v0")
    state_dim = 12*12
    
    # 13種輸入
    num_actions = 13
    agent = DQNAgent(state_dim, num_actions=num_actions)
    
    num_episodes = 1500  
    sync_target_steps = 10  

    print("開始訓練 DQN (連續力道離散化版)...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. 選擇動作索引 (0 ~ 20)
            action_idx = agent.choose_action(state)

            # 2. 將索引轉換成連續力道陣列送給自定義環境
            action_env = agent.get_action(action_idx)
            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated

            # 3. 經驗回放池記錄的是「動作索引 (action_idx)」
            agent.memory.push(state, action_idx, reward, next_state, done)

            # 4. 進行神經網路訓練
            agent.train_step()

            state = next_state
            total_reward += reward

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % sync_target_steps == 0:
            agent.update_target_net()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward} | Epsilon: {agent.epsilon:.2f}")

    print("訓練完成！")
    env.close()

