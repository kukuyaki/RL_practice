'''
這邊要來嘗試DQN
也就是要開始使用深度學習了！！
深度學習的概念，大家可以先去複習一下，主要就是神經網路，每一層對不同的輸入會有不同的權重和bias偏差值

神經網路可以看成是為了代替之前的Q table

Q table
    輸入        輸出              
        狀態s      路線的分數Q值
        行動a      

Q table
    輸入        輸出              
        狀態s      每個行動a的的分數Q值或機率


之前強化學習時，機器人在做的事情其實就是一直更新Q table的值
現在加入深度學習後，只是變成更新神經網路的權重和bias!
'''

import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as env_gym
import time

# ==========================================
# 1. 定義經驗回放池 (Experience Replay Buffer)
# ==========================================
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

# ==========================================
# 4. 開始執行訓練迴圈
# ==========================================
if __name__ == "__main__":
    # 建立 CartPole 環境
    env = env_gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 1000  # 訓練回合數
    sync_target_steps = 10  # 每隔幾回合更新一次 Target Network

    print("開始訓練 DQN...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. 選擇動作
            action = agent.choose_action(state)

            # 2. 跟環境互動
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 存入經驗回放池
            agent.memory.push(state, action, reward, next_state, done)

            # 4. 進行深度學習訓練
            agent.train_step()

            state = next_state
            total_reward += reward

        # 逐漸降低探索率 epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # 定期同步 Target 網路
        if episode % sync_target_steps == 0:
            agent.update_target_net()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward} | Epsilon: {agent.epsilon:.2f}")

    print("訓練完成！")
    env.close()

    # --- 第二階段：展示動畫（開啟 render_mode="human"） ---
    print("\n=== 開始展示訓練成果動畫 ===")
    # 建立一個有畫面的新環境
    eval_env = env_gym.make("CartPole-v1", render_mode="human")
    
    for test_ep in range(3):  # 示範玩 3 次
        state, _ = eval_env.reset()
        done = False
        total_test_reward = 0
        
        print(f"正在播放第 {test_ep + 1} 次展示...")
        while not done:
            # 測試時完全不用隨機探索 (evaluate=True)，展現最強實力
            action = agent.choose_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_test_reward += reward
            
            # 稍微停頓一下讓動畫看得更清楚 (如果電腦跑太快)
            time.sleep(0.01)

        print(f"第 {test_ep + 1} 次展示結束，獲得分數: {total_test_reward}")

    eval_env.close()
    print("所有展示結束！")