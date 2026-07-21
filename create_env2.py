import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as env_gym
import time
from gymnasium.envs.registration import register
'''
用DQN去嘗試離散解法
效果不彰
'''
# ==========================================
# 0. 註冊自定義環境
# ==========================================
register(
    id="gyenv/ContinuousCartPole_YO-v0",
    entry_point="Create_env1:CartPoleEnv_YO",  # 請確保 Create_env1.py 內有 CartPoleEnv_YO 類別
    max_episode_steps=500,
)

# ==========================================
# 1. 定義經驗回放池 (Experience Replay Buffer)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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
        # 輸出 21 個節點，分別代表 21 種不同的連續力道選擇
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
# 3. DQN 演算法主體（結合動作離散化）
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, num_actions=41):
        self.state_dim = state_dim
        self.num_actions = num_actions  # 總共有 21 種力量選項 (0 ~ 20)
        
        # 建立 -10.0 到 10.0 的 21 種力道對照表
        self.force_table = np.linspace(-200.0, 200.0, num_actions)

        # 參數設定
        self.gamma = 0.99          
        self.lr = 0.001            
        self.batch_size = 64       
        self.epsilon = 1.0         
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 網路輸入維度是 state_dim，輸出維度是 21 (num_actions)
        self.q_net = QNetwork(state_dim, num_actions)
        self.target_net = QNetwork(state_dim, num_actions)
        self.update_target_net()   

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(10000)

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def choose_action(self, state, evaluate=False):
        # 隨機探索時，隨機選 0~20 之間的整數索引
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()  # 回傳 Q 值最高的神經元索引 (0~20)

    def get_force_value(self, action_idx):
        # 將 0~20 的整數索引對應轉成環境要的連續力道陣列 [force_val]
        force_val = self.force_table[action_idx]
        return np.array([force_val], dtype=np.float32)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_t = torch.FloatTensor(states)
        action_t = torch.LongTensor(actions)
        reward_t = torch.FloatTensor(rewards)
        next_state_t = torch.FloatTensor(next_states)
        done_t = torch.FloatTensor(dones)

        q_values = self.q_net(state_t)
        state_action_values = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_t).max(1)[0]
            target_values = reward_t + self.gamma * next_q_values * (1 - done_t)

        loss = nn.MSELoss()(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ==========================================
# 4. 開始執行訓練迴圈
# ==========================================
if __name__ == "__main__":
    env = env_gym.make("gyenv/ContinuousCartPole_YO-v0")
    state_dim = env.observation_space.shape[0]
    
    # 這裡我們設定 21 種離散力量代號讓 DQN 去學
    num_actions = 41 
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
            action_env = agent.get_force_value(action_idx)
            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated

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

    # --- 第二階段：展示動畫（開啟 render_mode="human"） ---
    print("\n=== 開始展示訓練成果動畫 ===")
    # 建立一個有畫面的連續環境，並開啟人類渲染模式
    eval_env = env_gym.make("gyenv/ContinuousCartPole_YO-v0", render_mode="human")
    
    for test_ep in range(3):  
        state, _ = eval_env.reset()
        done = False
        total_test_reward = 0
        
        print(f"正在播放第 {test_ep + 1} 次展示...")
        while not done:
            action_idx = agent.choose_action(state, evaluate=True)
            action_env = agent.get_force_value(action_idx)
            
            state, reward, terminated, truncated, _ = eval_env.step(action_env)
            done = terminated or truncated
            total_test_reward += reward
            
            time.sleep(0.02)

        print(f"第 {test_ep + 1} 次展示結束，獲得分數: {total_test_reward}")

    eval_env.close()
    print("所有展示結束！")