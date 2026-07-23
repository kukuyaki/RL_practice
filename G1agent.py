import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as env_gym
import numpy as np
import time 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算裝置: {device}")

# --- 1. 定義 DQN 神經網路 ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# --- 2. 定義 DQN Agent 與 Q 值更新公式 ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.gamma = 0.99       # 折扣率 (Discount factor)
        self.epsilon = 1.0      # 探索率 (Epsilon-greedy)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=20000)

    def obs_to_tensor(self, obs):
        """將環境回傳的 Dict 觀察值壓平成一維陣列並轉成 Tensor"""
        flat_obs = np.concatenate([
            obs["chest_pos"],
            obs["steps"],
            [
                obs["surrunding_lu"], obs["surrunding_lm"], obs["surrunding_ld"],
                obs["surrunding_mu"], obs["surrunding_md"],
                obs["surrunding_ru"], obs["surrunding_rm"], obs["surrunding_rd"]
            ],
            obs["agent_pos"]
        ]).astype(np.float32)
        return torch.FloatTensor(flat_obs).unsqueeze(0).to(device)

    def select_action(self, obs,evaluate=False):
        """依據 Epsilon-Greedy 策略選擇動作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = self.obs_to_tensor(obs)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def update_model(self, batch_size=64):
        """Q 值更新公式與倒傳遞 (Backpropagation)"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        state_batch = torch.cat([self.obs_to_tensor(s) for s in states]).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        next_state_batch = torch.cat([self.obs_to_tensor(ns) for ns in next_states]).to(device)
        terminated_batch = torch.FloatTensor(terminateds).to(device)
        
        # 1. 計算目前的 Q 值: Q(s, a)
        q_values = self.q_network(state_batch)
        state_action_values = q_values.gather(1, action_batch).squeeze(1)
        
        # 2. 計算目標 Q 值: R + gamma * max Q(s', a') (Bellman Equation)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_values = reward_batch + (1 - terminated_batch) * self.gamma * max_next_q_values
            
        # 3. 計算 MSE Loss 並更新參數
        loss = self.criterion(state_action_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


env_gym.register(
    id="Game_yo",
    entry_point="G0env:Game_A",  # 請確保 Create_env1.py 內有 CartPoleEnv_YO 類別
    max_episode_steps=500,
)
# --- 3. 執行訓練主迴圈 ---
if __name__ == "__main__":
    env = env_gym.make("Game_yo")
    state_dim = 13
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    episodes = 500
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        ########################################################
        if ep%100 == 0 and ep != 0:
            print(f"\n--- 【Episode {ep} 成果展示】 ---")
            eval_env = env_gym.make("Game_yo", render_mode="human")
            eval_obs, _ = eval_env.reset()
            eval_terminated = False
            eval_total_reward = 0
            
            while not eval_terminated:
                # 測試展示時使用 evaluate=True，不隨機探索，展現最佳實力
                action = agent.select_action(eval_obs, evaluate=True)
                eval_obs, reward, eval_terminated, truncated, _ = eval_env.step(action)
                eval_total_reward += reward
                time.sleep(0.1)  # 控制畫面播放速度
                
            print(f"展示結束 | 獲得分數: {eval_total_reward:6.2f}\n")
            eval_env.close()
        ########################################################
        while not terminated:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, terminated)
            agent.update_model(batch_size=64)
            
            obs = next_obs
            total_reward += reward
            
        agent.decay_epsilon()
        if ep % 10 == 0:
            agent.update_target_network()
            print(f"Episode {ep:3d} | Total Reward: {total_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")
    

    # --- 新增：儲存訓練好的模型 ---
    model_path = "./G999model.pth"
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"模型已成功儲存至: {model_path}")