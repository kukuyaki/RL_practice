'''
播放模型
'''


import time
import torch
import gymnasium as env_gym
import numpy as np

# 假設你的 DQN 類別和 DQNAgent 類別放在同一個檔案，或者從你的訓練檔案中 import 進來
# from your_training_script import DQN, DQNAgent, device 
# 為了方便，這裡直接沿用你原本的定義：
from collections import deque
import random
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 定義 DQN 與 Agent (必須跟訓練時的架構一模一樣) ---
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

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_network = DQN(state_dim, action_dim).to(device)
        
    def obs_to_tensor(self, obs):
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

    def select_action(self, obs):
        with torch.no_grad():
            state_tensor = self.obs_to_tensor(obs)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

# --- 2. 註冊環境 (請確保你的 G0.py 路徑正確) ---
env_gym.register(
    id="Game_yo",
    entry_point="G0env:Game_A",
    max_episode_steps=500,
)

if __name__ == "__main__":
    # 建立環境並開啟畫面渲染 (render_mode="human")
    env = env_gym.make("Game_yo", render_mode="human")
    
    state_dim = 13
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # 3. 載入你存下來的模型權重檔案
    model_path = "./G999model.pth"
    agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_network.eval()  # 設定為評估模式
    print(f"成功載入模型: {model_path}")
    
    # 4. 開始播放展示動畫
    num_test_episodes = 5  # 想要連續看幾場
    
    for ep in range(num_test_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        print(f"\n--- 第 {ep+1} 場遊戲開始 ---")
        
        while not terminated and not truncated:
            # 挑選最佳動作 (完全不隨機探索)
            action = agent.select_action(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 控制畫面播放速度 (秒數)，避免跑太快看不清楚
            time.sleep(0.3) 
            
        print(f"遊戲結束！本場總得分: {total_reward:.2f}")
        time.sleep(1.0)  # 每場結束後暫停一秒再開新的一場

    env.close()
    print("播放完畢，視窗已關閉。")
    torch.save(ppo_agent.policy.state_dict(), "./ppo_pusher.pth")
    print(f"Model successfully saved to {filepath}")



'''
G系列實驗結語：
可看出：
    模型有效果
    正確的獎勵函數的重要性
    訓練速度不是很快
    訓練出來的模型會抽搐，成因不明確，多半和獎勵函式有關，但可以抵達終點
可改進：
    更好的獎勵函數
    調整行動列表（如新增停止不動）
可優化：
    使用tensor的cuda功能讓訓練更快
    使用gymnasium的SyncVectorEnv讓模型並行訓練，但這只能在GPU上使用
    經驗回放優先採用Loss值大的樣本
    調整資料結構，優先使用numpy
    境量避免cpu和gpu之間的切換

下一步：PPO_0.py 直接跳過policy based, 直接學習actor critic吧！！

'''