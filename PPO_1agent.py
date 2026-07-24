'''
PPO介紹：Proximal Policy Optimization(PPO)
    on policy
    model free actor critic

它的核心思想是在保證新策略與舊策略不會差異太大的前提下，尋找一個性能更好的策略
簡單來說，就是為了不要更新調整幅度太大，去設定一個上下限
關鍵函式PPO_Loss = Mean( Min( Ratio * Advantage, Clamp(Ratio, 1 - Epsilon, 1 + Epsilon) * Advantage ) )
Ratio = probs_new/probs_old
Ratio = torch.exp(log_probs - old_log_probs)
https://hackmd.io/@Su-Guan/SycB76Lqee

先創建兩個class
actor critic網路
PPO主體

在PPO init裡面會創建一個ac神經網路
我們會在main裡面創建一個ppo實體

ac網路要有兩個東西
    actor網路：policy決定動作
    critic網路：value給動作打分數

ppo要能夠做到：
    1.給輸入得輸出: 會去呼叫ac網路計算各個行動的機率分布，從中抽樣出新的aciton
    2.更新權重和bias：將memory的資料傳給神經網路，請他給出機率分布和熵


'''


import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn

import time
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 共享或獨立的特徵提取層（這裡採用獨立結構以求清晰）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        # 為了連續動作，引入可學習的對數標準差 (log_std)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        raise NotImplementedError
        
    def get_action(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action, action_log_prob

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_value = self.critic(state)
        
        return action_log_prob, state_value, dist_entropy

# 2. PPO 演算法主體
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
    def save(self, filepath="ppo_pusher.pth"):
        torch.save(self.policy.state_dict(), filepath)
        # print(f"Model successfully saved to {filepath}")

    # 載入模型權重
    def load(self, filepath="ppo_pusher.pth"):
        self.policy.load_state_dict(torch.load(filepath))
        self.policy.eval() # 載入後通常會切換到評估模式
        # print(f"Model successfully loaded from {filepath}")
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_log_prob = self.policy.get_action(state)
            state_val = self.policy.critic(state)
        return action.numpy().flatten(), action_log_prob.item(), state_val.item()

    def update(self, memory):
        # 轉換緩衝區資料為 Tensor
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.FloatTensor(np.array(memory['actions']))
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs']))
        rewards = memory['rewards']
        terminals = memory['terminals']
        values = memory['values']
        
        # 計算 GAE (Generalized Advantage Estimation)
        advantages = []
        gae = 0
        next_value = 0 # Pusher 通常依靠 truncation 結束，這裡簡化處理
        
        for step in reversed(range(len(rewards))):
            # 簡化版的優勢函數計算
            delta = rewards[step] + self.gamma * next_value * (1 - terminals[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - terminals[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
            
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        # 優勢函數正規化（助於訓練穩定）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # PPO 多次 Epoch 更新
        for _ in range(4): # 訓練 Epoch 數
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze(-1)
            
            # 計算比率 r(theta)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # 剪裁（Clipped）目標函數
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(state_values, returns)
            
            # 總損失：Actor 損失 + 價值損失 - 熵獎勵（鼓勵探索）
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
def main():
    # 建立 Pusher 環境（注意：需安裝 mujoco 套件）
    env = gym.make('Pusher-v5')
    render_env = gym.make('Pusher-v5', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo_agent = PPO(state_dim, action_dim)
    
    max_episodes = 1_000_000
    update_timestep = 3_000 # 每累積多少步進行一次 PPO 更新
    timestep = 0
    
    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'terminals': [], 'values': []}

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        if episode > 0 and episode % 50000 == 0:
            # print(f"--- 正在展示第 {episode} 次模擬的動畫 ---")
            test_state, _ = render_env.reset()
            test_done = False
            while not test_done:
                # 測試時不需要記錄 memory，直接讓 agent 選出動作
                action, _, _ = ppo_agent.select_action(test_state)
                test_next_state, _, terminated, truncated, _ = render_env.step(action)
                test_done = terminated or truncated
                test_state = test_next_state

        while not done:
            timestep += 1
            action, log_prob, val = ppo_agent.select_action(state)
            
            # 執行動作（Pusher 的 action 需要配合環境限制，gymnasium 會自動 clip，但我們程式輸出的分佈亦可）
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存入 Buffer
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['terminals'].append(terminated)
            memory['values'].append(val)
            
            state = next_state
            episode_reward += reward
            
            # 達到指定步數後執行 PPO 網路更新
            if timestep % update_timestep == 0:
                ppo_agent.update(memory)
                memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'terminals': [], 'values': []}
                
        # print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")
    
    torch.save(ppo_agent.policy.state_dict(), "./ppo_pusher.pth")
    # print(f"Model successfully saved to {"./ppo_pusher.pth"}")
    env.close()

if __name__ == '__main__':
    main()