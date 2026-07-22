import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as env_gym
import time
from gymnasium.envs.registration import register
'''
環境：Create_env1.py

用ppo去嘗試連續的輸出
有好一些，但還是有加強的空間
'''
# ==========================================
# 0. 註冊自定義環境
# ==========================================
register(
    id="gyenv/ContinuousCartPole_YO-v0",
    entry_point="Create_env1:CartPoleEnv_YO",  # 請確保 Create_env1.py 內有 CartPoleEnv_YO
    max_episode_steps=500,
)

# ==========================================
# 1. 定義 Actor-Critic 神經網路 (支援連續動作)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 共享或獨立的特徵提取層
        self.actor_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Actor 輸出：常態分佈的平均值 (Mean)
        self.actor_mean = nn.Linear(64, action_dim)
        
        # Actor 的標準差 (Log Std)，設為可學習參數
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic 輸出：狀態價值 V(s)
        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        """與環境互動時採樣動作"""
        action_space_low = -50.0
        action_space_high = 50.0

        features = self.actor_fc(state)
        mean = self.actor_mean(features)
        
        # 限制平均值在合理範圍內，並透過 tanh 映射，再縮放至 [-200, 200]
        mean = torch.tanh(mean) * action_space_high
        
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        raw_action = dist.sample()
        
        # 確保動作不超過環境極限
        action = torch.clamp(raw_action, action_space_low, action_space_high)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """訓練時計算給定動作的 log_prob、entropy 以及狀態價值"""
        action_space_high = 50.0

        features = self.actor_fc(state)
        mean = self.actor_mean(features)
        mean = torch.tanh(mean) * action_space_high
        
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        state_value = self.critic_fc(state)
        
        return log_prob, state_value, entropy

# ==========================================
# 2. PPO 演算法主體
# ==========================================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.lmbda = 0.95        # GAE 參數
        self.clip_eps = 0.2      # PPO 截斷範圍
        self.epochs = 10         # 每次更新的迭代次數
        self.batch_size = 64
        self.lr = 3e-4

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

        # 暫存一個 Rollout 的軌跡資料
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.policy.act(state_t)
            # 同時計算此時的 Critic 價值
            value = self.policy.critic_fc(state_t)
        
        self.states.append(state)
        self.actions.append(action.numpy()[0])
        self.log_probs.append(log_prob.numpy()[0])
        self.values.append(value.numpy()[0])

        return action.numpy()[0]  # 回傳一維陣列 [force]

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def update(self):
        # 轉換成 PyTorch Tensor
        old_states = torch.FloatTensor(np.array(self.states))
        old_actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))

        # 計算 Generalized Advantage Estimation (GAE)
        returns = []
        advantages = []
        gae = 0
        values = self.values + [np.array([0.0])]  # 補上最後一個

        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i+1][0] * (1 - self.dones[i]) - values[i][0]
            gae = delta + self.gamma * self.lmbda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i][0])

        returns = torch.FloatTensor(returns).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        
        # 優化優勢函數正規化 (加速收斂)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 進行多輪 PPO 梯度更新
        dataset_size = old_states.size(0)
        for _ in range(self.epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                if len(batch_indices) == 0:
                    continue

                b_states = old_states[batch_indices]
                b_actions = old_actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]

                # 評估當前政策
                log_probs, state_values, entropy = self.policy.evaluate(b_states, b_actions)

                # 計算機率比率 r(t)
                ratios = torch.exp(log_probs - b_old_log_probs)

                # PPO 核心 Clip Loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(state_values, b_returns)
                
                # 總 Loss（加入 entropy 鼓勵探索）
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.clear_memory()

# ==========================================
# 3. 執行訓練主迴圈
# ==========================================
if __name__ == "__main__":
    env = env_gym.make("gyenv/ContinuousCartPole_YO-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 1

    agent = PPOAgent(state_dim, action_dim)
    
    num_episodes = 5000
    update_timestep = 200  # 累積多少步驟後執行一次 PPO 更新
    total_steps = 0

    print("開始 PPO 訓練（連續動作空間，力道 ±200）...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. 透過常態分佈採樣連續動作
            action = agent.select_action(state)
            
            # 2. 與環境互動
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 儲存回饋
            agent.rewards.append(reward)
            agent.dones.append(float(done))

            state = next_state
            total_reward += reward
            total_steps += 1

            # 累積足夠步數後進行 PPO 更新
            if total_steps >= update_timestep:
                agent.update()
                total_steps = 0

        if (episode + 1) % 200 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward}")

    print("訓練完成！")
    env.close()

    # ==========================================
    # 4. 展示訓練成果動畫
    # ==========================================
    print("\n=== 開始展示 PPO 連續控制展示動畫 ===")
    eval_env = env_gym.make("gyenv/ContinuousCartPole_YO-v0", render_mode="human")
    
    for test_ep in range(15):
        state, _ = eval_env.reset()
        done = False
        total_test_reward = 0
        
        print(f"正在播放第 {test_ep + 1} 次展示...")
        while not done:
            # 測試時直接取常態分佈的平均值（不進行隨機抽樣，展現最穩定表現）
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                features = agent.policy.actor_fc(state_t)
                mean = agent.policy.actor_mean(features)
                action = torch.tanh(mean) * 50.0
                action = action.numpy()[0]

            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_test_reward += reward
            
            time.sleep(0.01)

        print(f"第 {test_ep + 1} 次展示結束，獲得分數: {total_test_reward}")

    eval_env.close()
    print("所有展示結束！")