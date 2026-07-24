import gymnasium as gym
from PPO_1agent import PPO  

def test_model():
    env = gym.make('Pusher-v5', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 建立 agent
    agent = PPO(state_dim, action_dim)
    
    # 載入訓練好的權重
    agent.load("./ppo_pusher.pth")
    
    # 跑幾次動畫看看成果
    for episode in range(50):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 測試時直接用 select_action，此時 agent 已經很聰明了
            action, _, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            
        print(f"Test Episode {episode+1}, Reward: {total_reward:.2f}")
        
    env.close()

if __name__ == '__main__':
    test_model()