# 介紹
這邊是練習強化學習的地方

馬可夫決策過程 (MDP)

先學習「有模型」(Model-based) 的方法，理解價值函數的意義。

動態規劃 (Dynamic Programming)：策略評估 (Policy Evaluation) 與策略改進 (Policy Improvement)。

蒙地卡羅方法 (Monte Carlo Methods)：透過隨機取樣來估算價值，無需模型。

時序差分學習 (Temporal-Difference Learning)：TD(0)、SARSA 與 Q-Learning（這三個是必學的經典）。






深度強化學習 (Deep RL)
當狀態空間過大（如影像輸入）時，必須引入深度神經網路作為函數近似器。

Value-Based Methods：

DQN (Deep Q-Network)：將 Q-Learning 與神經網路結合，理解 Experience Replay 與 Target Network 的重要性。

Policy-Based Methods：

Policy Gradient：直接優化策略，而非價值函數。

Actor-Critic 架構：同時使用 Policy (Actor) 與 Value (Critic) 進行優化，是現代算法的主流架構（如 A2C, A3C）。

進階算法 (State-of-the-art)：

PPO (Proximal Policy Optimization)：目前業界與研究最常用的穩定算法。

SAC (Soft Actor-Critic)：考慮探索性 (Entropy) 的高效率連續控制算法。


Richard S. Sutton 的 Reinforcement Learning: An Introduction