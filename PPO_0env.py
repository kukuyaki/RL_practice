'''
ppo其實就是一般的a2c算法但是限制了下一步的上下限，以避免學習一下過於偏差
ppo也是actor critic裡面算高級的算法
因為我們跳過了policy based算法，我們稍微提及一下從dqn到actor critic會遇到的名詞
value
    State-Value Function (V(s)，狀態價值函數)
    Action-Value Function (Q(s, a)，動作價值函數)
    Bellman Equation (貝爾曼方程式)
    Temporal Difference (TD Error，時序差分誤差)
    Replay Buffer (經驗回放池)
    Exploration vs. Exploitation (探索與利用)
policy
    Policy (π(a|s)，策略)
    Discrete vs. Continuous Action Space (離散與連續動作空間)
    Trajectory (τ，軌跡)
    On-Policy vs. Off-Policy (同策略與異策略)
actor critic
    Actor (演員)：即策略網路（Policy Network）
    Critic (評論家)：即價值網路（Value Network）
    Baseline (基準線)
    Advantage Function (A(s, a)，優勢函數)：極度重要！ 公式為 A(s, a) = Q(s, a) - V(s)
PPO
    Probability Ratio (\(r_t(\theta)\)，新舊策略機率比值)
    Importance Sampling (重要性採樣)
    Surrogate Objective (替代目標函數)
    Clipping Mechanism (裁剪機制)
    Generalized Advantage Estimation (GAE，廣義優勢估計)
SAC
    最大熵（Maximum Entropy）

------------------------------------------------------------------------------------------------------------------------------
我們來實作PPO
這次就先不自己創建新環境
使用gymnasium[mujoco] 的 pusher 環境，但其實只要直接gymnasium.make("Pusher-v5")就可以用了
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/pusher_v5.py

一個機械手臂把目標推到指定地點，環境有7個輸入節點，23個輸出節點
沒有termination
只有truncation, 100 步結束

reward懲罰設計
    目標物與指定地點距離
    動作幅度盡量小
    機械手臂指頭離目標物越遠分數扣越重

Action Space
Box(-2.0, 2.0, (7,), float32)

Observation Space
Box(-inf, inf, (23,), float64)
'''
