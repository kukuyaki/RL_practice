'''
1.將學習率逐步遞減，讓路線趨近於收斂
2.先用路線規劃演算法跑一次路線，在讓深度學習沿著這條路現去優化，可以大幅減輕前期訓練時間

第一點是為了讓學習趨於平穩，在訓練後期不被隨機走動干擾導致無法收斂
第二點則是q table初始化的一種方法，提供一點線索給機器人，目的是讓訓練前期的速度增加，

除了用bfs先跑一次路線，並給經過的路線行為一點正reward
還可以透過讓每次節點更新時，根據現在節點和終點距離作額外的獎勵或懲罰

那我們這次先試試看bfs，且讓走過的路有0.1的起始值
'''
'''
稍微修改
    1.起點和終點位於左上和右下
    2.最後一次模擬時，每一步走0.3秒
    3.在每個格子上顯示最大行動期望值
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time

# --- 1. 視覺化工具設定 ---
plt.ion()  # 開啟互動模式
fig, ax = plt.subplots(figsize=(12, 10))

def update_plot(map_list, Q_table, cur_s, episode, step, sp):
    ax.clear()
    row_n, col_n = len(map_list), len(map_list[0])
    
    # 計算每個狀態的最大 Q 值並重塑為二維陣列
    max_q = np.array([max(q) for q in Q_table]).reshape(row_n, col_n)
    
    # 繪製熱力圖 (coolwarm: 藍=低, 紅=高)
    ax.imshow(max_q, cmap='coolwarm', interpolation='nearest', vmin=-10, vmax=100)
    
    # 繪製地圖標記
    for r in range(row_n):
        for c in range(col_n):
            if map_list[r][c] == 1:
                ax.text(c, r, 'X', ha='center', va='center', color='black', fontweight='bold')
            elif map_list[r][c] == 2:
                ax.text(c, r, 'G', ha='center', va='center', color='yellow', fontweight='bold')
            elif r !=0 or c !=0:
                ax.text(c, r, str(int(max(Q_table[r*col_n+c]))), ha='center', va='center', color='white', fontweight='bold')

    ax.text(sp[1],sp[0], 'S', ha='center', va='center', color='yellow', fontweight='bold')

    # 標示 Agent
    ax.scatter(cur_s[1], cur_s[0], c='white', s=300, edgecolors='black', label='Agent')
    ax.set_title(f"Episode: {episode} | Step: {step}| a: {a:4f}")
    plt.pause(0.01)



def is_reachable(grid, start, goal, row, col):
    '''
    使用 BFS 尋找路徑，回傳一個座標 list (從起點到終點)
    如果不可達，回傳 None
    '''
    # queue 裡面存的不是單一座標，而是一整條路徑
    queue = deque([[start]])
    visited = {start}
    
    while queue:
        path = queue.popleft() # 取出當前路徑
        r, c = path[-1]        # 取得路徑的最後一個點（當前位置）
        
        # 檢查是否到達終點
        if (r, c) == goal:
            return path
        
        # 探索鄰居
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < row and 0 <= nc < col and grid[nr][nc] != 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                # 將新點加入路徑，並放回 queue
                new_path = list(path)
                new_path.append((nr, nc))
                queue.append(new_path)
                
    return None # 若找不到路徑

def generate_map(row=6, col=5):
    while True:
        grid = [[0 for _ in range(col)] for _ in range(row)]
        start = (0,0)
        for r in range(row):
            for c in range(col):
                if (r, c) == start: continue
                if random.random() < 0.2: grid[r][c] = 1
        goal = (row-1,col-1)
        if goal != start and grid[goal[0]][goal[1]] == 0:
            grid[goal[0]][goal[1]] = 2
            path_first = is_reachable(grid, start, goal, row, col)
            if path_first:
                return grid, start,path_first

def check_next(cur_s, eGreddy, Q_table, map_list, r_table):
    node_id = cur_s[0]*len(map_list[0]) + cur_s[1]
    action_q = Q_table[node_id]
    if random.random() > eGreddy:
        max_val = np.max(action_q)
        indices = np.where(action_q == max_val)[0]
        decided_action = random.choice(indices.tolist())
    else:
        decided_action = random.choice([0,1,2,3])
    
    moves = {0: [1,0], 1: [-1,0], 2: [0,-1], 3: [0,1]}
    next_s = [cur_s[0] + moves[decided_action][0], cur_s[1] + moves[decided_action][1]]
    
    # 邊界與牆壁檢測
    if next_s[0]<0 or next_s[0]>=len(map_list) or next_s[1]<0 or next_s[1]>=len(map_list[0]) or map_list[next_s[0]][next_s[1]] == 1:
        return cur_s, decided_action, r_table[1]
    
    return next_s, decided_action, r_table[map_list[next_s[0]][next_s[1]]]

def update_q_table(Q_table, cur_s, action, next_s, a, gamma, r, col_n):
    node_id_cur = cur_s[0]*col_n + cur_s[1]
    node_id_next = next_s[0]*col_n + next_s[1]
    max_v = max(Q_table[node_id_next])
    Q_table[node_id_cur][action] += a * (r + gamma * max_v - Q_table[node_id_cur][action])

def Q_table_init(Q_table, path_f,col_n):
    n = len(path_f)
    for i in range(n-1):
        x = path_f[i+1][0] - path_f[i][0]
        y = path_f[i+1][1] - path_f[i][1]
        if x == 1:
            dirc = [10,     0,   0,   0]
        elif x == -1:
            dirc = [0,     10,   0,   0]
        elif y == 1:
            dirc = [0,     0,   0,   10]
        elif y == -1:
            dirc = [0,     0,   10,   0]
        Q_table[path_f[i][0]*col_n + path_f[i][1]] = dirc
    return Q_table
# --- 3. 主程式 ---
row_n, col_n = 30,30
map_list, start_point,path_f = generate_map(row_n, col_n)
Q_table = [[0.0 for _ in range(4)] for _ in range(row_n*col_n)]
a, gamma, r_table = 0.1, 0.95, [-0.1, -5, 100]
eGreddy, episode_n = 0.5, 1000
Q_table = Q_table_init(Q_table, path_f,col_n)
update_plot(map_list, Q_table, start_point, 0, 0,start_point) # 1. 初始化圖表

for episode in range(episode_n):
    if episode == episode_n-1:
        eGreddy = 0
    cur_s = list(start_point)
    step, reward_c = 0, 0
    #每次模擬a值都變小一點
    if episode %10 == 0:
        a = a*0.99
    while map_list[cur_s[0]][cur_s[1]] != 2:
        step += 1
        next_s, action, reward = check_next(cur_s, 0 if episode == episode_n-1 else eGreddy, Q_table, map_list, r_table)
        update_q_table(Q_table, cur_s, action, next_s, a, gamma, reward, col_n)
        cur_s = next_s
        reward_c += reward

        ###################################################
        # 2. Step 更新時繪圖 (為了效能，每 5 步更新一次)
        # if (step%10 == 0 and episode%20 == 0) or episode == episode_n-1:
        #     update_plot(map_list, Q_table, cur_s, episode, step,start_point)
        if episode == episode_n-1:
            update_plot(map_list, Q_table, cur_s, episode, step,start_point)
            time.sleep(0.3)
    ###################################################
    # 3. Episode 更新時繪圖
    update_plot(map_list, Q_table, cur_s, episode, step,start_point)
    print(f"Episode {episode}: Total Reward = {reward_c:.2f}: a value = {a:.2f}")

plt.ioff()
plt.show()

'''
結論
    實驗發現
    只在終點設置正reward，會導致正回饋無法傳遞到路線前半段，因為學習率會讓正回饋越來越稀
    地圖如果太大速度會變慢，可能要要思考一下如何調整探索策略

讓我們來複習一下學到的東西
Q1.py 實作Q learning
Q2.py 同上
Q3.py 視覺化 
Q4.py 優化

感覺都是在圍繞Q learning轉圈圈，但也想必讓大家對Q learning有深刻的印象，隨後講到其他概念時，我們會把Q learning當標準，這樣你們會更好懂


強化學習演算法分類
    1. on policy:擇路策略 與 更新Q值策略 一樣     如：sarsa
      off policy:擇路策略 與 更新Q值策略 不一樣    如：Q learning,  DQN

    2. model free : 完全透過模擬試錯得到的回饋學習 Q learning, sarsa, DQN
       model based: 每一步都會停下來模擬後面X步後，才真的決定下一步
    
    3.Value-based:學的是「狀態（或動作）有多好」（Q值、V值），然後採取值最高的方法 Q learning, sarsa, DQN
     Policy-based:直接學一個機率分佈函數 $\pi(a\vert{}s)$，告訴你在某個狀態下選各個動作的機率是多少。PPO, TRPO
     Actor-Critic:結合前兩者。「Actor」負責根據策略選動作（Policy），「Critic」負責評估這個動作好不好（Value）。A2C / A3C, SAC

可以看出我們已經會的q learning 和類似的sarsa和下一步要使用神經網路的DQN，還只在on/off policy上面有差異而已
之後我們還要試試看model based
試試看policy based 和 actor critic

下一關，DQN.py
'''