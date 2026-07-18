'''
隨機生成地圖
且起點在地圖中央

if (step%10 == 0 and episode%20 == 0) or episode == episode_n-1:
且每隔20個模擬就跑一次繪圖，其中每10步紀錄一次
最後一次模擬時，每一個步驟都會模擬
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time

# --- 1. 視覺化工具設定 ---
plt.ion()  # 開啟互動模式
fig, ax = plt.subplots(figsize=(6, 5))

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
    ax.text(sp[0], sp[1], 'S', ha='center', va='center', color='yellow', fontweight='bold')

    # 標示 Agent
    ax.scatter(cur_s[1], cur_s[0], c='white', s=300, edgecolors='black', label='Agent')
    ax.set_title(f"Episode: {episode} | Step: {step}")
    plt.pause(0.01)

# --- 2. 核心邏輯 ---
def is_reachable(grid, start, goal, row, col):
    queue = deque([start])
    visited = {start}
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal: return True
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < row and 0 <= nc < col and grid[nr][nc] != 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

def generate_map(row=6, col=5):
    while True:
        grid = [[0 for _ in range(col)] for _ in range(row)]
        start = (row // 2, col // 2)
        for r in range(row):
            for c in range(col):
                if (r, c) == start: continue
                if random.random() < 0.2: grid[r][c] = 1
        goal = (random.randint(0, row-1), random.randint(0, col-1))
        if goal != start and grid[goal[0]][goal[1]] == 0:
            grid[goal[0]][goal[1]] = 2
            if is_reachable(grid, start, goal, row, col):
                return grid, start

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

# --- 3. 主程式 ---
row_n, col_n = 30,30
map_list, start_point = generate_map(row_n, col_n)
Q_table = [[0.0 for _ in range(4)] for _ in range(row_n*col_n)]
a, gamma, r_table = 0.1, 0.95, [-0.1, -5, 100]
eGreddy, episode_n = 0.5, 1000

update_plot(map_list, Q_table, start_point, 0, 0,start_point) # 1. 初始化圖表

for episode in range(episode_n):
    cur_s = list(start_point)
    step, reward_c = 0, 0
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
    ###################################################
    # 3. Episode 更新時繪圖
    update_plot(map_list, Q_table, cur_s, episode, step,start_point)
    print(f"Episode {episode}: Total Reward = {reward_c:.2f}")

plt.ioff()
plt.show()