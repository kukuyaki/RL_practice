'''
隨機生成地圖
且起點在地圖中央
'''

import numpy as np
import random
import os
import time
import random
from collections import deque
#兩個會用到的functoin，可以先往下看第一步，之後遇到再回來看函式細節
def check_next(cur_s, eGreddy):
    global Q_table
    global map_list
    global r_table
    global reward_c
    
    node_id = cur_s[0]*5 + cur_s[1]
    action = Q_table[node_id]    #action = ['u','d','l','r']
    #決定要按照Q_table還是隨機
    if random.random() > eGreddy:
        max_val = np.max(action)
        indices = np.where(action == max_val)[0]

        decided_action = random.choice(indices.tolist()) #0up     ,1down    ,2left      ,3right
        val = max_val
    else:
        decided_action = random.choice([0,1,2,3])
        val = action[decided_action]
    
    #計算下一個節點座標
    if decided_action == 0:
        next_s = [cur_s[0]+1,cur_s[1]+0]
    elif decided_action == 1:
        next_s = [cur_s[0]-1,cur_s[1]+0]
    elif decided_action == 2:
        next_s = [cur_s[0]+0,cur_s[1]-1]
    elif decided_action == 3:
        next_s = [cur_s[0]+0,cur_s[1]+1]

    #查看選擇的路線是否能走，不能的話就重新找一次
    
    xx =next_s[0]
    yy =next_s[1]
    while xx<0 or xx>=len(map_list) or yy<0 or yy>=len(map_list[0]):
        next_s, decided_action,reward = check_next(cur_s, eGreddy)
        xx =next_s[0]
        yy =next_s[1]

    #如果下點是牆壁，則保持員節點，回傳選擇的行動，和牆壁的reward懲罰
    if map_list[xx][yy] == 1:
        next_s = cur_s
        reward = r_table[1]
    else:
        reward = r_table[map_list[xx][yy]]
    
    return next_s, decided_action, reward


def update_q_table(cur_s, action, next_s, a, gamma, r):
    '''
    要注意不是每一次模擬才更新一次喔
    是每一步都會更新！！！

    根據公式更新數值
    Qsa = Qsa(1-a) + a(r + gamma * max(Qs'a'))
    '''
    global Q_table
    node_id_cur = cur_s[0]*5 + cur_s[1]
    node_id_next = next_s[0]*5 + next_s[1]
    max_v = max(Q_table[node_id_next])
    Q_table[node_id_cur][action] = Q_table[node_id_cur][action]*(1-a) + a*(r + gamma*max_v)
    return 0

def console_update(episode_eash,step, Q_table,cur_s,reward_c,next_s):
    # os.system('clear')
    print()
    print(f"{episode_eash =}")
    print(f"{step =}")
    print(f"{cur_s =}")
    print(f"{next_s =}")
    print(f"{reward_c =}")
    for i in Q_table:
        print(" ".join([f"{val:>5.2f}" for val in i]))
    return 0
##############################
#第一步：建立環境 和 初始化Q table
##############################
# -1 表示其點
#  0 表示可以走
#  1 表示不能走
#  2 表示目標


def generate_map_with_random_goal(row=6, col=5):
    while True:
        # 1. 建立全是 0 (路) 的地圖
        grid = [[0 for _ in range(col)] for _ in range(row)]
        
        # 2. 設定起點在正中央
        start = (row // 2, col // 2)
        grid[start[0]][start[1]] = 0 
        
        # 3. 隨機放置牆壁 (權重設定，例如 20%)
        for r in range(row):
            for c in range(col):
                if (r, c) == start: continue
                if random.random() < 0.2: # 20% 是牆
                    grid[r][c] = 1
        
        # 4. 隨機放置終點
        while True:
            goal = (random.randint(0, row-1), random.randint(0, col-1))
            if goal != start and grid[goal[0]][goal[1]] == 0:
                grid[goal[0]][goal[1]] = 2
                break
        
        # 5. 檢查連通性 (確保 Agent 不會被牆包圍)
        if is_reachable(grid, start, goal, row, col):
            return grid, start

def is_reachable(grid, start, goal, row, col):
    """使用 BFS 檢查是否有路可走"""
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

# 使用方式
row_n =6
column_n =5
map_list, start_point = generate_map_with_random_goal(row_n,column_n)
print(f"{row_n = }")
print(f"{column_n = }")
print(f"{start_point = }")
for i in map_list:
    print(i)
Q_table = [[0, 0, 0, 0] for _ in range(row_n*column_n)]

##############################
#第二步：設定參數
##############################
a = 0.1
gamma = 0.95
        #走到0      1    2時的即時reward
r_table = [-0.1,   -5,   100]

eGreddy = 0.5
episode_n = 100

start_time = time.time()
final_report_each_episode_reward = [999]  #最後用來展現訓練成果用的，和訓練無關

for episode_eash in range(episode_n):
    #在會後一次測試時，讓他完全按照最佳策略走，所以把eGreddy設成0
    if episode_eash == episode_n-1:
        eGreddy = 0

    cur_s = start_point

    done = 0
    step = 0
    reward_c = 0 #用來紀錄每次episode的分數累積，存在的目的是為了最後讓人類分析模型訓練過程有沒有逐漸進步，和訓練公式沒有關係

    #當以下條件發生時，結束當前episode
    #當走超過100步, 先不用因為會影響訓練
    #reward低於-100, 先不用因為會影響訓練 
    #走到終點！！！！！！！！！！！！！！！！！！！！！！！！！！要走到終點才會有正reward回饋
    # while step<100 and reward_c > -10000 and not done:
    while not done: #其實直接while 1:就好了，但我還是寫not done這樣比較清楚這個程式碼的邏輯
        step+=1
        #決定下一個節點，然後更新Q table
        next_s, action,reward = check_next(cur_s, eGreddy)
        update_q_table(cur_s, action, next_s, a, gamma, reward)
        cur_s = next_s
        #累加reward
        cur_node_type = map_list[cur_s[0]][cur_s[1]]
        # reward_c += r_table[cur_node_type]
        reward_c += reward
        #每秒更新資訊到終端畫面
        # if time.time() - start_time  >= 1:
            # start_time = time.time()
            # console_update(episode_eash,step, Q_table,cur_s,reward_c,next_s)
        #每step更新資訊到終端畫面
        # if 1:
        #     start_time = time.time()
        #     console_update(episode_eash,step, Q_table,cur_s,reward_c,next_s)
        #確認是否抵達終點
        if cur_node_type == 2:
            done = 1
            break
        

    final_report_each_episode_reward.append(reward_c)
print("finish!!!!!!!!!!!!!!!report time")
for i ,t in enumerate(final_report_each_episode_reward):
    if i % 1 == 0:
        print(f"{i:<10}:{t:>10.2f}")
print(f"{row_n = }")
print(f"{column_n = }")
print(f"{start_point = }")
for i in map_list:
    print(i)
