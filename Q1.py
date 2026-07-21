'''
Q learning 不難，先從名詞講起，你會看到如下名詞：
    state:當前節點名稱
    action:可選的動作，根據所選動作會移動到下一個節點
    reward: 走到當前節點要扣多少分

且應該會有兩個表格
    map list: 存放地圖，或節點之間關係，多半也會把reward寫在這裡
    Q table: 會存放每一個節點state做的每一個action的期望值，期望值越大表示越佳路線，也越容易走這條路

更新Q table公式
    Qsa = Qsa(1-a) + a(r + gamma * max(Qs'a'))
    把當前節點s的a行動的期望值
    更新成
    保留一定的原始值，然後根據學習率和gamma值去加上下一個節點的所有行動中最高的期望值

    常會看見寫成
    Qsa = Qsa + a(r + gamma * max(Qs'a') - Qsa)
    後面的(r + gamma * max(Qs'a') - Qsa)被稱為TD Error (時序差分誤差)
    
    a = 學習率(要保留多少原始記憶)
    r = reward（這一步的真實回饋）
    gamma = 折扣因子 (對未來的期望)
    max(Qs'a') = 貪婪預期

選路策略
    每次選擇節點的時候，除了查Q table選擇路線，還要有小機率隨機選擇路線，以保證agent能夠有機會去嘗試新路線
    eGreddy: 隨機選擇路線的機率
    episode: 總共要訓練幾次

重要概念
    Q learning是一種off policy的策略
    off policy： 選擇路線的策略 和 更新Q值的策略不一樣

    Qlearning的選錄策略是 隨機
    Qlearning的選錄策略是 最大值

目標：讓一個agent在5*5的迷宮中找到出路
'''

import numpy as np
import random
import os
import time

#兩個會用到的functoin，可以先往下看第一步，之後遇到再回來看函式細節
def check_next(cur_s, eGreddy):
    '''
    輸入
        現在的節點
        eGreddy
    輸出
        下一個節點
        所使用的行動
        reward
    如果撞到邊緣，則重新選
    如果撞到牆壁，則原地不動
    '''
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
    while xx<0 or xx>=6 or yy<0 or yy>=5:
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
    os.system('clear')
    print()
    print(f"{episode_eash =}")
    print(f"{step =}")
    print(f"{cur_s =}")
    print(f"{next_s =}")
    print(f"{reward_c =}")
    for i in Q_table:
        print(i)
    return 0
##############################
#第一步：建立環境 和 初始化Q table
##############################
# -1 表示其點
#  0 表示可以走
#  1 表示不能走
#  2 表示目標
map_list =[
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 2]
                ]
#30個節點 * 4個行動 都設定為0
Q_table = [#上 下 左 右
            [0, 0, 0, 0],  #s1
            [0, 0, 0, 0],   #s2
            [0, 0, 0, 0],   #s3
            [0, 0, 0, 0],   #s4
            [0, 0, 0, 0],   #依此類推

            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0], 

            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0], 

            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0], 

            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  

            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
            [0, 0, 0, 0],  
                ]

##############################
#第二步：設定參數
##############################
a = 0.1
gamma = 0.95
        #走到0      1    2時的即時reward
r_table = [-0.1,   -5,   100]

eGreddy = 0.5
episode_n = 100

start_point = [0,2]
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
        if time.time() - start_time  >= 1:
            start_time = time.time()
            console_update(episode_eash,step, Q_table,cur_s,reward_c,next_s)
        #確認是否抵達終點
        if cur_node_type == 2:
            done = 1
            break
        

    final_report_each_episode_reward.append(reward_c)
print("finish!!!!!!!!!!!!!!!report time")
for i ,t in enumerate(final_report_each_episode_reward):
    if i % 1 == 0:
        print(f"{i:<10}:{t:>10.2f}")

'''
這是一個簡單的Q learning程式碼
會跑100次模擬，每次模擬都要跑到終點才會結束，因此如果地圖太過複雜，導致要很久才能到達終點，可能會影響學習

Q2.py 我們可以稍微小改一下程式碼，讓他可以隨機生成地圖
Q3.py 新增視覺化過程，這是強化學習非常重要的步驟，有助於理解，而且非常有趣
'''