'''
G系列是作DQN的離散輸入遊戲實作

1.先創造環境，也就是遊戲本身，reward
2.寫agent，Q值更新公式

撿寶物遊戲
    從1,1出發，目標是抵達10,10的寶箱位置並且按下互動鍵
    碰到怪物就結束模擬
    在門口按下互動按鈕可以將門永久開啟
    無法穿透牆壁和未打開的門

輸入的節點改成2+2+8個
    2自己的xy座標
    2寶物的xy座標
    8以自己為中心的周圍8個格子有什麼以及他們的狀態

輸出5個節點
    上、下、左、右、互動

reward設計上使用
    1.每多一步-0.1分
    2.+(當前格子與寶物格子的距離)
    3.撞到怪物-100且結束當前模擬
    4.抵達寶箱位置且按下互動按鈕+100且結束當前模擬
    5.試圖往牆壁或關閉的門走-5分
    6.開啟門的時候+20分 



'''
from typing import Optional
import numpy as np
import gymnasium as gym

class game(gyn.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.agent_x = 1
        self.agent_y = 1
        self.chest_x = 10
        self.chest_y = 10
        self.wall = [[10,8],[8,8],[8,9],[8,10]]
        self.door = [9,8]
        self.open_door = []
        self.monster = [9,4]
        self.action_direct = {
                    1:[0,1],
                    2:[0,-1],
                    3:[-1,0],
                    4:[1,0]
                }
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 回傳開局的狀態 (np.ndarray) 與 info 字典
        state = np.zeros(12, dtype=np.float32)
        return state, {}
    
    def check_colli(pos):
        #超出地圖，撞到牆壁或門，怪物
        pass

    def step(self, action):
        terminated = 0
        reward = 0
        if action == 5:
            next_s, reward, terminated = interect_chest_door()
        else:
            direc =self.action_direct[action]
            next_s == [self.agent_x+direc[0], self.agent_y+direc[1]]
            next_s, reward, terminated = check_colli(next_s) #確認有無超出範圍或碰撞
        self.agent_x = next_s[0]
        self.agent_y = next_s[1]
        reward = -0.1
        return next_s, reward, terminated
