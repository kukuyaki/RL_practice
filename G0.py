'''
G系列是作DQN的離散輸入遊戲實作

1.先創造環境，也就是遊戲本身，reward
2.寫agent，Q值更新公式

撿寶物遊戲
    從1,1出發，目標是抵達10,10的寶箱位置並且按下互動鍵
    碰到怪物就結束模擬
    在門口按下互動按鈕可以將門永久開啟
    無法穿透牆壁和未打開的門

輸入的節點改成1+2+2+8個
    1累積步數
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

class Game_A(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.agent_x = 1
        self.agent_y = 1
        self.step_n = 0
        self.chest_x = 10
        self.chest_y = 10
        self.wall = [[10,8],[8,8],[8,9],[8,10]]
        self.door = [[9,8]]
        self.open_door = []
        self.monster = [[9,4],[8,4],[7,4],[6,4],[5,4],[4,4],[3,4],[3,5],[3,6],[3,7],[3,8],
                        [3,8],[3,7],[3,6],[3,5],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4],[9,4]]
        self.monster_pos = self.step_n%22
        self.boarder = [[0,i] for i in range(11) ] + [[i,0] for i in range(11)] + [[i,11] for i in range(11)] +[[11,i] for i in range(11)]
#   11邊邊邊邊邊邊邊邊邊邊邊邊邊    
#   10邊口口口口口口口口牆口箱邊
#    9邊口口口口口口口口牆口口邊
#    8邊口口怪口口口口口牆門牆邊
#    7邊口口口口口口口口口口口邊
#    6邊口口口口口口口口口口口邊
#    5邊口口口口口口口口口口口邊
#    4邊口口怪口口口口口怪口口邊
#    3邊口口口口口口口口口口口邊
#    2邊口口口口口口口口口口口邊
#    1邊我口口口口口口口口口口邊
#    0邊邊邊邊邊邊邊邊邊邊邊邊邊
#     0 1 2 34 5 67 8 9 1011
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": (self.agent_x, self.agent_y),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )
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
    
    def surrond_door(self, next_s):
        for i in [[1,0],[-1,0],[0,1],[0,-1]]:
            if [next_s[0] + i[0],  next_s[1] + i[1]] in self.door:
                self.door.remove([next_s[0] + i[0],  next_s[1] + i[1]])
                self.open_door.append([next_s[0] + i[0],  next_s[1] + i[1]])
                return 1
        return 0
    def _get_state(self):
        # 組合 13 維狀態陣列
        state = np.zeros(13, dtype=np.float32)
        state[0] = self.step_n
        state[1] = self.agent_x
        state[2] = self.agent_y
        state[3] = self.chest_x
        state[4] = self.chest_y
        
        # 以自己為中心的周圍 8 個格子狀態 (填入 state[5:13])
        dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        dy = [1, 1, 1, 0, 0, -1, -1, -1]
        for idx, (d_x, d_y) in enumerate(zip(dx, dy)):
            cx = self.agent_x + d_x
            cy = self.agent_y + d_y
            val = 0 # 0: 空地
            if [cx, cy] in self.boarder or [cx, cy] in self.wall:
                val = 1 # 1: 牆壁或邊界
            elif [cx, cy] in self.door:
                val = 2 # 2: 關閉的門
            elif [cx, cy] == self.monster[self.step_n % len(self.monster)]:
                val = 3 # 3: 怪物
            state[5 + idx] = val
            
        return state

    def step(self, action):
        self.step_n+=1
        self.monster_pos = self.step_n%22
        terminated = 0
        reward = 0
        cur_pos = [self.agent_x, self.agent_y]
        
        if action == 0: #互動
            next_s = cur_pos
            if self.agent_x == self.chest_x and self.agent_y == self.chest_y:
                reward+=100
                terminated = 1
            if self.surrond_door(next_s):
                reward+=20
        else: #移動
            direc =self.action_direct[action]
            next_s = [self.agent_x+direc[0], self.agent_y+direc[1]]
            if next_s in self.boarder:
                reward -= 5
                next_s = cur_pos
            elif next_s == self.monster[self.monster_pos]:
                reward-=100
                terminated = 1
        reward -= 0.1
        self.agent_x = next_s[0]
        self.agent_y = next_s[1]
        next_state = self._get_state()
        info = {}
        return (next_state, reward, terminated, False, False)
