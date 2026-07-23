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
    7.當離寶箱的距離破新紀錄時，加分

G0:製作環境
G1:agent
G2:用訓練好的agent資料展示動畫
G999:訓練好的模型
'''
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces



class Game_A(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.nearst = 9999
        self.touch_door = 0
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
        
        self.boarder = [[0,i] for i in range(12) ] + [[i,0] for i in range(12)] + [[i,11] for i in range(12)] +[[11,i] for i in range(12)]
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
            {   #寶箱位置
                "chest_pos": gym.spaces.Box(0, 12 - 1, shape=(2,), dtype=int),
                #步數
                "steps": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=int),
                #周圍八個格子的物品，0沒東西1邊界2牆壁3關閉的門4開啟的門5怪物6寶箱
                "surrunding_lu": gym.spaces.Discrete(7),#左上
                "surrunding_lm": gym.spaces.Discrete(7),#左中
                "surrunding_ld": gym.spaces.Discrete(7),#左下

                "surrunding_mu": gym.spaces.Discrete(7),#中上
                "agent_pos": gym.spaces.Box(0, 12 - 1, shape=(2,), dtype=int),
                "surrunding_md": gym.spaces.Discrete(7),#中下

                "surrunding_ru": gym.spaces.Discrete(7),#右上
                "surrunding_rm": gym.spaces.Discrete(7),#右中
                "surrunding_rd": gym.spaces.Discrete(7),#右下
            }
        )
        self.action_direct = {
                    1:[0,1],
                    2:[0,-1],
                    3:[-1,0],
                    4:[1,0]
                }
        
    def _get_obs(self):
        xxx = [-1,0,1]
        yyy = [1,0,-1]
        eight_char = []
        for i in xxx:
            for j in yyy:
                xt ,yt = self.agent_x+i, self.agent_y+j
                if [xt,yt] == [self.chest_x,self.chest_y]: eight_char.append(6)
                elif [xt,yt] in self.boarder: eight_char.append(1)
                elif [xt,yt] in self.wall: eight_char.append(2)
                elif [xt,yt] in self.door: eight_char.append(3)
                elif [xt,yt] in self.open_door: eight_char.append(4)
                elif [xt,yt] == self.monster[self.step_n%22]: eight_char.append(5)
                else:eight_char.append(0)
                
        observation = {
            "chest_pos": np.array([self.chest_x,self.chest_y], dtype=int),
            #步數
            "steps": np.array([self.step_n], dtype=int),
            #周圍八個格子的物品，0沒東西1邊界2牆壁3關閉的門4開啟的門5怪物6寶箱
            "surrunding_lu": np.array(eight_char[0], dtype=int),#左上
            "surrunding_lm": np.array(eight_char[1], dtype=int),#左中
            "surrunding_ld": np.array(eight_char[2], dtype=int),#左下

            "surrunding_mu": np.array(eight_char[3], dtype=int),#中上
            "agent_pos": np.array([self.agent_x,self.agent_y], dtype=int),
            "surrunding_md": np.array(eight_char[5], dtype=int),#中下

            "surrunding_ru": np.array(eight_char[6], dtype=int),#右上
            "surrunding_rm": np.array(eight_char[7], dtype=int),#右中
            "surrunding_rd": np.array(eight_char[8], dtype=int),#右下
        }
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 回傳開局的狀態 (np.ndarray) 與 info 字典
        self.agent_x = 1
        self.agent_y = 1
        self.step_n = 0
        self.door = [[9,8]]
        self.open_door = []
        observation = self._get_obs()
        info = {}
        return observation, info
    
    def surrond_door(self, next_s):
        for i in [[1,0],[-1,0],[0,1],[0,-1]]:
            if [next_s[0] + i[0],  next_s[1] + i[1]] in self.door:
                self.door.remove([next_s[0] + i[0],  next_s[1] + i[1]])
                self.open_door.append([next_s[0] + i[0],  next_s[1] + i[1]])
                return 1
        return 0


    def step(self, action):
        self.step_n+=1
        terminated = 0
        reward = 0
        cur_pos = [self.agent_x, self.agent_y]
        
        if action == 0: #互動
            next_s = cur_pos
            if self.agent_x == self.chest_x and self.agent_y == self.chest_y:
                reward+=100
                terminated = 1
            if self.surrond_door(next_s) and self.touch_door == 0:
                self.touch_door = 1
                reward+=20
        else: #移動
            direc =self.action_direct[action]
            next_s = [self.agent_x+direc[0], self.agent_y+direc[1]]
            distance_after = abs(next_s[0] - self.chest_x) + abs(next_s[1] - self.chest_y )
            if self.nearst > distance_after: 
                self.nearst = distance_after
                reward+=1
            if next_s in self.boarder or next_s in self.wall or next_s in self.door:
                reward -= 10
                next_s = cur_pos
            elif next_s == self.monster[self.step_n%22]:
                reward-=100
                terminated = 1
        
        reward -= 0.1
        self.agent_x = next_s[0]
        self.agent_y = next_s[1]

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, False, info


    def render(self):
        """實作 render 函式，用文字視覺化地圖狀態"""
        if self.render_mode == "human":
            # 建立一個 12x12 的空白地圖格子
            grid = [["口" for _ in range(12)] for _ in range(12)]
            
            # 填入邊界
            for b in self.boarder:
                grid[b[1]][b[0]] = "邊" # 為了直觀，x 對應欄、y 對應列
                
            # 填入牆壁
            for w in self.wall:
                grid[w[1]][w[0]] = "牆"
                
            # 填入關閉的門
            for d in self.door:
                grid[d[1]][d[0]] = "門"
                
            # 填入開啟的門
            for od in self.open_door:
                grid[od[1]][od[0]] = "開"
                
            # 填入寶箱
            grid[self.chest_y][self.chest_x] = "箱"
            
            # 填入怪物
            m_pos = self.monster[self.step_n%22]
            grid[m_pos[1]][m_pos[0]] = "怪"
            
            # 填入玩家自己
            grid[self.agent_y][self.agent_x] = "我"
            
            # 輸出畫面（從頂端 11 畫到 0，符合你原本的註解排版）
            print(f"\n--- 步數: {self.step_n} ---")
            for y in range(11, -1, -1):
                row_str = "".join(grid[y])
                print(f"{y:2d} {row_str}")
            print("    01234567891011")
    def close(self):
        pass