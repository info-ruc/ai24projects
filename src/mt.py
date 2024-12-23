import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from game import Board
import pickle
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet
from tkinter import messagebox
leixing=0
size=0
player1=0
player2=0
path="D:\\codes\\ai\\final\\ai24projects\\src\\models\\"
curplayer=1
players=[]


# 创建Tkinter根窗口
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 选择游戏类型
leixing_dialog = tk.Toplevel(root)  
leixing_dialog.geometry("300x100")
leixing_dialog.title("选择游戏类型")
leixing_input = tk.StringVar()
leixing_input.set("nogo")  
tk.Radiobutton(leixing_dialog, text="不围棋", variable=leixing_input, value="nogo").pack(anchor=tk.W)
tk.Radiobutton(leixing_dialog, text="田字棋", variable=leixing_input, value="sizi").pack(anchor=tk.W)
def on_ok():
    leixing_dialog.quit()
    leixing_dialog.destroy()
tk.Button(leixing_dialog, text="确定", command=on_ok).pack()
leixing_dialog.mainloop()


# 选择棋盘大小
size_dialog = tk.Toplevel(root)  
size_dialog.title("选择游戏类型")
size_dialog.geometry("300x200")
size_input = tk.StringVar()
size_input.set("3")  
tk.Radiobutton(size_dialog, text="3*3", variable=size_input, value="3").pack(anchor=tk.W)
tk.Radiobutton(size_dialog, text="4*4", variable=size_input, value="4").pack(anchor=tk.W)
tk.Radiobutton(size_dialog, text="5*5", variable=size_input, value="5").pack(anchor=tk.W)
tk.Radiobutton(size_dialog, text="6*6", variable=size_input, value="6").pack(anchor=tk.W)
def on_ok():
    size_dialog.quit()
    size_dialog.destroy()
tk.Button(size_dialog, text="确定", command=on_ok).pack()
size_dialog.mainloop()
size=int(size_input.get())

# 选择先手类型
p1_dialog = tk.Toplevel(root)  
p1_dialog.title("选择先手")
p1_dialog.geometry("300x100")
p1_input = tk.StringVar()
p1_input.set("human")  
tk.Radiobutton(p1_dialog, text="人类", variable=p1_input, value="human").pack(anchor=tk.W)
tk.Radiobutton(p1_dialog, text="AI", variable=p1_input, value="AI").pack(anchor=tk.W)
def on_ok():
    p1_dialog.quit()
    p1_dialog.destroy()  
tk.Button(p1_dialog, text="确定", command=on_ok).pack()
p1_dialog.mainloop()
players.append(p1_input.get())

# 选择后手类型
p2_dialog = tk.Toplevel(root)  
p2_dialog.title("选择后手")
p2_dialog.geometry("300x100")
p2_input = tk.StringVar()
p2_input.set("human") 
tk.Radiobutton(p2_dialog, text="人类", variable=p2_input, value="human").pack(anchor=tk.W)
tk.Radiobutton(p2_dialog, text="AI", variable=p2_input, value="AI").pack(anchor=tk.W)
def on_ok():
    p2_dialog.quit()
    p2_dialog.destroy()  
tk.Button(p2_dialog, text="确定", command=on_ok).pack()
p2_dialog.mainloop()
players.append(p2_input.get())
players.append("NULL")#避免游戏结束后的点击


root.quit()  # 结束参数设置

board = Board(width=size, height=size,leixing=leixing_input.get())
board.init_board()
model_file=path+leixing_input.get()+'_'+str(size)+'_'+str(size)+'.model'
try:
    policy_param = pickle.load(open(model_file, 'rb'))
    print("dbg1")
except:
    policy_param = pickle.load(open(model_file, 'rb'),
                                encoding='bytes')  # To support python3
    print("dbg2")

best_policy = PolicyValueNet(size, size, model_file)
mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                            c_puct=5,
                            n_playout=400)#加载选择的模型

class Chessboard:
    def __init__(self):
        self.size = size  # 棋盘的大小
        self.board = np.zeros((size, size))  # 初始化棋盘为 0（空格）
        self.current_player = 1  # 玩家 1 为先手（用 1 表示），玩家 2 用 2 表示
        self.fig, self.ax = plt.subplots()  # 创建图形和坐标轴
        self.ax.set_xticks(np.arange(0, self.size, 1))
        self.ax.set_yticks(np.arange(0, self.size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        self.ax.set_aspect('equal')  # 强制坐标轴的比例一致
        self.ax.set_xlim(0, self.size)  # X轴范围
        self.ax.set_ylim(0, self.size)  # Y轴范围

        # 绘制初始棋盘
        self.draw_board()

        # 连接鼠标点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def draw_board(self):
        # 绘制棋盘格子，白色和黑色交替排列
        chessboard = np.zeros((self.size, self.size))
        chessboard[1::2, ::2] = 1  # 白格
        chessboard[::2, 1::2] = 1  # 白格
        self.ax.imshow(chessboard, cmap='binary', interpolation='nearest', extent=(0, self.size, self.size, 0))

        # 绘制当前的棋子
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] == 1:  # 玩家1的棋子
                    self.ax.plot(col + 0.5, row + 0.5, 'ro', markersize=20)  # 红色棋子
                elif self.board[row, col] == 2:  # 玩家2的棋子
                    self.ax.plot(col + 0.5, row + 0.5, 'bo', markersize=20)  # 蓝色棋子
    def update(self):
        self.draw_board()
        plt.draw()
        plt.pause(0.2)
        end, winner = board.game_end()
        if end:
            self.current_player=3#避免游戏结束后的点击
            if winner != -1:
                if(winner==1):
                    messagebox.showinfo("结局", "先手获胜")
                if(winner==2):
                    messagebox.showinfo("结局", "后手获胜")
            else:
                messagebox.showinfo("结局", "平局")
            plt.close()
        else:
            if(players[self.current_player-1]=="AI"):
                mcts_player.set_player_ind(self.current_player)#让AI认为自己是某一方
                
                move = mcts_player.get_action(board)
                row,col=board.move_to_location(move)
                board.do_move(move)
                self.board[row, col] = self.current_player
                self.current_player = 3 - self.current_player
                self.update()

    def on_click(self, event):
        if(players[self.current_player-1]=="human"):
            # 获取点击位置的坐标
            if event.xdata is None or event.ydata is None:
                return  # 确保点击的是棋盘区域
            col = int(np.floor(event.xdata))  # 向下取整得到列号
            row = int(np.floor(event.ydata))  # 向下取整得到行号
            move = board.location_to_move([row,col])
            if move == -1 or move not in board.availables:
                return # 确保没被点击过
            board.do_move(move)

            '''# 判断该位置是否已经落子
            if self.board[row, col] != 0:
                print("This position is already occupied!")
                return'''

            # 落子
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player  # 切换玩家（1 <-> 2）

            # 重新绘制棋盘
            self.update()
chessboard = Chessboard()
chessboard.update()#如果AI先手先让AI下
plt.show()