from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import numpy as np
import random


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # 定义机器人类型
        robot_type = 0 if unum == 1 else 4 # 假设守门员使用编号1，其他球员使用其他编号

        # 初始化基础代理
        # 参数: 服务器IP, 代理端口, 监控端口, 球衣号, 机器人类型, 队伍名称, 启用日志, 启用绘图, 比赛模式校正, 等待服务器, 听回调
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, False, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-正常, 1-起身, 2-向左扑, 3-向右扑, 4-等待

        self.kick_dir = 0 # 踢球方向
        self.reset_kick = True # 当为True时，生成一个新的随机踢球方向
        

    def think_and_send(self):
        w = self.world
        r = self.world.robot 
        my_head_pos_2d = r.loc_head_position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
        behavior = self.behavior
        PM = w.play_mode

        #--------------------------------------- 1. 决定动作

        #让一个机器人绕一个半径为3m的圆圈

        if r.unum == 1: 
            behavior.execute("Circle", 3, 0.5, 0.5)

        if PM in [w.M_BEFORE_KICKOFF, w.M_THEIR_GOAL, w.M_OUR_GOAL]: # 传送到初始位置并等待
            self.state = 0
            self.reset_kick = True
            pos = (-14,0) if r.unum == 1 else (4.9,0)
            if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or behavior.is_ready("Get_Up"):
                self.scom.commit_beam(pos, 0) # 传送到初始位置
            else:
                behavior.execute("Zero_Bent_Knees") # 等待
        elif self.state == 2: # 向左扑
            self.state = 4 if behavior.execute("Dive_Left") else 2  # 技能完成后状态改为等待
        elif self.state == 3: # 向右扑
            self.state = 4 if behavior.execute("Dive_Right") else 3 # 技能完成后状态改为等待
        elif self.state == 4: # 等待（扑救后或对方踢球时）
            pass
        elif self.state == 1 or behavior.is_ready("Get_Up"): # 如果正在起身或倒地
            self.state = 0 if behavior.execute("Get_Up") else 1 # 起身行为完成后返回正常状态
        elif PM == w.M_OUR_KICKOFF and r.unum == 1 or PM == w.M_THEIR_KICKOFF and r.unum != 1:
            self.state = 4 # 等待直到下次传送
        elif r.unum == 1: # 守门员
            y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)
            behavior.execute("Walk", (-14,y_coordinate), True, 0, True, None) # 参数: 目标, 是否绝对目标, 方向, 是否绝对方向, 距离
            if ball_2d[0] < -10: 
                self.state = 2 if ball_2d[1] > 0 else 3 # 扑救防守
        else: # 踢球者
            if PM == w.M_OUR_KICKOFF and ball_2d[0] > 5: # 检查球的位置以确保我能看到它
                if self.reset_kick: 
                    self.kick_dir = random.choice([-7.5,7.5]) 
                    self.reset_kick = False
                behavior.execute("Basic_Kick", self.kick_dir)
            else:
                behavior.execute("Zero_Bent_Knees") # 等待

        #--------------------------------------- 2. 广播
        self.radio.broadcast()

        #--------------------------------------- 3. 发送至服务器
        self.scom.commit_and_send( r.get_command() )

        #---------------------- 调试注释
        if self.enable_draw: 
            d = w.draw
            if r.unum == 1:
                d.annotation((*my_head_pos_2d, 0.8), "Goalkeeper" , d.Color.yellow, "status")
            else:
                d.annotation((*my_head_pos_2d, 0.8), "Kicker" , d.Color.yellow, "status")
                if PM == w.M_OUR_KICKOFF: # 绘制箭头以指示踢球方向
                    d.arrow(ball_2d, ball_2d + 5*M.vector_from_angle(self.kick_dir), 0.4, 3, d.Color.cyan_light, "Target")
