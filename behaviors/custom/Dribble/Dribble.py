from agent.Base_Agent import Base_Agent
from behaviors.custom.Dribble.Env import Env
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import numpy as np
import pickle


class Dribble():

    def __init__(self, base_agent : Base_Agent) -> None:
        # 初始化行为对象，获取基础代理的相关属性
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "RL dribble"  # 行为描述
        self.auto_head = True  # 自动头部控制
        # 根据机器人类型初始化环境对象
        self.env = Env(base_agent, 0.9 if self.world.robot.type == 3 else 1.2)

        # 根据机器人类型加载对应的模型文件
        with open(M.get_active_directory([
            "/behaviors/custom/Dribble/dribble_R0.pkl",
            "/behaviors/custom/Dribble/dribble_R1.pkl",
            "/behaviors/custom/Dribble/dribble_R2.pkl",
            "/behaviors/custom/Dribble/dribble_R3.pkl",
            "/behaviors/custom/Dribble/dribble_R4.pkl"
            ][self.world.robot.type]), 'rb') as f:
            self.model = pickle.load(f)

    def define_approach_orientation(self):
        # 定义接近方向的方法
        w = self.world
        b = w.ball_abs_pos[:2]  # 球的绝对位置
        me = w.robot.loc_head_position[:2]  # 机器人的头部位置

        self.approach_orientation = None

        MARGIN = 0.8  # 安全边际（如果球接近场地边界，则考虑接近方向）
        M90 = 90/MARGIN  # 辅助变量
        DEV = 25  # 当站在边线或底线时，接近方向偏离该线的量
        MDEV = (90+DEV)/MARGIN  # 辅助变量

        a1 = -180  # 角度范围开始（逆时针旋转）
        a2 = 180  # 角度范围结束（逆时针旋转）

        # 根据球的位置调整接近方向
        if b[1] < -10 + MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = DEV - M90 * (b[1]+10)
                a2 = 90 - DEV + M90 * (b[0]+15)
            elif b[0] > 15 - MARGIN:
                a1 = 90 + DEV - M90 * (15-b[0])
                a2 = 180 - DEV + M90 * (b[1]+10)
            else:
                a1 = DEV - MDEV * (b[1]+10)
                a2 = 180 - DEV + MDEV * (b[1]+10)
        elif b[1] > 10 - MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = -90 + DEV - M90 * (b[0]+15)
                a2 = -DEV + M90 * (10-b[1])
            elif b[0] > 15 - MARGIN:
                a1 = 180 + DEV - M90 * (10-b[1])
                a2 = 270 - DEV + M90 * (15-b[0])
            else:
                a1 = -180 + DEV - MDEV * (10-b[1])
                a2 = -DEV + MDEV * (10-b[1])
        elif b[0] < -15 + MARGIN:
            a1 = -90 + DEV - MDEV * (b[0]+15)
            a2 = 90 - DEV + MDEV * (b[0]+15)
        elif b[0] > 15 - MARGIN and abs(b[1]) > 1.2:
            a1 = 90 + DEV - MDEV * (15-b[0])
            a2 = 270 - DEV + MDEV * (15-b[0])

        cad = M.vector_angle(b - me)  # 当前接近方向

        a1 = M.normalize_deg(a1)
        a2 = M.normalize_deg(a2)

        # 检查当前接近方向是否在可接受范围内
        if a1<a2: 
            if a1 <= cad <= a2:  
                return  # 当前接近方向在可接受范围内
        else:
            if a1 <= cad or cad <= a2:
                return  # 当前接近方向在可接受范围内

        a1_diff = abs(M.normalize_deg(a1 - cad))
        a2_diff = abs(M.normalize_deg(a2 - cad))

        self.approach_orientation = a1 if a1_diff < a2_diff else a2  # 固定归一化方向


    def execute(self, reset, orientation, is_orientation_absolute, speed=1, stop=False):
        '''
        参数
        ----------
        orientation : float
            躯干的绝对或相对方向（相对于imu_torso_orientation），以度为单位
            设置为None以运球朝向对方球门（忽略is_orientation_absolute）
        is_orientation_absolute : bool
            如果方向相对于场地为True，如果相对于机器人躯干为False
        speed : float
            速度从0到1（比例不是线性的）
        stop : bool
            如果正在行走，立即返回True；如果正在运球，减速并返回True
        '''
        w = self.world
        r = self.world.robot
        me = r.loc_head_position[:2]
        b = w.ball_abs_pos[:2]
        b_rel = w.ball_rel_torso_cart_pos[:2]
        b_dist = np.linalg.norm(b-me)
        behavior = self.behavior
        reset_dribble = False
        lost_ball = (w.ball_last_seen <= w.time_local_ms - w.VISUALSTEP_MS) or np.linalg.norm(b_rel)>0.4

        if reset:
            self.phase = 0
            if behavior.previous_behavior == "Push_RL" and 0<b_rel[0]<0.25 and abs(b_rel[1])<0.07:
                self.phase = 1
                reset_dribble = True

        if self.phase == 0:  # 走向球
            reset_walk = reset and behavior.previous_behavior != "Walk" and behavior.previous_behavior != "Push_RL"  # 如果前一个行为不是行走或推球，则重置行走

            #------------------------ 1. 决定是否需要更好的接近方向（当球接近边界时）
            if reset or b_dist > 0.4:  # 在接近球后停止定义方向以避免噪音
                self.define_approach_orientation()

            #------------------------ 2A. 需要更好的接近方向（球几乎出界）
            if self.approach_orientation is not None:
                next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                    x_ori=self.approach_orientation, x_dev=-0.24, torso_ori=self.approach_orientation, safety_margin=0.4)

                if b_rel[0]<0.26 and b_rel[0]>0.18 and abs(b_rel[1])<0.04 and w.ball_is_visible:  # 准备开始运球
                    self.phase += 1
                    reset_dribble = True
                else:
                    dist = max(0.08, dist_to_final_target*0.7)
                    behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist)  # 目标, is_target_abs, ori, is_ori_abs, distance

            #------------------------ 2B. 不需要更好的接近方向但机器人看不到球
            elif w.time_local_ms - w.ball_last_seen > 200:  # 如果球未被看到，则走向绝对目标
                abs_ori = M.vector_angle( b - me )
                behavior.execute_sub_behavior("Walk", reset_walk, b, True, abs_ori, True, None)  # 目标, is_target_abs, ori, is_ori_abs, distance
                
            #------------------------ 2C. 不需要更好的接近方向且机器人可以看到球
            else:  # 走向相对目标   
                if 0.18<b_rel[0]<0.25 and abs(b_rel[1])<0.05 and w.ball_is_visible:  # 准备开始运球
                    self.phase += 1
                    reset_dribble = True
                else:
                    rel_target = b_rel+(-0.23,0)     # 相对目标是圆（中心：球，半径：0.23m）
                    rel_ori = M.vector_angle(b_rel)  # 球的相对方向，不是目标的方向！
                    dist = max(0.08, np.linalg.norm(rel_target)*0.7)  # 慢速接近
                    behavior.execute_sub_behavior("Walk", reset_walk, rel_target, False, rel_ori, False, dist)  # 目标, is_target_abs, ori, is_ori_abs, distance

            if stop:
                return True

        if self.phase == 1 and (stop or (b_dist > 0.5 and lost_ball)):  # 回到行走
            self.phase += 1
        elif self.phase == 1:  # 运球
            #------------------------ 1. 定义运球参数 
            self.env.dribble_speed = speed
        
            # 相对方向值减少以避免过度调整
            if orientation is None:
                if b[0] < 0:  # 运球到两侧
                    if b[1] > 0:
                        dribble_target = (15,5)
                    else:
                        dribble_target = (15,-5)
                else:
                    dribble_target = None  # 朝向球门
                self.env.dribble_rel_orientation = self.path_manager.get_dribble_path(optional_2d_target=dribble_target)[1]
            elif is_orientation_absolute:
                self.env.dribble_rel_orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            else:
                self.env.dribble_rel_orientation = float(orientation)  # 如果是numpy浮点数则复制

            #------------------------ 2. 执行行为
            obs = self.env.observe(reset_dribble)
            action = run_mlp(obs, self.model)   
            self.env.execute(action)
        
        # 减速运球，然后重置阶段
        if self.phase > 1:
            WIND_DOWN_STEPS = 60
            #------------------------ 1. 定义运球减速参数 
            self.env.dribble_speed = 1 - self.phase/WIND_DOWN_STEPS 
            self.env.dribble_rel_orientation = 0

            #------------------------ 2. 执行行为
            obs = self.env.observe(reset_dribble, virtual_ball=True)
            action = run_mlp(obs, self.model)   
            self.env.execute(action)

            #------------------------ 3. 重置行为
            self.phase += 1
            if self.phase >= WIND_DOWN_STEPS - 5:
                self.phase = 0
                return True

            
        return False

    def is_ready(self):
        ''' 如果当前游戏/机器人条件下该行为准备开始/继续，则返回True '''
        return True
