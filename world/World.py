from collections import deque
from cpp.ball_predictor import ball_predictor
from cpp.localization import localization
from logs.Logger import Logger
from math import atan2, pi
from math_ops.Matrix_4x4 import Matrix_4x4
from world.commons.Draw import Draw
from world.commons.Other_Robot import Other_Robot
from world.Robot import Robot
import numpy as np


class World():
    STEPTIME = 0.02    # 固定步长时间
    STEPTIME_MS = 20   # 固定步长时间（毫秒）
    VISUALSTEP = 0.04  # 固定视觉步长时间
    VISUALSTEP_MS = 40 # 固定视觉步长时间（毫秒）

    # 对我们有利的比赛模式
    M_OUR_KICKOFF = 0
    M_OUR_KICK_IN = 1
    M_OUR_CORNER_KICK = 2
    M_OUR_GOAL_KICK = 3
    M_OUR_FREE_KICK = 4
    M_OUR_PASS = 5
    M_OUR_DIR_FREE_KICK = 6
    M_OUR_GOAL = 7
    M_OUR_OFFSIDE = 8

    # 对他们有利的比赛模式
    M_THEIR_KICKOFF = 9
    M_THEIR_KICK_IN = 10
    M_THEIR_CORNER_KICK = 11
    M_THEIR_GOAL_KICK = 12
    M_THEIR_FREE_KICK = 13
    M_THEIR_PASS = 14
    M_THEIR_DIR_FREE_KICK = 15
    M_THEIR_GOAL = 16
    M_THEIR_OFFSIDE = 17

    # 中立的比赛模式
    M_BEFORE_KICKOFF = 18
    M_GAME_OVER = 19
    M_PLAY_ON = 20

    # 比赛模式组
    MG_OUR_KICK = 0
    MG_THEIR_KICK = 1
    MG_ACTIVE_BEAM = 2
    MG_PASSIVE_BEAM = 3
    MG_OTHER = 4 # 比赛进行中，比赛结束

    FLAGS_CORNERS_POS = ((-15,-10,0), (-15,+10,0), (+15,-10,0), (+15,+10,0))
    FLAGS_POSTS_POS = ((-15,-1.05,0.8),(-15,+1.05,0.8),(+15,-1.05,0.8),(+15,+1.05,0.8))

    def __init__(self, robot_type:int, team_name:str, unum:int, apply_play_mode_correction:bool, 
                 enable_draw:bool, logger:Logger, host:str) -> None:
        # 初始化世界对象的属性
        self.team_name = team_name               # 我们队伍的名称
        self.team_name_opponent : str = None     # 对手队伍的名称
        self.apply_play_mode_correction = apply_play_mode_correction # 是否根据比赛模式调整球的位置
        self.step = 0           # 接收到的模拟步数（与self.time_local_ms同步）
        self.time_server = 0.0  # 服务器指示的时间（秒）（此时间不可靠，仅用于代理之间的同步）
        self.time_local_ms = 0  # 可靠的模拟时间（毫秒），尽可能使用此时间（每收到一个TCP消息增加20ms）
        self.time_game = 0.0    # 游戏时间（秒），由服务器指示
        self.goals_scored = 0   # 我们队伍的进球数
        self.goals_conceded = 0 # 我们队伍的失球数
        self.team_side_is_left : bool = None # 如果我们的队伍在左侧，则为True（此值由世界解析器稍后更改）
        self.play_mode = None                # 足球比赛的比赛模式，由服务器提供
        self.play_mode_group = None          # 某些比赛模式具有共同特征，因此将其分组是有意义的
        self.flags_corners : dict = None     # 角旗，键=(x,y,z)，始终假设我们在左侧
        self.flags_posts : dict = None       # 球门柱，键=(x,y,z)，始终假设我们在左侧
        self.ball_rel_head_sph_pos = np.zeros(3)     # 相对于头部的球位置（球坐标）（米，度，度）
        self.ball_rel_head_cart_pos = np.zeros(3)    # 相对于头部的球位置（笛卡尔坐标）（米）
        self.ball_rel_torso_cart_pos = np.zeros(3)   # 相对于躯干的球位置（笛卡尔坐标）（米）
        self.ball_rel_torso_cart_pos_history = deque(maxlen=20) # 相对于躯干的球位置历史（最多20个旧位置的队列，间隔0.04秒，索引0为前一个位置）
        self.ball_abs_pos = np.zeros(3)              # 球的绝对位置（如果self.ball_is_visible和self.robot.loc_is_up_to_date，则最新）（米）
        self.ball_abs_pos_history = deque(maxlen=20) # 球的绝对位置历史（最多20个旧位置的队列，间隔0.04秒，索引0为前一个位置）
        self.ball_abs_pos_last_update = 0        # self.ball_abs_pos上次由视觉或无线电更新的World.time_local_ms
        self.ball_abs_vel = np.zeros(3)          # 基于self.ball_abs_pos最后两个已知值的球速度矢量（米/秒）（警告：如果球距离远，则嘈杂，使用get_ball_abs_vel代替）
        self.ball_abs_speed = 0                  # 基于self.ball_abs_pos最后两个已知值的球标量速度（米/秒）（警告：如果球距离远，则嘈杂，使用||get_ball_abs_vel||代替）
        self.ball_is_visible = False             # 如果最后一个服务器消息包含与球相关的视觉信息，则为True
        self.is_ball_abs_pos_from_vision = False # 如果ball_abs_pos源自视觉，则为True，如果源自无线电，则为False
        self.ball_last_seen = 0                  # 上次看到球的World.time_local_ms（注意：可能与self.ball_abs_pos_last_update不同）
        self.ball_cheat_abs_pos = np.zeros(3)    # 服务器提供的作弊绝对球位置（米）
        self.ball_cheat_abs_vel = np.zeros(3)    # 基于self.ball_cheat_abs_pos最后两个值的绝对速度矢量（米/秒）
        self.ball_2d_pred_pos = np.zeros((1,2))  # 当前和未来2D球位置的预测*
        self.ball_2d_pred_vel = np.zeros((1,2))  # 当前和未来2D球速度的预测*
        self.ball_2d_pred_spd = np.zeros(1)      # 当前和未来2D球线性速度的预测*
        # *间隔0.02秒，直到球停止或根据预测出界
        self.lines = np.zeros((30,6))            # 可见线的位置，相对于头部，起点+终点（球坐标）（米，度，度，米，度，度）
        self.line_count = 0                      # 可见线的数量
        self.vision_last_update = 0                                   # 上次收到视觉更新的World.time_local_ms
        self.vision_is_up_to_date = False                             # 如果最后一个服务器消息包含视觉信息，则为True
        self.teammates = [Other_Robot(i, True ) for i in range(1,12)] # 队友列表，按unum排序
        self.opponents = [Other_Robot(i, False) for i in range(1,12)] # 对手列表，按unum排序
        self.teammates[unum-1].is_self = True                         # 这个队友是自己
        self.draw = Draw(enable_draw, unum, host, 32769)              # 当前玩家的绘图对象
        self.team_draw = Draw(enable_draw, 0, host, 32769)            # 与队友共享的绘图对象
        self.logger = logger
        self.robot = Robot(unum, robot_type)

    def log(self, msg:str):
        '''
        快捷方式：
        self.logger.write(msg, True, self.step)

        参数
        ----------
        msg : str
            在模拟步数之后写入的消息
        ''' 
        self.logger.write(msg, True, self.step)

    def get_ball_rel_vel(self, history_steps:int):
        '''
        获取相对于躯干的球速度（米/秒）

        参数
        ----------
        history_steps : int
            要考虑的历史步数 [1,20]

        示例
        --------
        get_ball_rel_vel(1) 等同于 (当前相对位置 - 上一个相对位置)      / 0.04
        get_ball_rel_vel(2) 等同于 (当前相对位置 - 0.08秒前的相对位置) / 0.08
        get_ball_rel_vel(3) 等同于 (当前相对位置 - 0.12秒前的相对位置) / 0.12
        '''
        assert 1 <= history_steps <= 20, "参数 'history_steps' 必须在范围 [1,20]"

        if len(self.ball_rel_torso_cart_pos_history) == 0:
            return np.zeros(3)

        h_step = min(history_steps, len(self.ball_rel_torso_cart_pos_history))
        t = h_step * World.VISUALSTEP

        return (self.ball_rel_torso_cart_pos - self.ball_rel_torso_cart_pos_history[h_step-1]) / t

    def get_ball_abs_vel(self, history_steps:int):
        '''
        获取球的绝对速度（米/秒）

        参数
        ----------
        history_steps : int
            要考虑的历史步数 [1,20]

        示例
        --------
        get_ball_abs_vel(1) 等同于 (当前绝对位置 - 上一个绝对位置)      / 0.04
        get_ball_abs_vel(2) 等同于 (当前绝对位置 - 0.08秒前的绝对位置) / 0.08
        get_ball_abs_vel(3) 等同于 (当前绝对位置 - 0.12秒前的绝对位置) / 0.12
        '''
        assert 1 <= history_steps <= 20, "参数 'history_steps' 必须在范围 [1,20]"

        if len(self.ball_abs_pos_history) == 0:
            return np.zeros(3)

        h_step = min(history_steps, len(self.ball_abs_pos_history))
        t = h_step * World.VISUALSTEP

        return (self.ball_abs_pos - self.ball_abs_pos_history[h_step-1]) / t

    def get_predicted_ball_pos(self, max_speed):
        '''
        当预测速度等于或小于`max_speed`时，获取预测的2D球位置
        如果该位置超过预测范围，则返回最后一个可用的预测

        参数
        ----------
        max_speed : float
            返回未来位置时球将移动的最大速度
        '''
        b_sp = self.ball_2d_pred_spd
        index = len(b_sp) - max( 1, np.searchsorted(b_sp[::-1], max_speed, side='right') )
        return self.ball_2d_pred_pos[index]
    
    def get_intersection_point_with_ball(self, player_speed):
        '''
        基于`self.ball_2d_pred_pos`获取与移动球的2D交点

        参数
        ----------
        player_speed : float
            机器人追逐球时的平均速度

        返回
        -------
        2D交点 : ndarray
            与移动球的2D交点，假设机器人以`player_speed`的平均速度移动
        交点距离 : float
            当前机器人位置与交点之间的距离
        '''
        
        params = np.array([*self.robot.loc_head_position[:2], player_speed*0.02, *self.ball_2d_pred_pos.flat], np.float32)
        pred_ret  = ball_predictor.get_intersection(params)
        return pred_ret[:2], pred_ret[2]
    
    def update(self):
        r = self.robot
        PM = self.play_mode
        W = World

        # 重置变量
        r.loc_is_up_to_date = False                   
        r.loc_head_z_is_up_to_date = False

        # 更新比赛模式组
        if PM in (W.M_PLAY_ON, W.M_GAME_OVER): # 最常见的组
            self.play_mode_group = W.MG_OTHER
        elif PM in (W.M_OUR_KICKOFF, W.M_OUR_KICK_IN, W.M_OUR_CORNER_KICK, W.M_OUR_GOAL_KICK,
                    W.M_OUR_OFFSIDE, W.M_OUR_PASS, W.M_OUR_DIR_FREE_KICK, W.M_OUR_FREE_KICK):
            self.play_mode_group = W.MG_OUR_KICK
        elif PM in (W.M_THEIR_KICK_IN, W.M_THEIR_CORNER_KICK, W.M_THEIR_GOAL_KICK, W.M_THEIR_OFFSIDE,
                    W.M_THEIR_PASS, W.M_THEIR_DIR_FREE_KICK, W.M_THEIR_FREE_KICK, W.M_THEIR_KICKOFF):
            self.play_mode_group = W.MG_THEIR_KICK
        elif PM in (W.M_BEFORE_KICKOFF, W.M_THEIR_GOAL):
            self.play_mode_group = W.MG_ACTIVE_BEAM
        elif PM in (W.M_OUR_GOAL,):
            self.play_mode_group = W.MG_PASSIVE_BEAM
        elif PM is not None:
            raise ValueError(f'意外的比赛模式ID: {PM}')

        r.update_pose() # 更新正向运动学

        if self.ball_is_visible:
            # 计算相对于躯干的球位置
            self.ball_rel_torso_cart_pos = r.head_to_body_part_transform("torso",self.ball_rel_head_cart_pos)

        if self.vision_is_up_to_date: # 更新基于视觉的定位

            # 准备所有变量用于定位

            feet_contact = np.zeros(6)

            lf_contact = r.frp.get('lf', None)
            rf_contact = r.frp.get('rf', None)
            if lf_contact is not None:
                feet_contact[0:3] = Matrix_4x4( r.body_parts["lfoot"].transform ).translate( lf_contact[0:3] , True).get_translation()
            if rf_contact is not None:
                feet_contact[3:6] = Matrix_4x4( r.body_parts["rfoot"].transform ).translate( rf_contact[0:3] , True).get_translation()

            ball_pos = np.concatenate(( self.ball_rel_head_cart_pos, self.ball_cheat_abs_pos))
            
            corners_list = [[key in self.flags_corners, 1.0, *key, *self.flags_corners.get(key,(0,0,0))] for key in World.FLAGS_CORNERS_POS]

            posts_list   = [[key in self.flags_posts  , 0.0, *key, *self.flags_posts.get(  key,(0,0,0))] for key in World.FLAGS_POSTS_POS]
            all_landmarks = np.array(corners_list + posts_list, float)

            # 计算定位
            loc = localization.compute(
                r.feet_toes_are_touching['lf'],
                r.feet_toes_are_touching['rf'],
                feet_contact,
                self.ball_is_visible,
                ball_pos,
                r.cheat_abs_pos,
                all_landmarks,
                self.lines[0:self.line_count])  

            r.update_localization(loc, self.time_local_ms)

            # 更新队友列表中的自己（仅更新最有用的参数，根据需要添加）
            me = self.teammates[r.unum-1]
            me.state_last_update = r.loc_last_update
            me.state_abs_pos = r.loc_head_position
            me.state_fallen = r.loc_head_z < 0.3 # 使用与其他队友相同的判断标准 - 不如player.behavior.is_ready("Get_Up")可靠
            me.state_orientation = r.loc_torso_orientation
            me.state_ground_area = (r.loc_head_position[:2],0.2) # 对定位演示相关

            # 每次视觉周期保存最后一个球位置到历史记录（即使不最新）
            self.ball_abs_pos_history.appendleft(self.ball_abs_pos) # 来自视觉或无线电
            self.ball_rel_torso_cart_pos_history.appendleft(self.ball_rel_torso_cart_pos)

            '''
            根据视觉或比赛模式获取球位置
            来源：
            角球位置 - rcssserver3d/plugin/soccer/soccerruleaspect/soccerruleaspect.cpp:1927 (May 2022)
            球门球位置 - rcssserver3d/plugin/soccer/soccerruleaspect/soccerruleaspect.cpp:1900 (May 2022)
            '''
            ball = None
            if self.apply_play_mode_correction:
                if PM == W.M_OUR_CORNER_KICK:
                    ball = np.array([15, 5.483 if self.ball_abs_pos[1] > 0 else -5.483, 0.042], float)
                elif PM == W.M_THEIR_CORNER_KICK:
                    ball = np.array([-15, 5.483 if self.ball_abs_pos[1] > 0 else -5.483, 0.042], float)
                elif PM in [W.M_OUR_KICKOFF, W.M_THEIR_KICKOFF, W.M_OUR_GOAL, W.M_THEIR_GOAL]:
                    ball = np.array([0, 0, 0.042], float)
                elif PM == W.M_OUR_GOAL_KICK:
                    ball = np.array([-14, 0, 0.042], float)
                elif PM == W.M_THEIR_GOAL_KICK:
                    ball = np.array([14, 0, 0.042], float)

                # 如果机器人接近该位置，则丢弃硬编码的球位置（优先使用自己的视觉）
                if ball is not None and np.linalg.norm(r.loc_head_position[:2] - ball[:2]) < 1:
                    ball = None

            if ball is None and self.ball_is_visible and r.loc_is_up_to_date:
                ball = r.loc_head_to_field_transform( self.ball_rel_head_cart_pos )
                ball[2] = max(ball[2], 0.042) # 最低z = 球半径
                if PM != W.M_BEFORE_KICKOFF: # 为了兼容没有激活足球规则的测试
                    ball[:2] = np.clip(ball[:2], [-15,-10], [15,10]) # 强制球位置在场地内

            # 更新内部球位置（也由无线电更新）
            if ball is not None:
                time_diff = (self.time_local_ms - self.ball_abs_pos_last_update) / 1000
                self.ball_abs_vel = (ball - self.ball_abs_pos) / time_diff
                self.ball_abs_speed = np.linalg.norm(self.ball_abs_vel)
                self.ball_abs_pos_last_update = self.time_local_ms
                self.ball_abs_pos = ball
                self.is_ball_abs_pos_from_vision = True

            # 队友和对手的速度衰减（如果速度更新，则中立化）
            for p in self.teammates:
                p.state_filtered_velocity *= p.vel_decay
            for p in self.opponents:
                p.state_filtered_velocity *= p.vel_decay

            # 更新队友和对手
            if r.loc_is_up_to_date:
                for p in self.teammates:
                    if not p.is_self:                     # 如果队友不是自己
                        if p.is_visible:                  # 如果队友可见，执行完整更新
                            self.update_other_robot(p)
                        elif p.state_abs_pos is not None: # 否则更新其水平距离（假设最后已知位置）
                            p.state_horizontal_dist = np.linalg.norm(r.loc_head_position[:2] - p.state_abs_pos[:2])

                for p in self.opponents:
                    if p.is_visible:                  # 如果对手可见，执行完整更新
                        self.update_other_robot(p)
                    elif p.state_abs_pos is not None: # 否则更新其水平距离（假设最后已知位置）
                        p.state_horizontal_dist = np.linalg.norm(r.loc_head_position[:2] - p.state_abs_pos[:2])

        # 更新球位置/速度的预测
        if self.play_mode_group != W.MG_OTHER: # 不是'比赛进行中'或'比赛结束'，所以球必须静止
            self.ball_2d_pred_pos = self.ball_abs_pos[:2].copy().reshape(1, 2)
            self.ball_2d_pred_vel = np.zeros((1,2))
            self.ball_2d_pred_spd = np.zeros(1)

        elif self.ball_abs_pos_last_update == self.time_local_ms: # 为新的球位置进行新的预测（来自视觉或无线电）

            params = np.array([*self.ball_abs_pos[:2], *np.copy(self.get_ball_abs_vel(6)[:2])], np.float32)
            pred_ret  = ball_predictor.predict_rolling_ball(params)
            sample_no = len(pred_ret) // 5 * 2
            self.ball_2d_pred_pos = pred_ret[:sample_no].reshape(-1, 2)
            self.ball_2d_pred_vel = pred_ret[sample_no:sample_no*2].reshape(-1, 2)
            self.ball_2d_pred_spd = pred_ret[sample_no*2:]

        elif len(self.ball_2d_pred_pos) > 1: # 否则，如果有可用预测，前进到下一个预测步骤
            self.ball_2d_pred_pos = self.ball_2d_pred_pos[1:]
            self.ball_2d_pred_vel = self.ball_2d_pred_vel[1:]
            self.ball_2d_pred_spd = self.ball_2d_pred_spd[1:]

        r.update_imu(self.time_local_ms)      # 更新IMU（必须在定位之后执行）


    def update_other_robot(self,other_robot : Other_Robot):
        ''' 
        根据可见身体部位的相对位置更新其他机器人状态
        （也由无线电更新，但state_orientation除外）
        '''
        o = other_robot
        r = self.robot

        # 更新身体部位的绝对位置
        o.state_body_parts_abs_pos = o.body_parts_cart_rel_pos.copy()
        for bp, pos in o.body_parts_cart_rel_pos.items():
            # 如果看不到其他机器人但能自我定位，使用IMU可能会有益
            o.state_body_parts_abs_pos[bp] = r.loc_head_to_field_transform( pos, False )

        # 辅助变量 
        bps_apos = o.state_body_parts_abs_pos                 # 只读快捷方式
        bps_2d_apos_list = [v[:2] for v in bps_apos.values()] # 身体部位的2D绝对位置列表
        avg_2d_pt = np.average(bps_2d_apos_list, axis=0)      # 可见身体部位的2D平均位置
        head_is_visible = 'head' in bps_apos

        # 评估机器人的状态（如果头部不可见，则保持不变）
        if head_is_visible:
            o.state_fallen = bps_apos['head'][2] < 0.3

        # 如果头部可见，计算速度
        if o.state_abs_pos is not None:
            time_diff = (self.time_local_ms - o.state_last_update) / 1000
            if head_is_visible:
                # 如果最后位置是2D，我们假设z坐标没有变化，所以v.z=0
                old_p = o.state_abs_pos if len(o.state_abs_pos)==3 else np.append(o.state_abs_pos, bps_apos['head'][2])            
                velocity = (bps_apos['head'] - old_p) / time_diff
                decay = o.vel_decay # 在所有轴中中立化衰减
            else: # 如果头部不可见，我们只更新速度的x & y分量
                velocity = np.append( (avg_2d_pt - o.state_abs_pos[:2]) / time_diff, 0)
                decay = (o.vel_decay,o.vel_decay,1) # 中立化衰减（除了z轴）
            # 应用滤波器
            if np.linalg.norm(velocity - o.state_filtered_velocity) < 4: # 否则假设它被传送
                o.state_filtered_velocity /= decay # 中立化衰减
                o.state_filtered_velocity += o.vel_filter * (velocity-o.state_filtered_velocity)

        # 计算机器人的位置（最好基于头部）  
        if head_is_visible:  
            o.state_abs_pos = bps_apos['head'] # 如果头部可见，则为3D头部位置
        else:   
            o.state_abs_pos = avg_2d_pt # 可见身体部位的2D平均位置

        # 计算机器人的水平距离（头部距离，或可见身体部位的平均距离）
        o.state_horizontal_dist = np.linalg.norm(r.loc_head_position[:2] - o.state_abs_pos[:2])
        
        # 基于一对下臂或脚计算方向，或两者的平均值
        lr_vec = None
        if 'llowerarm' in bps_apos and 'rlowerarm' in bps_apos:
            lr_vec = bps_apos['rlowerarm'] - bps_apos['llowerarm']
            
        if 'lfoot' in bps_apos and 'rfoot' in bps_apos:
            if lr_vec is None:
                lr_vec = bps_apos['rfoot'] - bps_apos['lfoot']
            else:
                lr_vec = (lr_vec + (bps_apos['rfoot'] - bps_apos['lfoot'])) / 2
        
        if lr_vec is not None:
            o.state_orientation = atan2(lr_vec[1],lr_vec[0]) * 180 / pi + 90

        # 计算地面上的玩家区域投影（圆形） 
        if o.state_horizontal_dist < 4: # 如果机器人距离超过4米，我们不需要精度
            max_dist = np.max(np.linalg.norm(bps_2d_apos_list - avg_2d_pt, axis=1))
        else:
            max_dist = 0.2
        o.state_ground_area = (avg_2d_pt,max_dist)

        # 更新时间戳
        o.state_last_update = self.time_local_ms
