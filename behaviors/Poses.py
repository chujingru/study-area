''' 
Pose - 指定关节的角度（以度为单位）
注意：没有脚趾的机器人会忽略脚趾位置

姿态可以控制所有关节或仅由“indices”变量定义的子组
'''

import numpy as np
from world.World import World


class Poses():
    def __init__(self, world : World) -> None:
        self.world = world
        self.tolerance = 0.05 # 角度误差容限，用于判断行为是否完成

        '''
        添加新姿态的说明：
        1. 在下面的字典中添加一个新的条目，使用唯一的姿态名称
        2. 就这样
        '''
        self.poses = {
            "Zero":(
                "中性姿态，包括头部", # 描述
                False, # 禁用自动头部方向
                np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13, 14, 15,16,17,18,19,20,21,22,23]), # 索引
                np.array([0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0,-90,-90, 0, 0,90,90, 0, 0, 0, 0])  # 值
            ),
            "Zero_Legacy":(
                "中性姿态，包括头部，肘部会导致碰撞（旧版）", # 描述
                False, # 禁用自动头部方向
                np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13, 14, 15,16,17,18,19,20,21,22,23]), # 索引
                np.array([0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0,-90,-90, 0, 0, 0, 0, 0, 0, 0, 0])  # 值
            ),
            "Zero_Bent_Knees":(
                "中性姿态，包括头部，弯曲膝盖", # 描述
                False, # 禁用自动头部方向
                np.array([0,1,2,3,4,5,6,  7,  8,  9,10,11,12,13, 14, 15,16,17,18,19,20,21,22,23]), # 索引
                np.array([0,0,0,0,0,0,30,30,-60,-60,30,30, 0, 0,-90,-90, 0, 0,90,90, 0, 0, 0, 0])  # 值
            ),
            "Zero_Bent_Knees_Auto_Head":(
                "中性姿态，自动头部方向，弯曲膝盖", # 描述
                True, # 启用自动头部方向
                np.array([2,3,4,5,6,  7,  8,  9,10,11,12,13, 14, 15,16,17,18,19,20,21,22,23]), # 索引
                np.array([0,0,0,0,30,30,-60,-60,30,30, 0, 0,-90,-90, 0, 0,90,90, 0, 0, 0, 0])  # 值
            ),
            "Fall_Back":(
                "倾斜脚部向后倒", # 描述
                True, # 启用自动头部方向
                np.array([ 10, 11]), # 索引
                np.array([-20,-20])  # 值
            ),
            "Fall_Front":(
                "倾斜脚部向前倒", # 描述
                True, # 启用自动头部方向
                np.array([10,11]), # 索引
                np.array([45,45])  # 值
            ),
            "Fall_Left":(
                "倾斜腿部向左倒", # 描述
                True, # 启用自动头部方向
                np.array([  4, 5]), # 索引
                np.array([-20,20])  # 值
            ),
            "Fall_Right":(
                "倾斜腿部向右倒", # 描述
                True, # 启用自动头部方向
                np.array([ 4,  5]), # 索引
                np.array([20,-20])  # 值
            ),
        }

        # 如果不是机器人4，则移除脚趾
        if world.robot.type != 4:
            for key, val in self.poses.items():
                idxs = np.where(val[2] >= 22)[0] # 搜索关节22和23
                if len(idxs) > 0:
                    self.poses[key] = (val[0], val[1], np.delete(val[2],idxs), np.delete(val[3],idxs)) # 移除这些关节


    def get_behaviors_callbacks(self):
        ''' 
        返回每个姿态行为的回调函数（内部使用）
        
        实现说明：
        --------------------
        使用虚拟默认参数，因为lambda表达式会记住作用域和变量名。
        在循环中，作用域和变量名不会改变。
        然而，默认参数在定义lambda时被评估。
        '''
        return {key: (val[0], val[1], lambda reset, key=key: self.execute(key), lambda: True) for key, val in self.poses.items()}

    def execute(self,name) -> bool:
        _, _, indices, values = self.poses[name]
        remaining_steps = self.world.robot.set_joints_target_position_direct(indices,values,True,tolerance=self.tolerance)
        return bool(remaining_steps == -1)