from math_ops.Math_Ops import Math_Ops as M
from os import listdir
from os.path import isfile, join
from world.World import World
import numpy as np
import xml.etree.ElementTree as xmlp

class Slot_Engine():

    def __init__(self, world : World) -> None:
        self.world = world
        self.state_slot_number = 0
        self.state_slot_start_time = 0
        self.state_slot_start_angles = None
        self.state_init_zero = True

        # ------------- 解析插槽行为

        dir = M.get_active_directory("/behaviors/slot/")

        common_dir = f"{dir}common/"
        files =  [(f,join(common_dir, f)) for f in listdir(common_dir) if isfile(join(common_dir, f)) and f.endswith(".xml")]
        robot_dir = f"{dir}r{world.robot.type}"
        files += [(f,join(robot_dir, f)) for f in listdir(robot_dir) if isfile(join(robot_dir, f)) and f.endswith(".xml")]

        self.behaviors = dict()
        self.descriptions = dict()
        self.auto_head_flags = dict()

        for fname, file in files:
            robot_xml_root = xmlp.parse(file).getroot()
            slots = []
            bname = fname[:-4] # 移除扩展名 ".xml"

            for xml_slot in robot_xml_root:
                assert xml_slot.tag == 'slot', f"在插槽行为 {fname} 中发现意外的 XML 元素: '{xml_slot.tag}'"
                indices, angles = [],[]
                
                for action in xml_slot:
                    indices.append(  int(action.attrib['id'])    )
                    angles.append( float(action.attrib['angle']) )

                delta_ms = float(xml_slot.attrib['delta']) * 1000
                assert delta_ms > 0, f"在插槽行为 {fname} 中发现无效的 delta <=0"
                slots.append((delta_ms, indices, angles))

            assert bname not in self.behaviors, f"发现至少两个同名的插槽行为: {fname}"

            self.descriptions[bname] = robot_xml_root.attrib["description"] if "description" in robot_xml_root.attrib else bname
            self.auto_head_flags[bname] = (robot_xml_root.attrib["auto_head"] == "1")
            self.behaviors[bname] = slots


    def get_behaviors_callbacks(self):
        ''' 
        返回每个插槽行为的回调函数（内部使用） 

        实现说明：
        --------------------
        使用虚拟默认参数，因为 lambda 表达式会记住作用域和变量名。
        在循环中，作用域和变量名不会改变。
        然而，默认参数在定义 lambda 时被评估。
        '''
        return {key: (self.descriptions[key],self.auto_head_flags[key],
                lambda reset,key=key: self.execute(key,reset), lambda key=key: self.is_ready(key)) for key in self.behaviors}


    def is_ready(self,name) -> bool:
        return True


    def reset(self, name):
        ''' 初始化/重置插槽行为 '''

        self.state_slot_number = 0
        self.state_slot_start_time_ms = self.world.time_local_ms
        self.state_slot_start_angles = np.copy(self.world.robot.joints_position)
        assert name in self.behaviors, f"请求的插槽行为不存在: {name}"


    def execute(self,name,reset) -> bool:
        ''' 执行一步 '''

        if reset: self.reset(name)

        elapsed_ms = self.world.time_local_ms - self.state_slot_start_time_ms
        delta_ms, indices, angles = self.behaviors[name][self.state_slot_number]

        # 检查插槽进度
        if elapsed_ms >= delta_ms:
            self.state_slot_start_angles[indices] = angles # 根据上次目标更新起始角度
             
            # 防止两种罕见情况：
            # 1 - 在行为完成后且 reset==False 时调用此函数
            # 2 - 我们在最后一个插槽，同步模式未激活，并且我们丢失了最后一步
            if self.state_slot_number+1 == len(self.behaviors[name]):
                return True # 因此，返回值表示行为已完成，直到通过参数发送重置

            self.state_slot_number += 1
            elapsed_ms = 0
            self.state_slot_start_time_ms = self.world.time_local_ms
            delta_ms, indices, angles = self.behaviors[name][self.state_slot_number]

        # 执行
        progress = (elapsed_ms+20) / delta_ms
        target = (angles - self.state_slot_start_angles[indices]) * progress + self.state_slot_start_angles[indices]
        self.world.robot.set_joints_target_position_direct(indices,target,False)

        # 如果完成（这是最后一步）返回 True
        return bool(elapsed_ms+20 >= delta_ms and self.state_slot_number + 1 == len(self.behaviors[name])) # 如果下一步（现在+20ms）超出范围则为 True