from abc import abstractmethod
from behaviors.Behavior import Behavior
from communication.Radio import Radio
from communication.Server_Comm import Server_Comm
from communication.World_Parser import World_Parser
from logs.Logger import Logger
from math_ops.Inverse_Kinematics import Inverse_Kinematics
from world.commons.Path_Manager import Path_Manager
from world.World import World

class Base_Agent():
    all_agents = []  # 所有代理的列表

    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int, robot_type:int, team_name:str, enable_log:bool=True,
                  enable_draw:bool=True, apply_play_mode_correction:bool=True, wait_for_server:bool=True, hear_callback=None) -> None:
        """
        初始化代理对象
        :param host: 服务器主机地址
        :param agent_port: 代理端口
        :param monitor_port: 监控端口
        :param unum: 代理编号
        :param robot_type: 机器人类型
        :param team_name: 队伍名称
        :param enable_log: 是否启用日志
        :param enable_draw: 是否启用绘图
        :param apply_play_mode_correction: 是否应用比赛模式校正
        :param wait_for_server: 是否等待服务器
        :param hear_callback: 消息回调函数
        """
        self.radio = None  # 在Server_Comm实例化期间可能会调用hear_message
        self.logger = Logger(enable_log, f"{team_name}_{unum}")  # 初始化日志记录器
        self.world = World(robot_type, team_name, unum, apply_play_mode_correction, enable_draw, self.logger, host)  # 初始化世界模型
        self.world_parser = World_Parser(self.world, self.hear_message if hear_callback is None else hear_callback)  # 初始化世界解析器
        self.scom = Server_Comm(host,agent_port,monitor_port,unum,robot_type,team_name,self.world_parser,self.world,Base_Agent.all_agents,wait_for_server)  # 初始化服务器通信
        self.inv_kinematics = Inverse_Kinematics(self.world.robot)  # 初始化逆运动学
        self.behavior = Behavior(self)  # 初始化行为
        self.path_manager = Path_Manager(self.world)  # 初始化路径管理器
        self.radio = Radio(self.world, self.scom.commit_announcement)  # 初始化无线电通信
        self.behavior.create_behaviors()  # 创建行为
        Base_Agent.all_agents.append(self)  # 将当前代理添加到所有代理列表中

    @abstractmethod
    def think_and_send(self):
        """
        抽象方法：思考并发送指令
        """
        pass

    def hear_message(self, msg:bytearray, direction, timestamp:float) -> None:
        """
        接收消息
        :param msg: 消息内容
        :param direction: 消息方向
        :param timestamp: 时间戳
        """
        if direction != "self" and self.radio is not None:
            self.radio.receive(msg)  # 接收消息

    def terminate(self):
        """
        终止代理
        """
        # 如果这是该线程上的最后一个代理，则关闭共享监控套接字
        self.scom.close(close_monitor_socket=(len(Base_Agent.all_agents)==1))
        Base_Agent.all_agents.remove(self)  # 从所有代理列表中移除当前代理

    @staticmethod
    def terminate_all():
        """
        终止所有代理
        """
        for o in Base_Agent.all_agents:
            o.scom.close(True)  # 关闭共享监控套接字（如果存在）
        Base_Agent.all_agents = []  # 清空所有代理列表
