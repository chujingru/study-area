from scripts.commons.Script import Script
script = Script() # 初始化：加载配置文件，解析参数，构建C++模块
a = script.args

from agent.Agent import Agent

# 参数：服务器IP，代理端口，监控端口，统一编号，队伍名称，启用日志，启用绘图
script.batch_create(Agent, ((a.i, a.p, a.m, a.u, a.t, True, True),)) # 主队的一个玩家
script.batch_create(Agent, ((a.i, a.p, a.m, a.u, "Opponent", True, True),)) # 客队的一个玩家

while True:
    script.batch_execute_agent() # 批量执行代理
    script.batch_receive() # 批量接收信息