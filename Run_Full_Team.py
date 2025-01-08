from scripts.commons.Script import Script
script = Script()  # 初始化：加载配置文件，解析参数，构建C++模块
a = script.args

from agent.Agent import Agent

# 参数：服务器IP，代理端口，监控端口，统一编号，团队名称，启用日志，启用绘图
team_args = ((a.i, a.p, a.m, u, a.t, True, True) for u in range(1,12))
script.batch_create(Agent, team_args)  # 批量创建代理

while True:
    script.batch_execute_agent()  # 批量执行代理
    script.batch_receive()  # 批量接收数据
