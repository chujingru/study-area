from scripts.commons.Script import Script
script = Script(cpp_builder_unum=1) # 初始化:加载配置文件,解析参数,构建C++模块
a = script.args

if a.P: # 点球大战
    from agent.Agent_Penalty import Agent
else: # 普通代理
    from agent.Agent import Agent

# 参数:服务器IP,代理端口,监控端口,统一编号,队伍名称,启用日志,启用绘图,等待服务器,是否为magmaFatProxy
if a.D: # 调试模式
    player = Agent(a.i, a.p, a.m, a.u, a.t, True, True, False, a.F)
else:
    player = Agent(a.i, a.p, None, a.u, a.t, False, False, False, a.F)

while True:
    player.think_and_send() # 思考并发送指令
    player.scom.receive() # 接收信息

'''
额外注释:

script = Script(cpp_builder_unum=1):初始化Script对象,指定统一编号为1,进行必要的配置加载和参数解析。
a = script.args:获取解析后的参数。
if a.P::如果参数a.P为真,表示进行点球大战,导入Agent_Penalty模块:否则导入普通的Agent模块。
if a.D::如果参数a.D为真,表示处于调试模式,创建Agent实例时启用日志和绘图,并指定监控端口；否则不启用日志和绘图,监控端口为None。
player = Agent(...):根据条件创建Agent实例,参数包括服务器IP、代理端口、监控端口、统一编号、队伍名称、启用日志、启用绘图、是否等待服务器、是否为magmaFatProxy。
while True::无限循环,用于持续执行代理的思考和发送指令,并接收信息。
player.think_and_send():代理进行思考并发送指令。
player.scom.receive():代理接收信息。
'''
