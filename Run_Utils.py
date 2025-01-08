def main():

    from scripts.commons.Script import Script
    script = Script() # 初始化：加载配置文件,解析参数,构建C++模块（在选择测试脚本之前警告用户不一致性)

    # 允许使用本地的StableBaselines3版本（例如：https://github.com/m-abr/Adaptive-Symmetry-Learning)
    # 将'stable-baselines3'文件夹放在此项目的父目录中
    import sys
    from os.path import dirname, abspath, join
    sys.path.insert( 0, join( dirname(dirname( abspath(__file__) )), "stable-baselines3") )

    from scripts.commons.UI import UI
    from os.path import isfile, join, realpath, dirname
    from os import listdir, getcwd
    from importlib import import_module

    _cwd = realpath( join(getcwd(), dirname(__file__)))
    gyms_path  = _cwd + "/scripts/gyms/"
    utils_path = _cwd + "/scripts/utils/"
    exclusions = ["__init__.py"]

    utils = sorted([f[:-3] for f in listdir(utils_path) if isfile(join(utils_path, f)) and f.endswith(".py") and f not in exclusions], key=lambda x: (x != "Server", x))
    gyms  = sorted([f[:-3] for f in listdir(gyms_path ) if isfile(join(gyms_path , f)) and f.endswith(".py") and f not in exclusions])

    while True:
        _, col_idx, col = UI.print_table( [utils, gyms], ["Demos & Tests & Utils","Gyms"], cols_per_title=[2,1], numbering=[True]*2, prompt='Choose script (ctrl+c to exit): ' )

        is_gym = False
        if col == 0:
            chosen = ("scripts.utils." , utils[col_idx])
        elif col == 1:
            chosen = ("scripts.gyms." , gyms[col_idx])
            is_gym = True

        cls_name = chosen[1]
        mod = import_module(chosen[0]+chosen[1])

        '''
        导入的脚本不应自动执行主代码,因为：
            - 多进程方法,如'forkserver'和'spawn',会在每个子进程中执行主代码
            - 除非重新加载,否则脚本只能被调用一次
        '''
        if not is_gym:
            ''' 
            Utils接收一个包含用户定义参数和有用方法的脚本支持对象
            每个util必须实现一个execute()方法,该方法可能返回也可能不返回
            '''
            from world.commons.Draw import Draw
            from agent.Base_Agent import Base_Agent
            obj = getattr(mod,cls_name)(script)
            try:
                obj.execute() # Util可能正常返回或通过KeyboardInterrupt中断
            except KeyboardInterrupt:
                print("\nctrl+c pressed, returning...\n")
            Draw.clear_all()            # 清除所有绘图
            Base_Agent.terminate_all()  # 关闭所有服务器套接字 + 监控套接字
            script.players = []         # 清除通过批处理命令创建的玩家列表
            del obj

        else:
            ''' 
            Gyms必须实现一个Train类,该类使用用户定义的参数进行初始化并实现：
                train() - 运行优化并保存新模型的方法
                test(folder_dir, folder_name, model_file) - 加载现有模型并测试它的方法
            '''
            from scripts.commons.Train_Base import Train_Base

            print("\n在使用GYMS之前,请确保所有服务器参数设置正确")
            print("（同步模式应为'On',实时模式应为'Off',作弊模式应为'On',...)")
            print("要更改这些参数,请返回上一菜单,并选择Server\n")
            print("此外,GYMS会启动自己的服务器,因此不要手动运行任何服务器")
            
            while True:
                try:
                    idx = UI.print_table([["Train","Test","Retrain"]], numbering=[True], prompt='Choose option (ctrl+c to return): ')[0]
                except KeyboardInterrupt:
                    print()
                    break

                if idx == 0:
                    mod.Train(script).train(dict())
                else:
                    model_info = Train_Base.prompt_user_for_model()
                    if model_info is not None and idx == 1:
                        mod.Train(script).test(model_info)
                    elif model_info is not None:
                        mod.Train(script).train(model_info)


# 允许子进程绕过此文件
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nctrl+c pressed, exiting...")
        exit()
