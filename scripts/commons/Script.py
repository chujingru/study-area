from os import path, listdir, getcwd, cpu_count
from os.path import join, realpath, dirname, isfile, isdir, getmtime
from scripts.commons.UI import UI
import __main__
import argparse,json,sys
import pickle
import subprocess


class Script():
    ROOT_DIR = path.dirname(path.dirname(realpath( join(getcwd(), dirname(__file__))) )) # 项目根目录

    def __init__(self, cpp_builder_unum=0) -> None:

        '''
        参数规范
        -----------------------
        - 要添加新参数，请编辑下面的信息
        - 更改信息后，必须手动删除config.json文件
        - 在其他模块中，可以通过它们的1字母ID访问这些参数
        '''
        # 参数列表：1字母ID, 描述, 硬编码默认值
        self.options = {'i': ('服务器主机名/IP', 'localhost'),
                        'p': ('代理端口',         '3100'),
                        'm': ('监控端口',       '3200'),
                        't': ('团队名称',          'FCPortugal'),
                        'u': ('球衣号码',     '1'),
                        'r': ('机器人类型',         '1'),
                        'P': ('点球大战',   '0'),
                        'F': ('magmaFatProxy',      '0'),
                        'D': ('调试模式',         '1')}

        # 参数列表：1字母ID, 数据类型, 可选值      
        self.op_types = {'i': (str, None),
                         'p': (int, None),
                         'm': (int, None),
                         't': (str, None),
                         'u': (int, range(1,12)),
                         'r': (int, [0,1,2,3,4]),
                         'P': (int, [0,1]),
                         'F': (int, [0,1]),
                         'D': (int, [0,1])}
            
        '''
        参数规范结束
        '''

        self.read_or_create_config()

        # 调整帮助文本位置
        formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
        parser = argparse.ArgumentParser(formatter_class=formatter)

        o = self.options
        t = self.op_types

        for id in self.options: # 为了美观原因缩短metavar
            parser.add_argument(f"-{id}", help=f"{o[id][0]:30}[{o[id][1]:20}]", type=t[id][0], nargs='?', default=o[id][1], metavar='X', choices=t[id][1])
        
        self.args = parser.parse_args()

        if getattr(sys, 'frozen', False): # 从二进制运行时禁用调试模式
            self.args.D = 0

        self.players = [] # 创建的代理列表

        Script.build_cpp_modules(exit_on_build = (cpp_builder_unum != 0 and cpp_builder_unum != self.args.u))

        if self.args.D:
            try:
                print(f"\n注意: 运行帮助请执行 \"python {__main__.__file__} -h\"")
            except:
                pass

            columns = [[],[],[]]
            for key, value in vars(self.args).items():
                columns[0].append(o[key][0])
                columns[1].append(o[key][1])
                columns[2].append(value)

            UI.print_table(columns, ["参数","默认值 /config.json","激活值"], alignment=["<","^","^"])


    def read_or_create_config(self) -> None:

        if not path.isfile('config.json'):       # 如果文件不存在，保存硬编码的默认值
            with open("config.json", "w") as f:
                json.dump(self.options, f, indent=4)
        else:                                    # 加载用户定义的值（可以被命令行参数覆盖）
            if path.getsize("config.json") == 0: # 等待可能的写操作，当启动多个代理时
                from time import sleep
                sleep(1)
            if path.getsize("config.json") == 0: # 1秒后仍然为空则中止
                print("中止: 'config.json' 是空的。手动验证并删除如果仍然为空。")
                exit()
                
            with open("config.json", "r") as f:
                self.options = json.loads(f.read())


    @staticmethod
    def build_cpp_modules(special_environment_prefix=[], exit_on_build=False):
        '''
        在文件夹 /cpp 中构建C++模块，使用Pybind11
        
        参数
        ----------
        special_environment_prefix : `list`
            在所需环境中运行给定命令的命令前缀
            用于为不同版本的Python解释器编译C++模块（而不是默认版本）
            Conda环境示例: ['conda', 'run', '-n', 'myEnv']
            如果为空，则使用默认的Python解释器作为编译目标
        exit_on_build : bool
            如果有东西需要构建则退出（这样每队只有一个代理构建c++模块）
        '''
        cpp_path = Script.ROOT_DIR + "/cpp/"
        exclusions = ["__pycache__"]

        cpp_modules = [d for d in listdir(cpp_path) if isdir(join(cpp_path, d)) and d not in exclusions]

        if not cpp_modules: return # 没有模块需要构建

        python_cmd = f"python{sys.version_info.major}.{sys.version_info.minor}" # "python3" 可能会选择错误的版本，这可以防止

        def init():
            print("--------------------------\nC++ 模块:",cpp_modules)

            try:
                process = subprocess.Popen(special_environment_prefix+[python_cmd, "-m", "pybind11", "--includes"], stdout=subprocess.PIPE)
                (includes, err) = process.communicate()
                process.wait()
            except:
                print(f"执行子程序时出错: '{python_cmd} -m pybind11 --includes'")
                exit()

            includes = includes.decode().rstrip() # 去掉尾随的换行符（和其他空白字符）
            print("使用 Pybind11 包含路径: '",includes,"'",sep="")
            return includes

        nproc = str(cpu_count())
        zero_modules = True

        for module in cpp_modules:
            module_path = join(cpp_path, module)

            # 如果模块没有Makefile，则跳过（典型分发情况）
            if not isfile(join(module_path, "Makefile")):
                continue

            # 在某些条件下跳过模块
            if isfile(join(module_path, module+".so")) and isfile(join(module_path, module+".c_info")):
                with open(join(module_path, module+".c_info"), 'rb') as f:
                    info = pickle.load(f)
                if info == python_cmd:
                    code_mod_time = max(getmtime(join(module_path, f)) for f in listdir(module_path) if f.endswith(".cpp") or f.endswith(".h"))
                    bin_mod_time = getmtime(join(module_path, module+".so"))
                    if bin_mod_time + 30 > code_mod_time: # 倾向于不构建，留出30秒的余量（场景：我们解压fcpy项目，包括二进制文件，修改时间都很相似）
                        continue

            # 初始化: 打印内容 & 获取Pybind11包含路径
            if zero_modules:
                if exit_on_build:
                    print("有C++模块需要构建。这个代理不允许构建。中止。")
                    exit()
                zero_modules = False
                includes = init()

            # 构建模块
            print(f'{f"构建: {module}... ":40}',end='',flush=True)
            process = subprocess.Popen(['make', '-j'+nproc, 'PYBIND_INCLUDES='+includes], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=module_path)
            (output, err) = process.communicate()
            exit_code = process.wait()
            if exit_code == 0:
                print("成功!")
                with open(join(module_path, module+".c_info"),"wb") as f: # 保存Python版本
                    pickle.dump(python_cmd, f, protocol=4) # 协议4向后兼容Python 3.4
            else:
                print("中止! 构建错误:")
                print(output.decode(), err.decode())
                exit()     

        if not zero_modules:
            print("所有模块构建成功!\n--------------------------")


    def batch_create(self, agent_cls, args_per_player):    
        ''' 创建一批代理 '''

        for a in args_per_player:
            self.players.append( agent_cls(*a) )

    def batch_execute_agent(self, index : slice = slice(None)):  
        ''' 
        正常执行代理（包括提交和发送）

        参数
        ----------
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''   
        for p in self.players[index]:
            p.think_and_send()

    def batch_execute_behavior(self, behavior, index : slice = slice(None)):
        '''
        执行行为

        参数
        ----------
        behavior : str
            要执行的行为名称
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''
        for p in self.players[index]:
            p.behavior.execute(behavior)

    def batch_commit_and_send(self, index : slice = slice(None)):
        '''
        提交并发送数据到服务器

        参数
        ----------
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''
        for p in self.players[index]:
            p.scom.commit_and_send( p.world.robot.get_command() ) 

    def batch_receive(self, index : slice = slice(None), update=True):
        ''' 
        等待服务器消息

        参数
        ----------
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        update : bool
            根据从服务器接收到的信息更新世界状态
            如果为False，代理将无法感知自己和周围环境
            这在演示中减少虚拟代理的CPU资源时很有用
        '''
        for p in self.players[index]:
            p.scom.receive(update)

    def batch_commit_beam(self, pos2d_and_rotation, index : slice = slice(None)):
        '''
        将所有玩家传送到2D位置并指定旋转

        参数
        ----------
        pos2d_and_rotation : `list`
            2D位置和旋转的可迭代对象，例如 [(0,0,45),(-5,0,90)]
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''        
        for p, pos_rot in zip(self.players[index], pos2d_and_rotation): 
            p.scom.commit_beam(pos_rot[0:2],pos_rot[2])

    def batch_unofficial_beam(self, pos3d_and_rotation, index : slice = slice(None)):
        '''
        将所有玩家传送到3D位置并指定旋转

        参数
        ----------
        pos3d_and_rotation : `list`
            3D位置和旋转的可迭代对象，例如 [(0,0,0.5,45),(-5,0,0.5,90)]
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''        
        for p, pos_rot in zip(self.players[index], pos3d_and_rotation): 
            p.scom.unofficial_beam(pos_rot[0:3],pos_rot[3])

    def batch_terminate(self, index : slice = slice(None)):
        '''
        关闭所有连接到代理端口的套接字
        对于代理生命周期直到应用程序结束的脚本，这不是必需的

        参数
        ----------
        index : slice
            代理的子集
            （例如 index=slice(1,2) 将选择第二个代理）
            （例如 index=slice(1,3) 将选择第二个和第三个代理）
            默认情况下，选择所有代理
        '''
        for p in self.players[index]:
            p.terminate()
        del self.players[index] # 删除选择的代理
