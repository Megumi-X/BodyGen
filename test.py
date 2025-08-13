import time
from single_opt.envs.ant import AntSingleEnv  # 确保引入你自己的环境类

def inspect_static(env_name='ant_single_tunnel_3'):
    """
    打开环境并暂停，让你用鼠标自由查看。
    """
    print("Initializing environment for static inspection...")
    env = AntSingleEnv(env_name=env_name)
    env.reset()
    
    print("\n--- 🖱️ 交互式窗口已打开 ---")
    print("你可以使用鼠标进行以下操作:")
    print("  - 左键拖动: 旋转视角")
    print("  - 右键拖动: 平移视角")
    print("  - 滚轮滚动: 缩放视角")
    print("按住 Ctrl 并单击身体部位可以打印其信息。")
    print("关闭渲染窗口或在终端按 Ctrl+C 即可退出程序。")

    # 循环渲染，直到窗口被关闭
    while True:
        try:
            env.render(mode='human')
            time.sleep(0.01) # 稍微暂停，降低CPU占用
        except Exception:
            print("渲染窗口已关闭，程序退出。")
            break
            
    env.close()

if __name__ == "__main__":
    inspect_static()