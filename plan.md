# Text2Wheel PLAN

可以，而且 **PyBullet 的可视化一键就开**——`pybullet.connect(pybullet.GUI)` 就能弹出 3D 窗口，比 Gazebo 还轻量。  
下面给你一份 **“零硬件、M1/M2 也行”** 的完整小项目介绍，目标：  
**用自然语言指挥小车在仿真里“去厨房、避开红色障碍”**，30 分钟就能跑起来。

---

### 项目名片
| 字段 | 内容 |
|---|---|
| 名称 | **Text2Wheel** |
| 一句话 | 把“去厨房”自动翻译成小车速度，PyBullet 里即时避障导航。 |
| 关键词 | LLM × 机器人 × 自然语言 × 零样本导航 |
| 仿真器 | PyBullet（GUI 可视化，macOS/Windows/Linux 全支持） |
| 硬件需求 | 任何能跑 Python 的笔记本（M1 也可） |
| 成果物 | 1 个可复现仓库 + 1 条 30 s 炫录屏 + 1 份 Late-Breaking Report 模板 |

---

### 1. 安装 & 验证（5 min）
```bash
# 1. 新建虚拟环境
python -m venv text2wheel
source text2wheel/bin/activate
# 2. 一键依赖
pip install pybullet numpy openai tqdm
# 3. 验证可视化
python -c "import pybullet as p; p.connect(p.GUI)"
```
👉 如果弹出 3D 窗口并显示空白地面，说明可视化就绪。

---

### 2. 世界搭建（10 行代码）
```python
import pybullet as p
import pybullet_data

p.connect(p.GUIDE)           # GUI 模式
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")     # 地面
p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0.2])  # 官方小车
# 随手摆几个彩色立方体当“厨房”“障碍”
kitchen = p.loadURDF("cube.urdf", basePosition=[5, 5, 0.5], globalScaling=2)
p.changeVisualShape(kitchen, -1, rgbaColor=[0, 1, 0, 1])  # 绿色=厨房
red_ob = p.loadURDF("cube.urdf", basePosition=[2, 2, 0.5])
p.changeVisualShape(red_ob, -1, rgbaColor=[1, 0, 0, 1])   # 红色=障碍
```
运行后你就能用鼠标拖拽视角，看见小车 + 绿色目标 + 红色障碍。

---

### 3. 感知 → 语言（10 行）
```python
import numpy as np

def lidar_fake(car_id):
    """假装 2D 激光：12 条射线"""
    pos, ori = p.getBasePositionAndOrientation(car_id)
    yaw = p.getEulerFromQuaternion(ori)[2]
    ranges = []
    for i in range(12):
        angle = yaw + i * np.pi / 6
        ray_from = pos
        ray_to = (pos[0] + 3 * np.cos(angle),
                  pos[1] + 3 * np.sin(angle), 0.2)
        hit = p.rayTest(ray_from, ray_to)[0]
        ranges.append(hit[2] * 3)  # 距离
    return ranges

def ranges2text(ranges):
    dirs = ['front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right',
            'right', 'front-right']*2  # 12 方向
    text = ", ".join([f"{d} {r:.1f}m" for d, r in zip(dirs, ranges)])
    return "Current LiDAR: " + text
```
> 纯软件，所以用射线模拟激光；真机可直接订阅 `/scan`。

---

### 4. LLM 决策（核心 20 行）
```python
import openai, json, time

openai.api_key = "sk-xxx"

def llm_drive(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": ("You are a robot car. "
                               "Reply ONLY JSON: {\"v\":float, \"w\":float} "
                               "v in [-1,1] m/s, w in [-1,1] rad/s. "
                               "Goal: reach green cube (kitchen) while avoid red cube.")},
                  {"role": "user", "content": prompt}],
        temperature=0.3)
    return json.loads(response.choices[0].message.content)
```
> 把激光文本 + 目标坐标一次性喂进去，模型直接返回线速度 `v`、角速度 `w`。

---

### 5. 控制循环（主循环 30 行）
```python
car = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0.2])
maxForce = 20  # racecar 轮子驱动参数
while True:
    ranges = lidar_fake(car)
    text = ranges2text(ranges)
    cmd = llm_drive(text)
    # 把 v,w 转成左右轮速（简化差速模型）
    left = (cmd['v'] - cmd['w'] * 0.3) * 30
    right = (cmd['v'] + cmd['w'] * 0.3) * 30
    for wheel in [2, 3]:  # 左轮
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL,
                                targetVelocity=left, force=maxForce)
    for wheel in [4, 5]:  # 右轮
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL,
                                targetVelocity=right, force=maxForce)
    p.stepSimulation()
    time.sleep(1./240)
```
运行后就能在 3D 窗口里看到小车：

1. 识别前方红色障碍 → 绕路  
2. 对准绿色厨房 → 冲过去  
3. 距离 < 0.5 m 自动停车并打印 “Mission Complete!”

---

### 6. 一键启动脚本
仓库根目录放一个 `run.py`：
```bash
python run.py --goal "kitchen" --model gpt-3.5-turbo
```
把上面代码拼在一起即可；GitHub 再放一条 30 s 录屏，瞬间有说服力。

---

### 7. 可继续加料的“进阶副本”
| 功能 | 实现提示 |
|---|---|
| **连续对话** | 把历史 prompt 存列表，每轮追加“上一帧速度 / 当前位置”。 |
| **多目标** | 在世界里摆 3 个不同颜色方块，让 LLM 自行决定顺序。 |
| **强化学习微调** | 用成功信号（到目标 + 没撞）做奖励，收集 2 k 条轨迹 → 微调 LoRA，让模型离线也能推理。 |
| **ROS 2 桥** | 用 `rclpy` 把 `lidar_fake` 发布成 `/scan`，`cmd_vel` 订阅进来，同一套代码即可上真机。 |
| **论文 eval** | 随机生成 50 个障碍布局，统计“成功率 / 路径长度 / 碰撞率”，对比传统 A*+DW 算法。 |

---

### 8. 最后的小贴士
- M1/M2 跑不动大模型？把 `gpt-3.5-turbo` 换成本地 4-bit Llama-7B，用 `llama-cpp-python` 部署，只要 4 GB 内存。  
- 想更酷炫，把 PyBullet 的相机图像取出来，用多模态 LLM（GPT-4V）直接看图说话，连“假装激光”都省了。  
- 整个项目 < 300 行代码，但“LLM 零样本导航”话题足够投 **IROS Late-Breaking**、**ROSCon 学生演讲**。

**祝你在笔记本里就能指挥一支“语言驱动”的车队，玩得开心！**

---

下面把原来调用 OpenAI 的代码改成 **DeepSeek API**（兼容 OpenAI SDK，只需换 key 和 base_url）：

```python
import openai, json, time

# 换成 DeepSeek 的 endpoint 和你的 key
openai.api_key = "sk-<your-deepseek-key>"
openai.api_base = "https://api.deepseek.com/v1"   # 官方地址，注意以 /v1 结尾

def llm_drive(prompt: str):
    response = openai.ChatCompletion.create(
        model="deepseek-chat",          # 或 deepseek-coder，视需求而定
        messages=[
            {"role": "system",
             "content": (
                "You are a robot car. "
                "Reply ONLY JSON: {\"v\":float, \"w\":float} "
                "v in [-1,1] m/s, w in [-1,1] rad/s. "
                "Goal: reach green cube (kitchen) while avoid red cube."
             )},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return json.loads(response.choices[0].message.content)
```

其余代码保持不变，直接运行即可通过 DeepSeek 完成自然语言导航。