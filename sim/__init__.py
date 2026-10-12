"""assist_sim —— 双滚刷手持洗地机单侧半车一维助力控制仿真。

模块划分（任务书 §9）：
    plant      被控对象：单侧平动动力学、滚刷合力、静/动摩擦（平滑 + 粘滑）
    actuator   执行器：力响应一阶环、限幅、速率限制、延迟、丢帧、失能
    sensors    IMU / 轮速测量、噪声、量化、时间戳、失帧
    estimator  互补滤波 / Kalman 状态估计（禁止访问 plant 真值）
    controllers C0/C1/C2/C3 (+C4 上限参考 / C5 导纳对照)
    human      固定 F_h 波形 + 闭环虚拟用户（带反应延迟）
    metrics    统一评价口径
    scenarios  S01–S12 与人机耦合任务定义与运行器
    sweep      单因素扫描、两因素网格、Monte Carlo、Pareto
    plots      曲线、叠加比较、热图
"""

__version__ = "1.1"
