# Alfvén Wave Perturbation for magnetoAlfven PGen

## 物理动机

在偶极磁层中，希望模拟沿磁力线传播的 Alfvén 波扰动。系统先在无扰动下演化到稳态（t≈50），然后在一段指定的纬度带内施加 Alfvénic 扰动。

## 扰动形式

扰动通过修改本地旋转频率 `Omega` 实现。原本刚性旋转的电场：

```
ex1 = Omega * B_theta * r * sin(theta)
ex2 = -Omega * B_r * r * sin(theta)
```

由于 `Omega = 0`（无整体旋转），扰动直接由 `dOmega` 控制：

```
modified_Omega = dOmega * f_theta(theta) * f_time(t)
```

### 空间包络：高斯型，定位于特定纬度带

```
f_theta(theta) = exp( -4.5 * u² )

其中 u = (2*theta - (theta1 + theta2)) / (theta1 - theta2)
```

- 峰值在 theta_mid = (theta1 + theta2) / 2
- 宽度由 theta2 - theta1 控制
- 系数 4.5 (= 9/2) 使得扰动在包络边缘衰减到峰值的 ~1%
- 南北对称：南半球用 `PI - theta` 代入，保证扰动模式关于赤道镜像对称

### 时间调制：正弦波包

```
f_time(t) = sin( 2*PI * num_cycles * t / t_stop )
```

- 频率 = num_cycles / t_stop
- 相位从 0 开始
- 没有缓包络调制（纯正弦）

### 激活窗口

```
if (time > t_trigger && time < t_trigger + t_stop)
```

- `t_trigger`：扰动启动时间（建议设 50，让系统先驰豫到稳态）
- `t_stop`：扰动持续时间

## 参数列表

| 参数 | 物理含义 | 典型值 |
|---|---|---|
| dOmega | 扰动振幅 | 0.1~1.0 |
| t_trigger | 扰动启动时间 | 50.0 |
| t_stop | 扰动持续时间 | 10.0 |
| num_cycles | 周期数 | 20 |
| theta1, theta2 | 扰动纬度范围 (rad) | 如 0.3, 1.2 |

## 数学注释

- 高斯中 `SQR(THREE)/TWO = 9/2`：这是为了让 `f_theta` 在 u=±1（即 theta=theta1 或 theta=theta2）时衰减到 `exp(-4.5) ≈ 0.01`
- 电场的 `r * sin(theta)` 因子来自于球坐标中旋转电场诱导的 E×B 漂移
- 扰动加在 `ex1`（θ 方向 E）和 `ex2`（r 方向 E）上。`ex1` 驱动 r 方向的 E×B 漂移，`ex2` 驱动 θ 方向的 E×B 漂移
- 该扰动生成的 Alfvén 波沿偶极磁力线传播，磁力线两端（南北半球高纬区）有对应的镜像包络结构

## 实现位置

- `pgens/magnetoAlfven/pgen.hpp` 中的 `DriveFields<D>::ex1()` 和 `ex2()`
- 需新增参数 `dOmega`（振幅），恢复 `t_stop`, `num_cycles`, `theta1`, `theta2`
- `DriveFields` 构造函数需要接收 `dOmega`, `time`, `t_stop`, `num_cycles`, `theta1`, `theta2`
- `Omega` 仍保留在 `AtmFields` 返回时传入但不影响（始终为 0），或直接在 `DriveFields` 中去掉
