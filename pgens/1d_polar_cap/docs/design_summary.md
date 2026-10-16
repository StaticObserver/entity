# 1D Polar-Cap PGen 设计总结

> 本文档是对 `pgen.hpp`、`initial_injection.hpp`、`qed/*.hpp` 与
> `1d_polar_cap.toml` 的中文设计总结。权威设计文档为同目录的
> `design.md`(英文,含逐组件状态);本文只作概括,不一致处以 `design.md`
> 与源码为准。总结日期:2026-07-24,基于 Entity v1.4.4 源码检出。

## 1. 物理目标

在 Entity v1.4.4 上实现局部一维脉冲星极冠放电:电荷匮乏产生平行电场
`E_parallel`,被加速的正负电子沿磁力线发射曲率辐射光子,光子累积磁转化
光深后转化为正负电子对,新对通过标准 SRPIC 电流沉积反馈到放电回路。

第一版明确排除:横向朗道能级、变化的磁场几何、广义相对论效应、通用
QED 模块;磁力线曲率半径 `rho_c` 为常数给定值。

## 2. 基本配置

- 引擎 SRPIC,度规 Minkowski(笛卡尔),一维(`Dim::_1D`),traits 在
  `pgen.hpp` 中静态限定。
- 生产构建契约:Esirkepov 电流沉积 + 三阶粒子形状(`SHAPE_ORDER=3`),
  输出密度矩用同阶 spline 平滑。
- 种类契约:电子 = 1,正电子 = 2;QED 开启时光子 = 3(无质量、中性、
  光子推进器、≥3 个实数 payload)。QED 关闭时只允许两个种类,不分配
  光子容器,也不读谱表。
- 归一化仍未定标:检入参数是从旧原型迁移来的基线值
  (`larmor0=4e-4`,`skindepth0=0.02`),QED 路径默认
  `qed.enable = false` 保持惰性。

## 3. 初始电磁场

filled start(CPT2024 / Witte+2026 式)是唯一初始条件,无
`initial_condition` 开关。`InitialFields` 给定常数 `B_x = B0`,且全空间
`E_x = 0`:计算域被等离子体填满并逐点满足 Gauss 闭合 —— 沉积净电荷等于
给定 GJ 背景
`rho_GJ(x) = extra_positron_density * (1 + rho_gj_gradient * (x - x_surface) / L)`,
故 `dE_x/dx = rho_deposited - rho_GJ = 0`,放电由给定外电流经 Ampere
定律驱动失衡,而非由初始真空电场驱动。背景电荷是物理解释的一部分,
不是动态种类。

## 4. 初始粒子

三条注入路径:

1. **中性大气**:Entity 非均匀注入器 `InjectNonUniform`,把配置的总密度
   均分给电子和正电子,指数衰减标高由
   `grid.boundaries.atmosphere.height` 控制(只拟合中性大气,不能拟合
   含过剩电荷的总密度)。
2. **均匀体填充**:`initial_injection.fill_density`(n0 单位)的均匀
   中性电子-正电子填充,经同一 `InjectNonUniform` 路径从 `x_surface`
   注入到整个活动域,模拟上一次放电留下的等离子体(参考值 10 n_GJ);
   `fill_density = 0` 时跳过。
3. **过剩正电子**:PGen 本地单种类注入器
   (`initial_injection.hpp` 的 `SingleSpeciesInjector`),只注入电荷
   不平衡部分;轮廓贯穿全计算域并按 GJ 梯度
   `1 + rho_gj_gradient * (x - x_surface) / L` 延伸,使沉积净电荷逐点
   等于 GJ 背景。密度低于 `minimum_density` 的格子跳过;目标 PPC 低于
   `minimum_ppc` 时保留固定宏粒子数、用权重
   `target_ppc / minimum_ppc` 编码密度(权重参与 SRPIC 电流沉积并被
   发射光子/转化对继承);高于下限时用无偏随机舍入。带显式容量检查,
   注入后更新 `npart`、`counter` 与排序状态。

所有路径都用标准 1D3V Maxwellian;旧原型强制 `u2 = u3 = 0` 的全局修改
未恢复。大气路径读取全局 `particles.use_weights = true`,补充注入与
`N_*` 矩使用同一加权密度约定。

## 5. 边界

- `x1 min`:场 ATMOSPHERE、粒子 ATMOSPHERE(持续补充中性大气);
- `x1 max`:场 MATCH(`B_x` 放松到 `B0`、`E_x` 放松到 0,抑制吸收边缘
  的鞘层累积)、粒子 ABSORB(引擎内 flux-conserving 簿记:推进器把越界
  粒子钳在边界面、标准沉积计入末段位移,`srpic::BoundaryAbsorb` 在
  `CurrentsDeposit` 之后才标记 dead);
- 不修改任何全局边界或求解器行为;大气侧边界场强制 `E_x = 0`、
  `B_x = B0`。

## 6. 自定义行为(QED 路径)

三个独立开关 `curvature_drag`、`curvature_emission`、
`magnetic_pair_creation`,全部再受 `qed.enable` 总门控。

- **外部电流**:`MagnetosphericCurrent` 把 tetrad 分量(以 `n0 q0 c`
  归一)转成逆变分量 `J^1 = J^(hat1)/dx` 后由 Ampere 核直接叠加;
  电流在全空间(含右侧 MATCH 层)保持均匀;`ppc0` 因子由 Entity 内部
  处理。
- **曲率辐射**(`qed/curvature_emission.hpp`,EmissionPolicy):
  - 连续反冲 `-gamma^3 u / rho_c^2`,单步动量损失上限
    `max_drag_fraction`(默认 0.2);
  - 期望光子数由截断在 `[photon_energy_min, 母粒子动能]` 区间的曲率
    CCDF 积分给出,随机舍入无偏;超过
    `max_photons_per_particle_step` 时封顶宏粒子数、超额多重数进权重;
  - 光子能量在保留 CCDF 区间内均匀抽样后经 `inverse_ccdf` 变换;
  - 完整初始化光子 payload(见下)与权重。
  - 刻意不用内置同步辐射:它依赖本地电磁加速,对理想平行运动为零,
    不能表达给定的磁力线曲率半径;TOML 中不得出现
    `radiative_drag = "synchrotron"`。
- **谱表**(`qed/curvature_spectrum.hpp`):单一 CCDF 表
  `data/curvature_ccdf.tsv`(由 `generate_curvature_table.py` 生成),
  host 端严格校验单调性,log-log 插值,小 x 用解析渐近、大 x 用指数
  尾外推;路径先按启动目录解析,再相对 PGen 目录解析。
- **光子不透明度**(`qed/photon_opacity.hpp`,CustomParticleUpdate):
  payload 约定 `pld_r = [epsilon_gamma, tau_pair, theta_B]`。沿光子实际
  轨迹长度用偶数子步复合 Simpson 积分 Erber 型磁转化率;角度增量只用
  x1 位移(`dt * |ux1|/|u| / rho_c`);率中含 `abs(sin(theta_B))` 且
  `epsilon_gamma * |sin theta_B| < 2` 时为零。`opacity_substeps` 必须
  为正偶数。
- **磁对转化**(`qed/magnetic_pair_creation.hpp`,CustomPostStep):
  光子同时满足 `epsilon_gamma * |sin theta_B| >= 2` 与
  `tau_pair >= conversion_optical_depth` 时确定性转化为一对电子/正
  电子,等权重、各带一半光子能量,沿光子 x1 传播方向放置;原子计数器
  保证一一配对,两个子粒子写完才将光子标记为 dead。对在标准场/电流
  步之后产生,下一时间步才参与。

## 7. 自定义输出

未使用。只请求标准场与种类密度输出(`N_1, N_2, E, B, J`),无
`CustomFieldOutput` / `CustomStat`。

## 8. 关键契约(PGen <-> TOML)

- 种类索引、质量、电荷、推进器与 emission 策略须与第 2 节一致;
  QED-off 时输出不得请求 `N_3`。
- `rho_c`、`gamma_emit`、`photon_energy_min`、`b_over_bq`、
  `max_photons_per_particle_step`、`opacity_substeps`(偶数)、
  `conversion_optical_depth` 均为正;`max_drag_fraction ∈ (0,1)`;
  构造函数集中做 host 端校验。
- `extra_positron_density > 0`(默认 1)是 GJ 净电荷尺度,使沉积净电荷
  逐点等于 GJ 背景、初始 `E_x = 0` 满足 Gauss 闭合。
- `external_current` 是内部区域的 tetrad 归一化电流,PGen 负责换算基底,
  电流在全空间均匀(含 MATCH 层)。
- `x_surface` 由 `algorithms.current_filters + 2` 与 5 取大得到的缓冲
  格数决定,不在 TOML 中直接配置。

## 9. 当前状态(以 `design.md` 为准)

已完成:v1.4.4 迁移的 PGen/TOML/谱表及生成脚本、曲率发射与连续反冲、
光子角度/光深更新、确定性磁对转化与粒子簿记、非编译的 Python 参考检
查(`tests/reference_models.py`)。

待办:Entity 编译与冒烟运行、Kokkos 内单粒子拖曳/发射测试、初态离散
Gauss/Ampere 检查、MPI payload 传输与多 domain 转化检查、生产归一化
与辐射能量系综定标。
