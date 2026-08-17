# axion_wald Design

## 1. Goal

在 Kerr（Kerr–Schild 坐标）真空中研究固定轴子背景对电磁场的作用：
Wald（或 vertical）背景磁场 + 轴子有效电流 J_a = eps·(ȧB + ∇a×E)，
通过引擎的 `-D axion=ON` 通道注入（两次 Ampere pass，时间二阶对齐）。
轴子背景**不演化**（不解 KG 方程），无粒子背反应之外的轴子耗散。

明确的排除项：m ≥ 1 的超辐射轴子云（含 φ 依赖，2D 轴对称网格无法表示）；
轴子场自身的演化与超辐射增长。

轴子背景两种模式：

- `sinusoid`：a = A·sin(ωt − k_r·r)——单元测试专用（归一化、时间收敛阶），
  不追求物理形式。
- `cloud`：m=0 引力原子本征态（Detweiler 类氢近似，α ≲ 0.2 严格适用，
  测试可放宽）：

  ```
  a(t,r,θ) = A · f_l(r) · P_l(cosθ) · cos(ω_c t + φ)
  f_0(r) = exp(−λr),          λ = α²/M
  f_1(r) = (λr)·exp(−λr),     λ = α²/(2M)        # 峰值 r_max = 2M/α²
  ω_c = (α/M)·(1 − α²/(2(l+1)²))                 # 束缚能修正, n=0
  ȧ   = −ω_c·A·f·g·sin(ω_c t + φ)
  ∂_r a = A·f·(l/r − λ)·g·cos(ω_c t + φ)
  ∂_θ a = A·f·g'(θ)·cos(ω_c t + φ)   (l=0: 0; l=1: −sinθ)
  ∂_φ a = 0
  ```

  注：|0,1,1⟩ 的径向轮廓与 l=1, m=0 相同（m 独立性在 O(α²) 成立），
  角向从 sinθ·cos(φ−ωt) 换成 P_1(cosθ)=cosθ。
  M = 1（代码单位 r_g = 1）。参考 Tang & Papantonopoulos,
  arXiv:2506.16036 式 27——指数因子为 exp(−(α²/2M)r)，
  本项目 axion_cloud_eigenstate.md 中的 e^{−r/(2Mα²)} 是笔误（已向用户指出）。

## 2. Basic Configuration

- Entity v1.4.5（fork StaticObserver/entity，分支 dev/axion-grpic），GRPIC 引擎，
  2D 轴对称；构建需 `-D axion=ON`（pgen 在 AXION_ENABLED 未定义时 `#error`）。
- 度规：Kerr_Schild / QKerr_Schild / Kerr_Schild_0（traits 同 wald）。
- 场以代码单位逆变坐标基存储（含 dr/dθ 因子）；归一化：`scales.larmor0`、
  `scales.skindepth0` → B0 = 1/larmor0，q0/B0 由引擎处理；eps 乘在引擎的
  Ampere 系数上（fieldsolvers.h AmpereAxion）。
- 单元测试用 Kerr_Schild_0（a=0）；物理测试用 QKerr_Schild（a=0.9）。

## 3. Initial Electromagnetic Fields

State: implemented（wald 已验证；bhac 型新增，待验证）

- `InitFields`（复制自 pgens/wald）：`init_field = "wald" | "vertical" | "bhac"`，
  Wald 精确真空解，供 MATCH 边界每步弛豫。
- `init_field = "bhac"`（移植自 pgens/bhac）：B 来自
  A_φ = field_B0·r·exp(−r/r_decay)·sin²θ（极区解析正则化），
  参数 `setup.field_B0`（默认 1.0）、`setup.r_decay`（默认 200.0）；
  D 只含轴子 Gauss 项 −ε̃·a·B（与 "vertical" 分支同一形式，不含真空 D）。

## 4. Initial Particles

State: implemented（本地 D·B 触发注入，移植自 bh-reconnection pgen，
Parfrey et al. 2019 处方）

- `DdotBWeightedPairs`：注入盒 `setup.xi_min`–`setup.xi_max`（物理坐标，
  作为 `InjectNonUniform` 的 box 限制迭代范围）内，直接用本地
  staggered EM 数组值（不插值、不转 tetrad），以空间度规降指标计算
  D·B 与 B²，对满足
    |D·B|/B² > ddotb_threshold  且  B² > sigma_min_fraction·ρ
  的格子每次调用恰好注入一对 e±（`number_density = 2/ppc0`）。
  ρ = `FldsID::Rho`（species {1,2}），每次注入前由
  `ComputeMomentWithSpecies` 实时重算，注入自调节。
- 注入密度编码在粒子权重里（Parfrey 本地处方）：
    Δñ = pair_creation_rate · ñ_GJ · |D·B|/√B²，  ñ_GJ = B0·skindepth0²
    weight = ppc0 · Δñ
  非笛卡尔注入器存的 weight 会乘 √det(h)/V0，密度矩又除 n0·√det(h)
  （n0 = ppc0/V0），二者相消，每个粒子恰好沉积 Δñ——ppc0 只影响
  采样容量，不改变物理注入密度。
- `InitPrtls` 与 `CustomPostStep` 调用同一 `InjectPlasma`：Maxwellian
  （`setup.temperature`）+ `InjectNonUniform`（species {1,2}, use_weights=true）。
- 注入盒缺省或非法（xi_min ≥ xi_max）时整体跳过，真空算例行为不变。
- 物理要点：pusher 直接插值代码 D 场作 E（tetrad 系），而本约定的 D 已是
  含轴子修正的场，故粒子自动感受正确的物理 Lorentz 力，引擎零改动。

## 5. Boundaries

State: implemented（沿用 wald 惯例，未验证）

- fields/particles：GR 只需给 rmax；rmin 由框架强制 HORIZON
  （grid.cpp:212–213，故 extent 的 rmin 必须 < r_+）；x2 方向 AXIS 自动。
- rmax：fields MATCH（弛豫回 init_flds），particles ABSORB。

## 6. Custom Behavior

State: implemented（未验证）

- `axion` functor（`AxionField<D>`）：引擎 trait `HasAxionField` 检测，
  每步在 aux（t=n）与 main（t=n+1/2）两次 Ampere pass 注入 J_a。
  - `a(x_Ph, t)` / `dot_a(x_Ph, t)` / `grad_a(x_Ph, t, g)`：物理坐标 (r,θ)
    的场值、ȧ 与协变梯度；kernel 内部经 `transform<Idx::PD, Idx::D>` 转到
    代码坐标梯度并组装。
  - `eps`：耦合强度 ε（≡ g_{aγ}·a₀ 的代码单位对应量，由平坦极限测试锁定）。
- Gauss 一致初始化：`InitFields::dx1/2/3` 附加 −ε̃·a(x,0)·B(x,0)
  （ε̃ = (q0/B0)·eps，与注入 kernel 同系数）——这就是广义 Gauss 定律
  ∇·D = ρ_a 在真空中的精确特解（GR 下同样精确，见 axion-Komissarov 笔记
  "D of a pure axion background" 节），不是什么额外的近似；
  B 分量直接复用 `bx1/2/3` 在 D 的 staggered 位置求值。
  开关 `setup.axion_gauss_init`（默认 true；eps=0 时自动为零）。
  开启后任意 phase 均满足初始约束，不必迁就 phase=π/2。
- TOML 映射：
  - `setup.axion_mode` = "sinusoid" | "cloud"（默认 "sinusoid"）
  - `setup.axion_eps`（默认 0）
  - sinusoid：`setup.axion_amplitude`、`setup.axion_omega`、`setup.axion_k1`
  - cloud：`setup.axion_alpha`（α，默认 0.5）、`setup.axion_l`（0|1，默认 1）、
    `setup.axion_amplitude`（A）、`setup.axion_phase`（默认 0）
  - plasma（本地 D·B 触发注入，见 §4）：`setup.xi_min`、`setup.xi_max`
    （注入盒，缺省 = 不注入）、`setup.pair_creation_rate`（默认 0.5）、
    `setup.ddotb_threshold`（默认 1e-2）、`setup.sigma_min_fraction`
    （默认 0.05）、`setup.temperature`（默认 0.01）；需配 `[particles]`
    species {1,2} 与 `use_weights = true`（`setup.multiplicity` 已弃用，
    toml 中保留无害；旧的 `setup.sigma_max`/r^{-3/2} 剖面已被 D·B
    触发注入取代）

## 7. Custom Output

State: not-used

- 能量诊断（η_EM + η_prtl = 1、−∫E·J_a dV）计划用 CustomStat 实现，后续工作。

## 8. PGen-TOML Contract

- traits（GRPIC / Kerr_Schild+QKerr_Schild+Kerr_Schild_0 / 2D）<-> TOML
  `simulation.engine=grpic`、`grid.metric.metric`、2 维 resolution。
- `setup.axion_*` 键名与 PGen 构造函数的 `params.get` 一一对应（见 §6）。
- cloud 模式的 α 与 extent 必须一起设计：r_max = 2M/α²（l=1）应落在
  计算域内；rmin < r_+ = 1+√(1−a²)。
- extent rmin < r_+（HORIZON 边界强制）。

## 9. Current Status

**状态：正式通过测试（2026-07-31，用户确认）。** 注入归一化、GR 定性、
upstream 回归、Gauss 约束四项全部通过，可用于正式物理 run。

- implemented：sinusoid 模式、cloud 模式（m=0 本征态）、两个 TOML、Wald 初始场。
- verified（m87, 2026-07-31, build-A sha256 0262b8da…）：
  - **注入验证（kerr_schild_0, a=0, cloud 模式）**：eps1−eps0 差分对照
    −(q0/B0)·ε·[a(t)−a(0)]·B，早时逐点比率 1.0000±0.0002（全网格、Dr/Dth）；
    归一化（q0/B0 与 eps）锁定；三档 CFL（0.5/0.25/0.125）偏差单调减小，
    晚期 ~0.5% 偏差为 dt 无关的电磁响应物理（波动再分配 + ∇a×E 项），
    非注入误差。
  - **GR 真空（qkerr_schild, a=0.9, cloud α=0.6 l=1）**：早期模式投影
    ~0.8（ȧB 主项）+ ~0.2（∇a×E 交叉项，a=0.9 时 Wald E≠0 被激活——
    与 a=0 时比率恰为 1.000 对照一致）；轴/视界/MATCH 边界无伪模。
    晚期（t≳6）eps1/eps0 差分被基线自身的非线性漂移（a=0.9 真空 Wald
    不平静）主导，属物理而非误差。
  - **回归**：axion=ON/OFF 与 upstream v1.4.5 短跑一致（最差 ~7e-7 相对，
    位于发电机放大的 Bph，三对差异同量级——roundoff 种子级）。
  - **Gauss 约束（kerr_schild_0, cloud α=0.6 l=1, phase=0, eps=1, 70661460）**：
    C ≡ ∇·D − ρ_a 由输出 D、B 直接计算（散度算子 (1/r²sinθ)[∂_r(r²sinθ D¹)
    + ∂_θ(r²sinθ D²)]，注意该度规输出分量为物理逆变分量）。
    `axion_gauss_init=false`：max|C| = max|ρ_a|（relC = 1.000），全程守恒
    （ρ_a 振荡衰减时 C 保持 1.322e-9 不变）——精确复现"未屏蔽初态偏离
    −ρ_a(0) 且永久保持"的理论预言，同时证明 J_a 注入的离散连续性极好。
    `axion_gauss_init=true`（默认）：max|C| ≈ 4e-13（relC ≈ 3e-4，截断级别），
    全程稳定——**∇·D = ρ_a 严格成立**，相位自由。
- 备注：`kerr_schild_0` 在 Entity 中是平直球坐标（α=1、β=0、h_11=1，
  见 metrics/kerr_schild_0.h），即 t1 系列是严格的平坦极限测试；
  真 GR 测试用 `qkerr_schild`（t2 系列）。GR 下的 Gauss 验证需用对应
  度规因子构造散度算子，方法相同（连续恒等式 ∇·(aB)=B·∇a 与度规无关，
  pgen 屏蔽项按构造在任意度规下精确）。
- pending：Ledger run 登记（render-run/record 流程）；粒子接入与
  η_EM + η_prtl = 1 诊断（CustomStat）。
- open：cloud 模式 l ≥ 2 扩展；α > 0.2 时解析本征态精度；
  ~~初始约束~~ → 已由 Gauss 一致初始化解决（见 §6）：a(0)≠0 时
  D(0) += −ε̃a(0)B(0) 使 ∇·D = ρ_a 在 t=0 精确成立（离散截断级别），
  相位自由。`setup.axion_gauss_init=false` 可退回旧行为做对照。

## 10. Important Changes

- 2026-07-31：初版（sinusoid）+ 引擎 axion 通道（d02648bf）。
- 2026-07-31：新增 cloud 模式（m=0 本征态）；修正本征态径向衰减率为
  α²/(2M)（对照 arXiv:2506.16036 式 27）。
- 2026-07-31：`setup.axion_l` 改按 int 读取（自定义 setup 整数以 int 存储，
  parameters.cpp:267）（10f35220）。m87 全测试矩阵通过（见 §9）。
- 2026-07-31：新增 Gauss 一致初始化（`AxionField::a` + `InitFields::d_axion`，
  70661460）；m87 Gauss 约束验证通过（见 §9）。
- 2026-08-03：新增 accretion 式等离子体注入（`PointDistribution` +
  `InitPrtls`/`CustomPostStep`，注入盒缺省跳过保持真空兼容）；
  新增 `axion_plasma.toml`（ε=0 baseline，extent [1,7]，256²，runtime 50）。
- 2026-08-17：注入方法更换为本地 D·B 触发的 Parfrey 式固定对注入
  （`DdotBWeightedPairs`，移植自 bh-reconnection pgen；触发条件
  |D·B|/B² > ddotb_threshold 且 B² > sigma_min_fraction·ρ，每格每步
  恰好一对 e±，物理密度由权重 Δñ = R·ñ_GJ·|D·B|/√B² 携带）。
  取代旧的磁化判据 + r^{-3/2} 剖面；`setup.sigma_max` 移除，新增
  `setup.pair_creation_rate` / `setup.ddotb_threshold` /
  `setup.sigma_min_fraction`，5 个 plasma toml 同步更新
  （分支 dev/axion-grpic-ddotb-inj）。
