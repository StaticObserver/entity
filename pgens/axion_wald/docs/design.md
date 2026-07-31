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

State: implemented（沿用 wald，未验证）

- `InitFields`（复制自 pgens/wald）：`init_field = "wald" | "vertical"`，
  Wald 精确真空解，供 MATCH 边界每步弛豫。

## 4. Initial Particles

State: not-used

- 真空算例，`algorithms.deposit.enable = false`。粒子接入属后续工作
  （η_EM + η_prtl 诊断）。

## 5. Boundaries

State: implemented（沿用 wald 惯例，未验证）

- fields/particles：GR 只需给 rmax；rmin 由框架强制 HORIZON
  （grid.cpp:212–213，故 extent 的 rmin 必须 < r_+）；x2 方向 AXIS 自动。
- rmax：fields MATCH（弛豫回 init_flds），particles ABSORB。

## 6. Custom Behavior

State: implemented（未验证）

- `axion` functor（`AxionField<D>`）：引擎 trait `HasAxionField` 检测，
  每步在 aux（t=n）与 main（t=n+1/2）两次 Ampere pass 注入 J_a。
  - `dot_a(x_Ph, t)` / `grad_a(x_Ph, t, g)`：物理坐标 (r,θ) 的 ȧ 与协变梯度；
    kernel 内部经 `transform<Idx::PD, Idx::D>` 转到代码坐标梯度并组装。
  - `eps`：耦合强度 ε（≡ g_{aγ}·a₀ 的代码单位对应量，由平坦极限测试锁定）。
- TOML 映射：
  - `setup.axion_mode` = "sinusoid" | "cloud"（默认 "sinusoid"）
  - `setup.axion_eps`（默认 0）
  - sinusoid：`setup.axion_amplitude`、`setup.axion_omega`、`setup.axion_k1`
  - cloud：`setup.axion_alpha`（α，默认 0.5）、`setup.axion_l`（0|1，默认 1）、
    `setup.axion_amplitude`（A）、`setup.axion_phase`（默认 0）

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
- pending：Ledger run 登记（render-run/record 流程）；粒子接入与
  η_EM + η_prtl = 1 诊断（CustomStat）。
- open：cloud 模式 l ≥ 2 扩展；α > 0.2 时解析本征态精度；
  **初始约束**：引擎未实现 Gauss 源项 ρ_a = −g_{aγ}B·∇a。由轴子流守恒，
  ∇·D − ρ_a 在演化中保持常数；a(0)≠0（phase=0）时初态偏离约束
  −ρ_a(0) 且永久保持（不影响 D 的动力学，但影响电荷/能量诊断的解释）。
  **建议生产运行取 phase=π/2（a(0)=0，ρ_a(0)=0）使约束自动精确成立。**

## 10. Important Changes

- 2026-07-31：初版（sinusoid）+ 引擎 axion 通道（d02648bf）。
- 2026-07-31：新增 cloud 模式（m=0 本征态）；修正本征态径向衰减率为
  α²/(2M)（对照 arXiv:2506.16036 式 27）。
- 2026-07-31：`setup.axion_l` 改按 int 读取（自定义 setup 整数以 int 存储，
  parameters.cpp:267）（10f35220）。m87 全测试矩阵通过（见 §9）。
