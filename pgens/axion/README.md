# Axion PGen

PGen 是独立 struct（无基类）。引擎通过 concept 自动检测各成员：

| 成员 | Concept | 触发时机 |
|------|---------|---------|
| `ext_current` | `HasExtCurrent` | Ampere 求解器每个子步 |
| `init_flds` | `HasInitFlds` | 场初始化 |
| `InitPrtls(domain)` | `HasInitPrtls` | 粒子初始化 |

PGen 声明为 2D（`compatible_with<Dim::_2D>`），1D 运行时设 `resolution = [N]`。

## TOML `[setup]` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epsilon` | 必填 | g_{aγ} × a₀，轴子耦合强度 |
| `omega_ratio` | 必填 | ω_a / skindepth0，轴子频率比 |
| `k` | 0.0 | 轴子波数（沿 x₁） |
| `B0` | 1.0 | 背景磁场 Bx₁ 强度 |
| `B0_perp` | 0.0 | 均匀横向 Bx₂（当前 2D 实验用） |
| `temperature` | 0.0 | 初始 pair plasma 温度 (m_e c² 单位) |
| `density` | 0.0 | 初始粒子密度，0 = 真空 |
| `use_grad_a_cross_e` | true | 是否在 J_a 中包含 ∇a×E 项（ON/OFF 判别实验用） |
| `E0_x3` | 0.0 | 均匀 Ex₃ 初始种子（真空 ∇a×E 受迫波测试用） |
| `profile` | `"uniform"` | x₁ 向密度剖面：`uniform` / `ramp` / `barrier` |
| `n_min` | 1.0 | 剖面密度下限（× density） |
| `n_max` | 1.0 | 剖面密度上限（× density），profile ≠ uniform 时必须 > 0 |
| `ramp_W` | 1.0 | barrier 台肩陡度，台肩宽 ~ 1/ramp_W |
| `ramp_xa` | 0.0 | 剖面起点（profile ≠ uniform 时必填） |
| `ramp_xb` | -1.0 | 剖面终点（必须 > ramp_xa） |

剖面形状（n 以 `density` 为单位）：

- `uniform`：n = n_max（代码路径与加剖面前完全一致）
- `ramp`：xa → xb 线性 n_min → n_max，区间外截断
- `barrier`：[xa, xb] 内 n_max 平台，外部 n_min，tanh 台肩宽 ~ 1/ramp_W

注入按 **ppc ∝ n(x₁)** 变化、粒子权重恒为 1，使 N_D = ppc·(λ_D/dx)²
沿梯度恒定。注意 `maxnpart` 要按 n_max（而非平均密度）预留。

## v1.4.3 TOML 注意事项

- `engine = "SRPIC"`（PascalCase）
- `metric = "Minkowski"`（PascalCase）
- `log_level`: `"VERBOSE"`, `"WARNING"`, `"ERROR"`（无 `"INFO"`）
- `[output.fields.smoothing]`: `order`（0=NGP, 1=CIC）和 `method`（`"spline"`/`"const"`）
- `current_filters = 0`：必须关闭，>0 会破坏电荷守恒，导致高斯残差增长

## InitFields

t=0 时设置初始电磁场:

- `Ex₁ = -ε·B₀·cos(k·x₁)` → 满足 ∇·E = ρ_a
- `Ex₂ = 0`，`Ex₃ = E0_x3`（均匀种子，默认 0）
- `Bx₁ = B₀`（常量）
- `Bx₂ = B0_perp`（常量）
- `Bx₃ = 0`

注意: 不要额外除以 `larmor0`。

## ext_current — 轴子电流注入

v1.4.3 原版 ext_current 只接受 `coord_t<D>`。我们扩展了 `ExtCurrentContext<D>` struct，包含:

- `x_Ph`, `i1`/`i2`/`i3` — 空间坐标
- `em` — EM 场访问（`ctx.em(i1, em::bx1)`）
- `time` — 当前时间（half-step）
- `dx` — 网格间距

PGen 的 `jx1/jx2/jx3(ctx)` 方法通过 ctx 访问实时 EM 场和当前时间，计算完整轴子电流:
- `J_a = ε · (∂_t a · B + ∇a × E)`
- `∂_t a = ω·sin(φ)`，`∂_x₁ a = -k·sin(φ)`，`φ = k·x₁ - ω·t`
- `(∇a×E)₂ = +k·sinφ·E₃`，`(∇a×E)₃ = -k·sinφ·E₂`

2D 基约定：`stored[ex1, ex2, bx1, bx2] = 物理值/dx`，`stored[ex3, bx3] = 物理值`。
ȧB 项指标相同因子自动抵消；∇a×E 跨指标，须显式补 `dx`：
`jx2 += coef·ε·k·sinφ·em(ex3)/ctx.dx`，`jx3 -= coef·ε·k·sinφ·em(ex2)·ctx.dx`
（2D 专属，3D 下须重新推导）。∇a×E 可用 `use_grad_a_cross_e = false` 关闭。

注入发生在 `CurrentsAmpere_kernel`，使用 `source_time = time + HALF * dt`。

## Bx2 横向场

当前 `InitFields` 的 `Bx₂ = B0_perp`（常量）。文档与知识库中提到的
`bx2_wavenumbers` / `bx2_amplitudes` 多模谱在当前 `pgen.hpp` 中不存在
（历史版本曾有，见 git 历史）；如需多模谱须重新实现。

## 密度剖面

`InitPrtls` 支持 `setup.profile = uniform/ramp/barrier`（沿 x₁）：

- 注入概率 ∝ n(x₁)/n_max，粒子权重恒为 1 → N_D 沿梯度恒定
- `profile = "uniform"` 时代码路径与引入剖面前逐字一致（回归安全）
- 测试 TOML：`test_profile_barrier.toml`

## 测试

- `test_adotb_wave.toml` / `test_k0.toml` / `test_b0perp.toml` / `test_gradaE_on.toml` /
  `test_gradaE_off.toml` / `test_profile_barrier.toml` / `test_profile_ramp.toml`：
  真空 J_a 各通道与密度剖面的解析对照测试。
- `test_invalid_*.toml`：5 个参数校验负测试（应报 `raise::ErrorIf`）。
- 检查脚本 `scripts/check_pgen_suite.py`（仓库外，本地 `axion-pic/scripts/` 与
  m87 `~/entity/axion-pic/scripts/` 同步）。
- 横向 E2 的解析参考是受迫波动方程解（见 `check_pgen_suite.py` 模块注释），
  不是局域 ODE——忽略波传播会引入 ~10% 虚假偏差。
- 验证报告：`analysis_results/pgen-verification-2026-09-01/reports/`
  （2026-09-01 全部 PASS，commit a65cf7d3）。
