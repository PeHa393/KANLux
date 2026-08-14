# KANLux.jl

> ⚠️ **警告**：本包含有大量未经人工审查的 AI 生成代码，仓库作者目前尚未完成对其检查。任何人在使用或修改本仓库时，都需要特别注意。

[简体中文](README_Zh_CN.md) · [English](README.md)

基于 [PyKAN](https://github.com/KindXiaoming/pykan) 迁移而来的 Kolmogorov-Arnold 网络（KAN）的 Julia/Lux.jl 实现。本包提供了构建与评估 KAN 模型所需的 B 样条数学、符号函数注册表、数据工具、层容器以及结构操作。训练循环、绘图、检查点保存以及符号自动回归被有意地留给了应用层。

## 安装

需要 Julia 1.9 或更新版本。请从以下三种方式中任选其一。

### 1. 从源码安装（开发模式）

当你拥有本地代码副本、并希望修改代码时，使用此方式：

```julia
using Pkg
Pkg.develop(path="KANLux")
```

### 2. 从 GitHub URL 安装

直接从公开仓库安装当前的 `main` 分支：

```julia
using Pkg
Pkg.add(url="https://github.com/PeHa393/KANLux")
```

添加 `rev="main"`（或某个具体的提交/标签）可以固定到指定版本：

```julia
using Pkg
Pkg.add(url="https://github.com/PeHa393/KANLux", rev="main")
```

### 3. 通过包名安装（从 General 注册表）

```julia
using Pkg
Pkg.add("KANLux")
```

> ⚠️ **警告**：KANLux **尚未上传到 General 注册表**，因此该方式目前**不可用**。在注册完成之前，请使用上述两种方式之一。

本包依赖 `Lux`、`Random`、`LinearAlgebra` 和 `Statistics`。测试依赖（`Zygote`、`FiniteDifferences`、`LuxTestUtils`、`NPZ`）仅声明在 test 目标中。

## 快速上手

构建一个两层 KAN 并在合成数据上评估：

```julia
using KANLux, Lux, Random

rng = Xoshiro(42)

# width = [输入, 隐藏层, 输出]
model = MultKAN([2, 3, 1]; grid=3, k=3)

# 初始化可训练参数（`ps`）与非训练状态（`st`）。
ps, st = Lux.setup(rng, model)

x = randn(rng, 16, 2)
y, st = model(x, ps, st)

println(size(y))  # (16, 1)
```

`KANLux.create_dataset` 可以为标量值函数生成训练/测试张量：

```julia
using KANLux

f = x -> sum(x .^ 2; dims=2)
train_input, train_label, test_input, test_label =
    create_dataset(f, 2, (-1.0, 1.0), 100, 50, 1)
```

## 公共 API

本包导出了十二个符号。除非特别说明，所有张量均为 `Float64` 类型的 Julia 数组，并采用与 PyKAN/PyTorch 相同的布局（`batch` 为第一个维度）。

| 导出 | 描述 |
|--------|-------------|
| `B_batch` | Cox-de Boor 基函数求值 |
| `coef2curve` | 将 B 样条系数转换为曲线值 |
| `curve2coef` | 最小二乘曲线到系数的转换 |
| `extend_grid` | 向 B 样条网格添加均匀幽灵节点 |
| `SYMBOLIC_LIB` | 27 个符号函数（29 个键）的注册表 |
| `create_dataset` | 合成数据集生成器 |
| `KANLayer` | 数值 B 样条激活层 |
| `SymbolicKANLayer` | 闭式符号激活层 |
| `fix_symbolic!` | 为某条边指派符号函数 |
| `MultKAN` | 数值 + 符号复合 KAN 容器 |
| `prune` | 结构化的边/节点剪枝 |
| `refine` | B 样条网格细化 |

另外还有两个公开但未导出的辅助函数：`KANLux.fit_symbolic_params`（`fix_symbolic!` 使用的 a/b 网格搜索）和 `KANLux.silu`（默认的基函数）。

每个层和容器都实现了 Lux 层接口：

```julia
using KANLux, Lux, Random

rng = Xoshiro(0)
layer = KANLayer(2, 4, 5, 3)
ps, st = Lux.setup(rng, layer)           # ps = 可训练，st = 非训练
y, st = layer(randn(rng, 8, 2), ps, st)  # 前向返回 (输出, 新状态)
```

### 数组布局

设 `G` 为网格区间数、`k` 为阶数，扩展后的网格有 `S = G + 2k + 1` 个节点：

| 数组 | 形状 |
|-------|-------|
| `x`、`x_eval`（输入） | `(batch, in_dim)` |
| `grid` | `(in_dim, S)` |
| `coef` | `(in_dim, out_dim, G + k)` |
| `y_eval`（目标曲线） | `(batch, in_dim, out_dim)` |
| `B_batch(...)` 结果 | `(batch, in_dim, G + k)` |
| `coef2curve(...)` 结果 | `(batch, in_dim, out_dim)` |
| `curve2coef(...)` 结果 | `(in_dim, out_dim, G + k)` |

### B 样条原语

```julia
B_batch(x, grid, k)
coef2curve(x_eval, grid, coef, k)
curve2coef(x_eval, y_eval, grid, k)
extend_grid(grid, k_extend=0)
```

`B_batch` 计算阶数为 `k` 的 Cox-de Boor 基函数；`coef2curve` 即 PyKAN 中的 `einsum('ijk,jlk->ijl')`。`curve2coef` 为每条边求解最小二乘问题，以反转 `coef2curve` —— 它是一个结构操作辅助函数（供 `refine` 使用），**不**属于可微的训练前向过程。`extend_grid` 使用原始网格步长，在每一行两端各追加 `k_extend` 个均匀幽灵节点。

```julia
grid = extend_grid(repeat(reshape(collect(range(-1.0, 1.0; length=6)), 1, :), 2, 1), 3)
x = randn(16, 2)
coef = randn(2, 3, 8)                 # (in_dim, out_dim, G + k)，其中 G=5, k=3
y = coef2curve(x, grid, coef, 3)      # (16, 2, 3)
coef_rec = curve2coef(x, y, grid, 3)  # (2, 3, 8)
```

### `SYMBOLIC_LIB`

`SYMBOLIC_LIB` 是一个 `Dict{String, Tuple{Function, Int, Function}}`。每个值为 `(forward_fn, complexity, singularity_fn)`；`complexity` 是用于符号回归偏好的非负整数，`singularity_fn(x, y_th)` 在奇异点处返回一个有限的、受阈值限制的值（当 `singularity_avoiding=true` 时使用）。

29 个键覆盖 27 个不同函数（`x^0.5` 是 `sqrt` 的别名，`1/x^0.5` 是 `1/sqrt(x)` 的别名）：

```text
x, x^2, x^3, x^4, x^5,
1/x, 1/x^2, 1/x^3, 1/x^4, 1/x^5,
sqrt, x^0.5, x^1.5, 1/sqrt(x), 1/x^0.5,
exp, log, abs, sin, cos, tan, tanh, sgn,
arcsin, arccos, arctan, arctanh,
gaussian, 0
```

```julia
SYMBOLIC_LIB["sin"][1](0.5)           # 前向值
SYMBOLIC_LIB["sin"][2]                # 复杂度 = 2
SYMBOLIC_LIB["tan"][3](pi / 2, 10.0)  # 有限的奇异点保护值
```

### `create_dataset`

```julia
create_dataset(f, n_var, ranges=(-1.0, 1.0),
               train_num=1000, test_num=1000, seed=0)
```

`f` 接收一个 `(N, n_var)` 矩阵并返回标量、向量或矩阵（向量会被重塑为 `(N, 1)`）。`ranges` 可以是一个公共的 `(lo, hi)` 区间、一个 `n_var × 2` 矩阵，或一个逐变量的元组/向量集合。返回 `(train_input, train_label, test_input, test_label)`。

```julia
train_x, train_y, test_x, test_y =
    create_dataset(x -> sum(x .^ 2; dims=2), 2, (-1.0, 1.0), 100, 50, 1)
```

### `KANLayer`

```julia
KANLayer(in_dim, out_dim, num, k;
         grid_eps=0.02, grid_range=(-1.0, 1.0), base_fun=silu,
         sp_trainable=true, sb_trainable=true)
```

数值 B 样条边层。每条 `(input, output)` 边计算 `scale_base * base_fun(x) + scale_sp * spline(x)`，再乘以 `st.mask` 掩码，然后对输入求和。`ps` 保存 `coef` 以及可训练的缩放系数；`st` 保存 `grid`、`mask` 以及任何非训练的缩放系数。

```julia
layer = KANLayer(2, 4, 5, 3)   # in=2, out=4, 网格区间=5, 阶=3
ps, st = Lux.setup(rng, layer)
# ps.coef       :: (2, 4, 8)      (in_dim, out_dim, G + k)
# ps.scale_base :: (2, 4)
# ps.scale_sp   :: (2, 4)
# st.grid       :: (2, 12)        (S = 5 + 2*3 + 1)
# st.mask       :: (2, 4)
y, _ = layer(randn(rng, 8, 2), ps, st)  # (8, 4)
```

当 `sp_trainable=false` / `sb_trainable=false` 时，相应的缩放系数会存放在 `st` 中而非 `ps` 中。

### `SymbolicKANLayer`、`fix_symbolic!`、`fit_symbolic_params`

```julia
SymbolicKANLayer(in_dim, out_dim)

fix_symbolic!(layer, i, j, name; random=false, a_range=(-10.0, 10.0),
              b_range=(-10.0, 10.0), grid_number=21)
fix_symbolic!(layer, i, j, name, x, y; random=false, ...)

KANLux.fit_symbolic_params(x, y, fun; a_range=(-10.0, 10.0),
                           b_range=(-10.0, 10.0), grid_number=21)
```

符号层对每条边应用闭式变换 `c * f(a*x + b) + d`。`ps` 保存 `affine :: (out_dim, in_dim, 4)`，其元素为 `[a, b, c, d]`；`st` 保存 `mask`、`funs` 和 `funs_name`。未被指派函数的边输出为零。

`fix_symbolic!(layer, i, j, name)` 将 `SYMBOLIC_LIB[name]` 指派给边 `(输入 i, 输出 j)`。当不提供 `x`/`y` 时，仿射变换为恒等 `[1, 0, 1, 0]`（或当 `random=true` 时为随机值）；当提供 `x`/`y` 时，则通过 `fit_symbolic_params`（a/b 网格搜索 + c/d 最小二乘）拟合为 `y ≈ c*f(a*x+b)+d`。该层会被原地修改；需要通过设置 `layer.mask[j, i] = 1.0` 来启用这条边。

```julia
layer = SymbolicKANLayer(2, 2)
fix_symbolic!(layer, 1, 1, "sin")   # 输入 1 -> 输出 1
layer.mask[1, 1] = 1.0
ps, st = Lux.setup(rng, layer)
y, _ = layer(randn(rng, 8, 2), ps, st;
             singularity_avoiding=true, y_th=10.0)
```

### `MultKAN`

```julia
MultKAN(width; grid=3, k=3, mult_arity=2, base_fun=silu,
        symbolic_enabled=true, grid_eps=0.02, grid_range=(-1.0, 1.0),
        sp_trainable=true, sb_trainable=true)
```

复合容器，堆叠 `depth = length(width) - 1` 个数值层、符号层、子节点仿射变换、可选的乘法节点以及节点仿射变换。`width` 的每一项是整数神经元数量或 `[n_sum, n_mult]` 对。`grid`/`k` 可以是标量或逐层向量；`mult_arity` 可以是标量（同构）或与 `width` 匹配的向量。

```julia
model = MultKAN([2, 3, 1]; grid=3, k=3)
ps, st = Lux.setup(rng, model)
# ps.act_fun[d]      :: KANLayer 参数（逐层）
# ps.symbolic_fun[d] :: (affine=...)（逐层）
# ps.node_bias, ps.node_scale        :: （逐层）
# ps.subnode_bias, ps.subnode_scale  :: （逐层）
# st.act_fun[d].grid/.mask, st.symbolic_fun[d].mask/.funs/.funs_name
y, _ = model(randn(rng, 16, 2), ps, st)
```

通过内嵌层指派符号边：

```julia
model = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=true)
fix_symbolic!(model.symbolic_fun[1], 1, 2, "x^2")  # 第 1 层，输入 1 -> 输出 2
model.symbolic_fun[1].mask[2, 1] = 1.0
ps, st = Lux.setup(rng, model)
```

乘法节点使用 `[n_sum, n_mult]` 宽度对；`mult_arity` 控制每个乘积节点由多少个子节点列相乘得到。

### 结构操作：`prune` 与 `refine`

```julia
prune(model, ps, st, x; node_th=1e-2, edge_th=3e-2)
prune(model, x; node_th=1e-2, edge_th=3e-2, rng=Random.default_rng())

refine(model, ps, st, x, new_grid)
refine(model, x, new_grid; rng=Random.default_rng())
```

两者都返回 `(model=..., ps=..., st=...)`，且不修改输入。

`prune` 根据样本 `x` 对隐藏节点/边打分，移除低于 `node_th` 的加法节点，并将低于 `edge_th` 的边置零。输入/输出节点以及乘法节点始终保留。`refine` 提高 B 样条网格分辨率（单个整数或逐层向量），同时用 `curve2coef` 重新拟合旧曲线以保持已学到的函数不变；符号层和仿射参数原样复制。

```julia
model = MultKAN([2, 3, 1]; grid=3, k=3)
ps, st = Lux.setup(rng, model)
x = randn(rng, 32, 2)

pruned = prune(model, ps, st, x; node_th=1e-2, edge_th=3e-2)
y = pruned.model(x, pruned.ps, pruned.st)[1]

refined = refine(model, ps, st, x, 10)   # 或逐层使用 [10, 10]
```

## 与 PyKAN 的差异

KANLux 移植了 PyKAN 的*可微核心*，并用 PyKAN 生成的 `.npy` 参考数据对其进行了数值验证。它不是完整 `MultKAN` 类的 Julia 逐一对等实现：PyKAN 那些有状态的便捷方法按设计留在了应用层。

### 已移植

- B 样条数学（`B_batch`、`coef2curve`、`curve2coef`、`extend_grid`）。
- 符号注册表 `SYMBOLIC_LIB` 及其奇异点保护。
- `create_dataset`、`KANLayer`、`SymbolicKANLayer`、`fix_symbolic!`，以及包含数值 + 符号分支和乘法节点的 `MultKAN` 容器。
- 结构操作 `prune` 和 `refine`（对应 PyKAN 的网格更新步骤）。

### 未移植（应用层职责）

以下 PyKAN `kan/MultKAN.py` 方法被有意地从 `src/` 中省略，必须由应用层实现：

| PyKAN 方法 | 应用层对应实现 |
|--------------|------------------------------|
| `fit()` | 你自己的 Lux 训练循环 |
| `plot()` | Makie.jl（+ GraphMakie.jl） |
| `save_ckpt` / `load_ckpt` | 对 `(ps, st)` 使用 BSON.jl / JLD2.jl |
| `auto_symbolic()` | 符号回归搜索循环 |
| `symbolic_formula()` | 用 SymPy.jl / Symbolics.jl 组装表达式 |

#### 训练循环

`Lux.setup` 会生成 `(ps, st)`。损失函数必须只对 `ps` 求导 —— `st`（`grid`、`mask`、`funs`、…）是非训练状态。惯用循环是 `Training.TrainState` + `Training.single_train_step!`，配合 `Optimisers.jl` 规则（`Optimisers` 已经是 Lux 的依赖；请在应用工程中声明它，以及 Zygote 或 Enzyme）：

```julia
# 应用层
using KANLux, Lux, Random, Zygote

rng = Xoshiro(42)
model = MultKAN([2, 3, 1]; grid=3, k=3)
ps, st = Lux.setup(rng, model)

x = randn(rng, 32, 2)
target = randn(rng, 32, 1)

loss(p) = sum(abs2, model(x, p, st)[1] .- target)
grads = Zygote.gradient(loss, ps)[1]     # 只对 ps 求导
```

PyKAN 的网格更新流程（训练 → `refine` → `curve2coef`）可通过导出的 `refine`/`curve2coef` 来表达。PyKAN 风格的 LBFGS 需要在应用层添加 `Optim.jl` 或 `Nonconvex.jl`。两种自动微分后端均已验证：**Zygote**（全覆盖）和 **Enzyme**（反向模式冒烟测试），包括有限差分交叉验证。

#### 绘图

使用 `Makie.jl`（+ 用于 KAN 图布局的 `GraphMakie.jl`）。PyKAN 的 `plot()` 所需的一切均已暴露：

- `st.act_fun[d].mask` —— 边的开/关
- `st.act_fun[d].grid` 和 `ps.act_fun[d].coef` —— 样条曲线
- `st.symbolic_fun[d].funs_name` —— 符号边标签
- `model.width` —— 图结构

#### 检查点

使用 `BSON.jl` 或 `JLD2.jl` 保存/加载 `(ps, st)`。注意 `st.symbolic_fun[d].funs::Array{Function,2}` 无法直接序列化 —— 应持久化 `funs_name` + `affine` + `mask`，并在加载时根据 `SYMBOLIC_LIB` 重建函数数组。

#### 符号自动回归

本包已经提供了 PyKAN 的构建模块：

- 带复杂度分数的 `SYMBOLIC_LIB`，
- `KANLux.fit_symbolic_params`（a/b 网格搜索 + c/d 最小二乘），
- `fix_symbolic!`（边指派）。

应用层负责实现 `auto_symbolic`（对每条边拟合每个库候选函数，按 R² + 复杂度选取，并在 R² 阈值以下剪枝）以及 `symbolic_formula`（例如用 `SymPy.jl`/`Symbolics.jl` 组装最终表达式）。

## 运行测试

```julia
using Pkg
Pkg.activate("KANLux")
Pkg.test()
```

测试套件包括与 PyKAN 生成参考数据的前向值对比、层/容器状态传递检查、梯度路径检查、边界情况，以及全局 Zygote 对有限差分的梯度交叉验证。

