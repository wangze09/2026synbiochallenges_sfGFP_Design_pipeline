# 2026synbiochallenges_sfGFP_Design_pipeline
# GFP 荧光强度预测与热稳定引导进化设计

基于机器学习的 GFP 突变体荧光强度预测，结合 ThermoMPNN 热稳定性约束，通过遗传算法搜索高亮度、高热稳定性的 sfGFP 工程化突变体序列。

---

## 项目流程概览

```
GFP_data_avGFP.xlsx
        │
        ▼
  突变信息 One-Hot 编码 (238 × 20 = 4760 维特征)
        │
        ▼
  XGBoost 荧光亮度回归模型 (仅对突变信息编码建模.py)
        │
        ▼
  xgb_gfp_brightness_model.joblib
        │
        │        ThermoMPNN (sfGFP, PDB: 2B3P, chain A)
        │                │
        │                ▼
        │     mutation_ddg_thermoMPNN.csv
        │                │
        └────────────────┘
                   │
                   ▼
    热稳定引导遗传进化搜索 (thermo_guided_gfp_evolution_V2.py)
                   │
          ┌────────┴────────┐
          ▼                 ▼
  max_mutations=6       max_mutations=8
  top20_max_mutaion_6   top20_max_mutation_8
  _..._evolved_...csv   _..._evolved_...csv
          │                 │
          └────────┬────────┘
                   │
                   ▼
     FPbase 排除列表过滤 (filter_by_exclusion_list.py)
     Exclusion_List.csv (135,414 条已知序列)
                   │
          ┌────────┴────────┐
          ▼                 ▼
  top20_max_mutaion_6   top20_max_mutation_8
  _..._novel.csv        _..._novel.csv
  (新颖候选序列)         (新颖候选序列)
```

---

## 数据与模型训练

### 数据集

- **文件**：`GFP_data_avGFP.xlsx`
- **来源**：avGFP 荧光蛋白的大规模突变-荧光数据集（约 5 万条以上记录）
- **字段**：`aaMutations`（氨基酸突变描述）、`Brightness`（归一化荧光亮度）
- **采样**：为加速训练，从完整数据集中随机采样 **5000 条**（`random_state=42`）
  - 训练集：4000 条
  - 测试集：1000 条

### 特征编码（One-Hot）

- **蛋白质长度**：238 个氨基酸位点
- **氨基酸字母表**：20 种标准氨基酸（`ACDEFGHIKLMNPQRSTVWY`）
- **特征维度**：238 × 20 = **4760 维**（加上 `mutation_count` 共 4761 维）
- **编码方式**：对每条突变信息中的每个单点突变，将对应 `pos{i}_{AA}` 特征位设为 1，其余为 0
- **位置偏移**：`POSITION_OFFSET = 1`，即数据集中标注位点需 +1 才对应实际蛋白序列位置

### XGBoost 模型

- **文件**：`仅对突变信息编码建模.py`
- **模型**：`XGBRegressor`
- **超参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_estimators` | 500 | 树的数量 |
| `max_depth` | 4 | 每棵树最大深度 |
| `learning_rate` | 0.03 | 学习率（步长） |
| `subsample` | 0.8 | 每棵树使用 80% 样本 |
| `colsample_bytree` | 0.8 | 每棵树使用 80% 特征 |
| `reg_alpha` | 0.1 | L1 正则化 |
| `reg_lambda` | 1.0 | L2 正则化 |
| `objective` | `reg:squarederror` | 均方误差回归目标 |

### 测试集性能

| 指标 | 值 |
|------|----|
| R² | 0.5830 |
| RMSE | 0.6776 |
| MAE | 0.5395 |

- **模型保存文件**：`xgb_gfp_brightness_model.joblib`（以 `joblib` 字典格式存储模型、特征名、蛋白长度等元信息）

---

## 热稳定性预测（ThermoMPNN）

### 工具

使用 [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) 提供的 Colab 接口，输入 sfGFP 的 PDB 结构（PDB ID: **2B3P**，链：**A**），预测蛋白质每个可突变位点处所有单点突变对折叠自由能的影响（ΔΔG，单位 kcal/mol）。

### 输出文件

- **文件**：`mutation_ddg_thermoMPNN.csv`
- **关键字段**：
  - `Mutation`：突变标识（如 `A2G`）
  - `pos`：ThermoMPNN 输出的位点编号（需 **+2** 才与 GFP 实际位点对应）
  - `wtAA` / `mutAA`：野生型 / 突变型氨基酸
  - `ddG (kcal/mol)`：ΔΔG 值，负值表示稳定化突变，正值表示去稳定化突变

> **位点校正说明**：ThermoMPNN 对 2B3P 链 A 给出的 `pos` 编号需要 `+2` 才与 avGFP/sfGFP 实际序列编号对齐，代码中通过 `THERMOMPNN_POSITION_SHIFT = 2` 自动完成校正。

---

## 热稳定引导遗传进化（thermo_guided_gfp_evolution_V2.py）

### 设计思路

比赛评分代理公式（近似真实实验评分）：

```
competition_proxy_score = (F_initial / F_initialWT) × estimated_heat_retention
```

- `F_initial / F_initialWT`：候选序列相对 sfGFP 的**初始荧光相对亮度**（由 XGBoost 模型预测）
- `estimated_heat_retention`：热处理后荧光**保留率估计**（由 ThermoMPNN ddG_sum 推算）
- 若候选序列亮度低于 sfGFP 的 30%，直接记 0 分（低亮度门控）

### 进化算法流程

1. **初始化种群**：以 sfGFP 为起始，预播种 ThermoMPNN 最稳定的单点突变体，再随机/热稳定引导生成多样化初始种群
2. **每代评分**：预测荧光亮度 → 计算热稳定指标 → 计算比赛代理分数 → 综合选择分数排序
3. **选择**：精英保留 + 父代池采样
4. **交叉 + 变异**：随机交叉两亲本序列，再以一定概率使用 ThermoMPNN 引导突变
5. **约束执行**：保护关键位点不变；超过最大突变数时优先回退去稳定化突变
6. **循环 80 代**，记录所有历史最优序列

### 参数说明

#### 序列参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `AVGFP_WT_SEQUENCE` | avGFP 全长序列 | XGBoost 模型的参考序列（one-hot 编码基准） |
| `SFGFP_START_SEQUENCE` | sfGFP 全长序列 | 进化搜索的起始序列 |
| `PROTECTED_POSITIONS` | `{1,30,39,65,80,99,105,145,153,163,171,206}` | 不允许突变的关键位点（1-based），包含生色团形成及折叠关键残基 |

#### 比赛评分代理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LOW_BRIGHTNESS_RELATIVE_THRESHOLD` | 0.30 | 相对 sfGFP 亮度低于此阈值时直接记 0 分（30% 门控） |
| `COMPETITION_PROXY_WEIGHT` | 0.85 | 综合选择分数中，比赛代理分数的权重 |
| `RANK_TIEBREAKER_WEIGHT` | 0.15 | 综合选择分数中，排名辅助项的权重（用于打破平局、避免早期收敛） |
| `AUX_BRIGHTNESS_RANK_WEIGHT` | 0.70 | 辅助排名中，亮度排名的比例 |
| `AUX_THERMO_RANK_WEIGHT` | 0.30 | 辅助排名中，热稳定排名的比例 |
| `HEAT_RETENTION_DDG_SCALE` | 1.0 | 热保留率估算的 ddG 缩放系数；值越小对去稳定化突变惩罚越强。计算公式：`exp(-max(ddG_sum,0) / scale)` |
| `HEAT_RETENTION_FLOOR` | 0.02 | 估算热保留率的最低值（避免数值下溢至 0） |
| `THERMO_UNSCORED_MUTATION_PENALTY` | 0.15 | 对 ThermoMPNN 未覆盖的突变位点，每个扣减 0.15 的热稳定分（保守惩罚） |

#### 热稳定引导突变参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `THERMO_SELECTION_WEIGHT` | 0.2 | 热稳定分数在选择中的权重（内部辅助） |
| `THERMO_GUIDED_MUTATION_PROB` | 0.5 | 突变时以 ThermoMPNN 引导的概率；剩余概率为随机突变 |
| `THERMO_GUIDED_MAX_DDG` | 0.0 | 热引导突变只接受 ddG ≤ 此值的突变（0.0 表示只允许稳定化或中性突变） |
| `THERMO_SAMPLING_TEMPERATURE` | 0.5 | Boltzmann 采样温度；越低越倾向于选择 ddG 最负（最稳定）的突变 |
| `THERMO_PRESEED_SINGLE_MUTANTS` | 100 | 初始化种群时预播种的 ThermoMPNN 最稳定单点突变体数量 |
| `THERMOMPNN_POSITION_SHIFT` | 2 | ThermoMPNN 输出 pos 与 GFP 实际位点的偏移量（corrected_pos = pos + 2） |

#### 进化算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POPULATION_SIZE` | 300 | 每代种群大小 |
| `N_GENERATIONS` | 80 | 进化代数 |
| `MAX_NEW_MUTATIONS_VS_SFGFP` | 6 或 8 | 相对 sfGFP 最多允许新增的突变数（两次运行分别使用 6 和 8） |
| `MIN_MUTATIONS_PER_STEP` | 1 | 每次变异操作最少添加的突变数 |
| `MAX_MUTATIONS_PER_STEP` | 3 | 每次变异操作最多添加的突变数 |
| `ELITE_FRAC` | 0.15 | 每代直接保留到下一代的精英个体比例（15%） |
| `PARENT_POOL_FRAC` | 0.40 | 参与交叉繁殖的父代池比例（取前 40% 的个体） |
| `TOP_N` | 20 | 最终输出的 Top N 序列数量 |
| `RANDOM_SEED` | 42 | 随机种子，保证结果可复现 |

#### 三阶段动态策略

进化过程按代数进度自动分为三个阶段，各阶段参数动态调整：

| 阶段 | 进度区间 | 特点 |
|------|----------|------|
| **早期探索** (`early_exploration`) | 前 30% 代 | 热引导概率降低、ddG 阈值放松（允许 ≤0.5）、采样温度升高；鼓励多样性探索 |
| **中期平衡** (`middle_balanced`) | 30%–70% 代 | 使用基础参数，兼顾探索与收敛 |
| **后期收敛** (`late_exploitation`) | 后 30% 代 | 热引导概率提升（≥0.70）、ddG 阈值收紧（≤-0.1）、采样温度降低；专注挑选稳定突变 |

---

## 输出文件

### 进化搜索结果

| 文件 | 说明 |
|------|------|
| `max_mutaion_6_all_competition_proxy_thermo_guided_evolutionary_search_results_from_sfGFP.csv` | `MAX_NEW_MUTATIONS=6` 时全部历史候选序列（去重后按分数排序） |
| `max_mutation_8_all_competition_proxy_thermo_guided_evolutionary_search_results_from_sfGFP.csv` | `MAX_NEW_MUTATIONS=8` 时全部历史候选序列（去重后按分数排序） |
| `top20_max_mutaion_6_competition_proxy_thermo_guided_evolved_sequences_from_sfGFP.csv` | `MAX_NEW_MUTATIONS=6` 进化搜索 Top 20 候选序列 |
| `top20_max_mutation_8_competition_proxy_thermo_guided_evolved_sequences_from_sfGFP.csv` | `MAX_NEW_MUTATIONS=8` 进化搜索 Top 20 候选序列 |
| `top20_max_mutaion_6_..._novel.csv` | Top 20 (max_6) 经 FPbase 排除列表过滤后的新颖序列 |
| `top20_max_mutation_8_..._novel.csv` | Top 20 (max_8) 经 FPbase 排除列表过滤后的新颖序列 |

### FPbase 排除列表

- **文件**：`Exclusion_List.csv`
- **来源**：FPbase 荧光蛋白数据库全量序列（135,414 条，包含 82,351 条长度为 238 的序列）
- **用途**：过滤掉进化结果中已存在于数据库的已知序列，确保最终候选序列的新颖性

### 输出列说明

| 列名 | 说明 |
|------|------|
| `sequence` | 候选蛋白序列（全长氨基酸序列） |
| `aaMutations_vs_sfGFP` | 相对 sfGFP 的突变描述 |
| `mutation_count_vs_sfGFP` | 相对 sfGFP 的突变数量 |
| `aaMutations_vs_avGFP` | 相对 avGFP（模型参考序列）的突变描述 |
| `mutation_count_vs_avGFP` | 相对 avGFP 的突变数量 |
| `predicted_brightness` | XGBoost 模型预测的荧光亮度 |
| `relative_initial_brightness_vs_sfGFP` | 相对 sfGFP 预测亮度的比值（`F_initial / F_initialWT`） |
| `estimated_heat_retention` | 基于 ThermoMPNN ddG_sum 估算的热处理后荧光保留率 |
| `passes_low_brightness_gate` | 是否通过 30% 低亮度门控（True/False） |
| `competition_proxy_score` | 比赛代理分数（通过门控时 = `relative_brightness × heat_retention`，否则为 0） |
| `competition_proxy_score_raw` | 未经门控的原始代理分数 |
| `selection_score` | 综合选择分数（0.85 × 代理分 + 0.15 × 辅助排名分） |
| `thermo_ddG_sum` | 所有已评分突变 ddG 之和（负值更优） |
| `thermo_ddG_mean` | 所有已评分突变 ddG 均值 |
| `thermo_stability_score` | 热稳定综合分（= -ddG_sum - 未评分突变惩罚，越高越稳定） |
| `thermo_n_scored_mutations` | ThermoMPNN 覆盖的突变数量 |
| `thermo_n_unscored_mutations` | ThermoMPNN 未覆盖的突变数量 |
| `thermo_stabilizing_mutations_vs_sfGFP` | 稳定化突变列表（ddG < 0） |
| `thermo_destabilizing_mutations_vs_sfGFP` | 去稳定化突变列表（ddG > 0） |
| `thermo_unscored_mutations_vs_sfGFP` | ThermoMPNN 未评分突变列表 |
| `generation` | 该序列出现的进化代数 |
| `search_stage` | 该代所处的搜索阶段 |
| `brightness_rank_pct` | 每一代中亮度的排名百分比 |
| `thermo_rank_pct` | 每一代中热稳定性的排名百分比） |
| `auxiliary_rank_score` | 辅助排名分（亮度排名 + 热稳定排名加权） |

---

## 环境依赖

```
python >= 3.8
numpy
pandas
scikit-learn
xgboost
joblib
openpyxl       # 用于读取 .xlsx 文件
```

---

## 快速复现

### Step 1：训练荧光预测模型

```bash
python 仅对突变信息编码建模.py
# 输出：xgb_gfp_brightness_model.joblib
```

### Step 2：获取 ThermoMPNN ddG 预测

使用 [ThermoMPNN Colab](https://github.com/Kuhlman-Lab/ThermoMPNN) 输入 PDB ID `2B3P`（chain A），下载输出结果并重命名为：

```
mutation_ddg_thermoMPNN.csv
```

### Step 3：运行进化搜索

```bash
# max_mutations = 6
python thermo_guided_gfp_evolution_V2.py
# 修改 MAX_NEW_MUTATIONS_VS_SFGFP = 8 后再次运行可得 max_mutations = 8 结果
```

每次运行输出两个文件：
- `top20_max_mutaion_6_..._evolved_sequences_from_sfGFP.csv`（Top 20 候选序列）
- `max_mutaion_6_all_..._evolutionary_search_results_from_sfGFP.csv`（全部历史记录）

### Step 4：FPbase 排除列表过滤

将 Top 20 候选序列与 FPbase 数据库比对，过滤掉已知序列，保留新颖候选：

```bash
python filter_by_exclusion_list.py
```

输出：
- `top20_max_mutaion_6_..._novel.csv`
- `top20_max_mutation_8_..._novel.csv`

> 过滤逻辑：对候选序列做精确字符串匹配（大小写不敏感），命中 Exclusion_List 的序列直接剔除。

---

## 参考

- ThermoMPNN: [https://github.com/Kuhlman-Lab/ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN)
- avGFP 突变数据集：Sarkisyan et al., *Nature*, 2016
- sfGFP 结构：PDB 2B3P
