# Q4 Consistency, Attribution and Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结Q4正式模型和结果的条件下，完成规划—执行证据检查、费用归因和月度稳定性派生分析。

**Architecture:** 新增独立分析模块读取`outputs/q4_final/N0`，不调用`src/q4_experiments.py`或任何优化函数。计划轨迹缺失时输出不可识别结论，并通过独立F0逐槽重放验证实际执行；收益使用账本会计恒等式和已有1月候选证据；稳定性由正式日账本直接聚合。

**Tech Stack:** Python 3.12、NumPy、pandas、标准库SVG、Markdown。

**Spec:** `docs/q4/Q4_review_analysis_design.md`

## Global Constraints

- 不修改或调用Q1—Q3、Q4核心solver、目标、约束、正式参数和正式求解入口。
- 不覆盖`outputs/q4_final/`或`final_results/`。
- 不下载依赖；图表使用标准SVG。
- 新增文件仅位于`analysis/q4_review/`、`outputs/q4/`、`docs/q4/`和本计划。
- 所有计划储能偏差结论必须来自持久化计划字段；字段不存在时标为不可识别。

---

### Task 1: 锁定分析接口行为

**Files:**
- Create: `analysis/q4_review/validate_stage2.py`
- Create: `analysis/q4_review/stage2_analysis.py`

**Interfaces:**
- Consumes: pandas DataFrame和单槽F0输入。
- Produces: `compare_plan_execution(plan, actual)`、`f0_step_independent(...)`、`accounting_attribution(...)`、`monthly_stability(...)`。

- [x] **Step 1: 先写验证脚本**

验证脚本用手算样例断言：有盈余20 kWh时F0充电20 kWh并使SOC增加18 kWh；低SOC且缺口100 kWh时紧急购电100 kWh；计划字段缺失时接口返回不可识别；费用分项`150-120=30`元严格对账；两天收益`10,-5`汇总为5元、1个收益日和1个损失日。

- [x] **Step 2: 运行并确认失败**

运行：

```powershell
& 'C:\Users\Wellerman\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' analysis/q4_review/validate_stage2.py --self-test
```

预期因`stage2_analysis`尚不存在而失败，证明验证入口能拦截缺失实现。

- [x] **Step 3: 实现最小分析接口**

独立实现F0公式、可选计划CSV比较、会计归因和月度聚合；不得从`src/`导入执行器或优化器。

- [x] **Step 4: 重新运行自检**

预期输出`SELF_TEST=PASS`。

### Task 2: 生成规划—执行一致性分析

**Files:**
- Modify: `analysis/q4_review/stage2_analysis.py`
- Create: `outputs/q4/consistency/plan_record_availability.json`
- Create: `outputs/q4/consistency/f0_replay_residuals.csv`
- Create: `outputs/q4/consistency/f0_replay_daily.csv`
- Create: `outputs/q4/consistency/f0_replay_monthly.csv`
- Create: `outputs/q4/consistency/contract_execution_monthly.csv`
- Create: `outputs/q4/consistency/summary.json`
- Create: `docs/q4/plan_execution_consistency.md`

**Interfaces:**
- Consumes: N0 Q4-2/Q4-3逐槽账本和当前源码/载荷结构。
- Produces: 计划记录可用性、F0重放误差和一致性报告。

- [x] **Step 1: 检查持久化计划字段**

扫描正式CSV表头及工作簿载荷键，确认是否存在`charge_plan`、`discharge_plan`、`SOC_plan`或同义字段，并记录产生逻辑存在但未落盘的源码位置。

- [x] **Step 2: 独立重放F0**

对N0两条48,096槽账本逐槽重建`c,d,e,unused,spill,E_end`；12月31日使用冻结的末日6000闭合，其余日期不闭合。

- [x] **Step 3: 汇总误差与合同执行指标**

按日、月和全年统计平均绝对误差、最大绝对误差和标准差；另行统计`q-g`与`actual_grid-q`，不得把它们称作计划储能偏差。

- [x] **Step 4: 生成报告**

报告分别回答计划轨迹是否可识别、F0实际执行是否一致、已有证据能否证明偏差可控及对成本结论的影响。

### Task 3: 生成收益归因分析

**Files:**
- Modify: `analysis/q4_review/stage2_analysis.py`
- Create: `outputs/q4/ablation/annual_strategy_comparison.csv`
- Create: `outputs/q4/ablation/annual_accounting_attribution.csv`
- Create: `outputs/q4/ablation/monthly_accounting_attribution.csv`
- Create: `outputs/q4/ablation/existing_january_ablation.csv`
- Create: `outputs/q4/ablation/daily_savings.csv`
- Create: `outputs/q4/ablation/summary.json`
- Create: `docs/q4/revenue_attribution.md`

**Interfaces:**
- Consumes: N0日账本、N0协议中的既有1月候选评分和正式摘要。
- Produces: 全年会计分项归因、已有局部消融证据和不可分离项说明。

- [x] **Step 1: 对账全年策略费用**

由334天日账本重新汇总Q4-2与Q4-3费用、节省金额和节省比例。

- [x] **Step 2: 计算会计分项贡献**

按`plan_cost + up_cost + down_fee - refund + emergency_cost`分解总节省，并要求分项和与总节省误差不超过`1e-6`元。

- [x] **Step 3: 整理已有1月候选**

读取相同风险参数下R0、Rall/`gamma=0`及Rall/`gamma=1`已有17天成本；明确短窗口证据不能替代全年正交消融。

- [x] **Step 4: 生成归因报告**

动态价格正式贡献标为0；信息更新与滚动调整只报告合并效果和证据边界，不伪造独立比例。

### Task 4: 生成月度稳定性分析与图表

**Files:**
- Modify: `analysis/q4_review/stage2_analysis.py`
- Create: `outputs/q4/stability/monthly_stability.csv`
- Create: `outputs/q4/stability/daily_benefit_statistics.json`
- Create: `outputs/q4/stability/monthly_savings.svg`
- Create: `outputs/q4/stability/daily_benefit_histogram.svg`
- Create: `outputs/q4/stability/cumulative_savings.svg`
- Create: `docs/q4/stability_analysis.md`

**Interfaces:**
- Consumes: `outputs/q4/ablation/daily_savings.csv`。
- Produces: 12个月表、每日分布、累计曲线和异常日集中度指标。

- [x] **Step 1: 聚合月度稳定性**

计算每月两策略成本、节省、节省率、收益/损失天数、均值、中位数、标准差、极值和累计节省。

- [x] **Step 2: 计算全年分布与集中度**

统计正收益天数占比、损失天数占比、分位数、TOP5正收益占全年节省比例及剔除TOP5后的剩余节省。

- [x] **Step 3: 生成标准SVG图表**

生成12个月节省柱状图、每日收益直方图和累计收益曲线，图中数值来自同一派生CSV。

- [x] **Step 4: 生成稳定性报告**

根据月份符号、正收益天数、累计曲线和TOP5集中度回答优势是否由少数异常日驱动。

### Task 5: 独立验收和冻结检查

**Files:**
- Modify: `analysis/q4_review/validate_stage2.py`
- Create: `outputs/q4/stage2_verification.json`
- Verify unchanged: `src/`, `outputs/q4_final/`, `final_results/`和Q1—Q3。

**Interfaces:**
- Consumes: Tasks 2—4全部输出。
- Produces: 数值、结构、日期、会计恒等式、SVG存在性和冻结哈希验证结果。

- [x] **Step 1: 运行分析生成器**

运行：

```powershell
& 'C:\Users\Wellerman\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' analysis/q4_review/stage2_analysis.py
```

- [x] **Step 2: 运行完整验证**

运行：

```powershell
& 'C:\Users\Wellerman\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' analysis/q4_review/validate_stage2.py --verify
```

要求输出`VERIFY=PASS`，334天无重复，月度恰为12行，会计差小于`1e-6`，F0最大误差小于`1e-6`，三张SVG非空。

- [x] **Step 3: 复核保护哈希**

重新计算基线审计中冻结的Q4核心、N0账本和主工作簿SHA-256；任何变化均失败。
