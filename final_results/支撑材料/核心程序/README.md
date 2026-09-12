# Reproduction Code Snapshot

本目录是与最终成果匹配的最小源码快照。运行入口和参数说明以仓库根目录 `src/README.md` 为准；脚本中的路径约定仍以仓库根目录为工作目录。`requirements.txt` 列出 Python 依赖，工作簿导出脚本需要项目已有的 Node 运行时和 `@oai/artifact-tool`。

纳入的脚本仅覆盖 Q1 求解/导出、Q2 review v2、Q3 unified v1、Q4 final 的求解、报告、工作簿和交付核验链；未纳入单元测试、旧版求解器、讨论阶段探针和独立实验脚本。

本支撑材料的核心程序还包含四个正式 Excel 展示脚本：`q1_workbook.mjs`、`q2_workbook.mjs`、`q3_workbook.mjs`、`q4_workbook.mjs`。它们不重新求解模型，但负责将已核验结果写入题目模板，因此属于成果交付链。

## 必要性分层

这里的“必要”是指能否重新计算并验证论文中的正式数值；“可选”是指不改变模型和数值，只负责把结果整理成论文表格、Excel 或展示材料。

| 层级 | 程序 | 用途 | 是否影响正式数值 |
| --- | --- | --- | --- |
| 核心必需 | `q1_solver.py`、`q2_solver.py`、`q3_solver.py`、`q4_solver.py` | 四问优化模型和物理约束求解 | 是 |
| 核心必需 | `q3_forecast.py`、`q4_forecast.py` | Q3 风险储备、Q4 历史价格预测 | 是 |
| 核心必需 | `dispatch_control.py`、`q2_realtime_control.py`、`q3_q2_bridge.py` | 实时反馈和问题间接口 | 是 |
| 完整重算必需 | `q2_review_experiments.py`、`q3_experiments.py`、`q4_experiments.py` | 候选冻结、因果回测、正式期回放和 N0/N1 选型 | 是；已有正式结果时可不重复运行 |
| 独立验证必需 | `q2_review_checks.py`、`q2_review_verify.py`、`q3_verify_delivery.py`、`q4_verify_delivery.py` | 核验日期、账本、物理约束、费用和工作簿 | 不生成数值，但决定结果能否验收 |
| 辅助必需 | `q2_rolling_validation.py`、`q2_analyze.py`、`q3_workbook_data.py`、`q4_workbook_data.py` | 为选型、核验或工作簿整理中间结构 | 间接；被正式流程调用 |
| 展示可选 | `q1_workbook.mjs`、`q2_workbook.mjs`、`q3_workbook.mjs`、`q4_workbook.mjs` | 把结果写入题目 Excel 模板 | 否，不改变求解结果 |
| 展示可选 | `q2_review_report.py`、`q3_report.py`、`q4_report.py` | 生成论文汇总、指定日期表格和展示材料 | 否，不改变求解结果 |

最小验证集合是“核心必需 + 独立验证必需”，再加上对应问题的输入缓存；要从原始输入完整重算，则还需运行“完整重算必需”程序。工作簿导出和报告程序只在需要重新生成最终展示文件时运行。

## 程序说明

### 第一问

- `q1_solver.py`：读取典型日负荷、光伏和电价，求解储能日内调度 LP，并检查能量平衡、储能边界、充放电互斥和最优性证书。
- `q1_workbook.mjs`：把 Q1 求解结果写入题目提供的 `result1.xlsx` 模板，生成最终工作簿。

### 第二问

- `q2_solver.py`：实现 Q2 的日内储能优化器，支持计划购电、实时执行、储能状态连续和紧急购电。
- `q2_review_experiments.py`：执行 Q2 review v2 的因果回测、候选策略比较、月度滚动选型和正式期回放。
- `q2_realtime_control.py`：提供 Q2 日内实时反馈控制器的执行逻辑。
- `dispatch_control.py`：共享 F0 反馈控制步骤，被 Q2/Q3/Q4 的执行逻辑复用。
- `q2_rolling_validation.py`：汇总和检查 Q2 月度滚动选型结果。
- `q2_review_checks.py`：执行 Q2 review v2 的协议、参数和结果一致性检查。
- `q2_review_report.py`：根据 Q2 正式回放结果生成结果分析材料和汇总数据。
- `q2_review_verify.py`：独立核验 Q2 的调度明细、物理约束、费用、连续状态、论文和工作簿。
- `q2_analyze.py`：提供 Q2 报告和核验所需的时段汇总辅助函数。
- `q2_workbook.mjs`：把 Q2 结果载荷写入 `result2.xlsx` 模板并导出工作簿。

### 第三问

- `q3_forecast.py`：生成 Q3 风险储备计算所需的预测残差样本和风险分位数。
- `q3_solver.py`：实现 Q3 计划、调整费结算、储能状态转移和日调度求解。
- `q3_q2_bridge.py`：把 Q2 的正式调度结果和预测接口转换为 Q3 可用的输入。
- `q3_experiments.py`：执行 Q3 unified smoke、正式回放、N0/N1 选型和消融流程。
- `q3_workbook_data.py`：由 Q3 调度明细生成工作簿所需的结构化载荷。
- `q3_workbook.mjs`：把 Q3 载荷写入 `result3.xlsx` 模板并导出工作簿。
- `q3_report.py`：生成 Q3 指定日期、费用和风险结果分析材料。
- `q3_verify_delivery.py`：独立核验 Q3 两种结算口径的调度账本、物理约束、日期完整性和工作簿。

### 第四问

- `q4_forecast.py`：仅使用历史可见价格生成 Q4 的价格预测模型和预测值。
- `q4_solver.py`：实现 Q4-2/Q4-3 的储能调度、价格结算、风险储备和实时价格更新。
- `q4_experiments.py`：执行 Q4 N0/N1 候选冻结、334 天因果回放和正式结果汇总。
- `q4_workbook_data.py`：由 Q4 调度明细生成两个结算口径的工作簿载荷。
- `q4_workbook.mjs`：把 Q4-2/Q4-3 载荷写入对应 Excel 模板并导出工作簿。
- `q4_report.py`：从 Q4 最终账本生成论文要求的指定日期表格。
- `q4_verify_delivery.py`：独立核验 Q4 四条账本、价格版本、物理约束、费用、日期和工作簿结构。

### 依赖文件

- `requirements.txt`：Python 运行依赖版本范围。
- `README.md`：本目录的源码范围、运行边界和文件用途说明。
