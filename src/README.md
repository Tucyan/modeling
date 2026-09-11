# 第一问计算与交付复现

## 文件

- `q1_solver.py`：读取典型日数据，求解LP，检查物理约束与原始—对偶最优性证据，保存全部10分钟决策与汇总。
- `q1_workbook.mjs`：读取原result1模板和求解结果，填写指定单元格，保留原工作表结构，导出Excel并生成核查预览。
- `q1_assumption_probe.py`：讨论阶段的效率、初始电量敏感性比较，与正式主结果分开。
- `../tests/test_q1_solver.py`：三个两时段解析算例，核对套利效率、价差阈值、容量与弃光。

## 运行顺序

从项目根目录运行，使用当前桌面环境的内置Python与Node。SciPy 1.18.1位于项目`.local-deps`，其他Python库来自内置环境；`node_modules`为指向内置依赖的目录联接，不包含自行复制的运行时。

```powershell
& 'C:\Users\ALmerb\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tests\test_q1_solver.py
& 'C:\Users\ALmerb\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\q1_solver.py
& 'C:\Users\ALmerb\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' src\q1_workbook.mjs
```

输出目录为`outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q1/`，其中`result1.xlsx`为题目指定交付，`dispatch_10min.csv`为完整决策及状态明细，`solution.json`包含参数、数据哈希、汇总和最优性核查。`verification.json`记录本次对导出Excel、CSV与论文的独立交叉复核，重新生成结果后应重新做交付复核，不能把旧记录视为新结果的验证。

论文片段为`docs/result/第一问求解与结果分析.md`。若模型或输入变化，需同步更新论文表格和论述。

## 边界

正式求解脚本要求LP最优解满足互斥及全部数值核查；若改变数据后不再满足，脚本会停止，不会把该LP结果当作有效调度交付。此时应实现并验证完整MILP求解。脚本设置30秒求解时间上限，不自动重试。
