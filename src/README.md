# 第一问计算与交付复现

## 第二问新增复现入口

第二问输出位于\`outputs/01a08b78-ed6b-7e60-b40d-9f266855d1ef/q2\`，论文位于\`docs/result/第二问求解与结果分析.md\`。使用同一Python/Node解释器；项目\`.local-deps\`另含scikit-learn、matplotlib及其依赖。

复现顺序：运行\`tests/test_q2_solver.py\`；运行\`src/q2_solver.py predict\`；分别运行M0—M4的1月影子验证和334天回测；运行\`src/q2_analyze.py --bound\`及\`src/q2_analyze.py\`；运行Node脚本导出\`result2.xlsx\`；最后运行\`src/q2_verify_delivery.py\`。

正式选型结果是M1。若模型或选型变化，需要同步更新复核器。M4是共享控制的受限场景模型，未实现完整时间联合自适应优化。各日优化限时30秒，失败即停止，不自动重试。

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

## 第二问v2：当前正式复现入口

正式结果为 `outputs/q2_review_v2/result2.xlsx`，论文为 `docs/result/第二问求解与结果分析.md`（输出目录另存同版副本）。旧目录及旧Excel保留。以下命令从项目根目录执行，使用现有内置Python、Node及项目`.local-deps`，无需重新下载依赖。

```powershell
$q2Python = 'C:/Users/ALmerb/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$q2Node = 'C:/Users/ALmerb/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
& $q2Python -m unittest discover -s tests -p 'test_q2*.py'
& $q2Python src/q2_review_experiments.py trial
& $q2Python src/q2_review_experiments.py terminal
& $q2Python src/q2_review_experiments.py risk
& $q2Python src/q2_review_experiments.py stress
& $q2Python src/q2_review_experiments.py models
& $q2Python src/q2_review_checks.py
& $q2Python src/q2_rolling_validation.py
& $q2Python -c 'import sys; sys.path.insert(0,"src"); from q2_review_experiments import *; save_run("trial_MPC",Config(controller="F1"),31,38)'
& $q2Python src/q2_review_report.py
& $q2Node src/q2_workbook.mjs --out outputs/q2_review_v2
& $q2Python src/q2_review_verify.py
```

预测缓存复用旧版`forecasts.npz`，其产生过程只用各日以前数据；需要重建时才运行`src/q2_solver.py predict`，随后完整重跑以上实验。不要用旧`q2_analyze.py`生成v2汇总。新建代码文件与原求解器分离，旧版精确重现证据保存在`baseline_reproduction.json`。

v2终端恢复修正了任意窗口中旧包络可能超负载放电的问题：闭合日提前抑制高于目标库存，放电不超过当前负载，缺少库存按5倍费用补购，无法实际到达终端则停止。它仅用于影子窗口最后一天及正式12月31日，不在实际每个月末重置。新旧闭合成本不能混合解释。

正式选择：1月影子比较选择T0和M1；3月起以此前最多56天影子总费用在M1/M3/M4中选择，月内固定。所有真实状态独立连续，不能拼接候选的预存成本。0.8预设不按全年敏感性回调。报告明确新增规则为复审后因果历史回测，而非未接触的最终测试集。

MPC（F1）已通过解析测试并完成7天，费用增加，未扩大到334天；完整场景树仅设计未实现。正式反馈为F0。MPC试验最后一天也用共同闭合反馈；其他日MPC可以按5倍价购入用于充电，全部计费，计划g仍冻结。

报告及验证旧入口在检测到v2协议时默认转到v2；旧版核验可显式加`--legacy`，读取存档论文。Excel脚本默认v2，可用`--out`显式指定历史输出路径。数值结果为求解器生成的静态交付，并非Excel内重新优化公式；修改数据需重新求解、导出和验证。

每个LP/MILP限时30秒，失败立即停止、不自动重试。长期命令若实际超时，按用户约定交由用户执行。输出session id仅表示进程仍在运行，不等于超时错误。
