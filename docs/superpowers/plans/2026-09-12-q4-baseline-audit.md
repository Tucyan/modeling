# Q4 Baseline Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不运行优化、不修改冻结文件的前提下，完成 Q4 项目结构、模型流程、正式产物、验证证据和潜在评委质疑点的基线审计。

**Architecture:** 仅使用只读文件清单、源码定位、JSON/CSV 表头与 SHA-256 提取事实。审计报告引用冻结文件并把“已有证据”和“冻结产物缺口”分开，不生成实验结果。

**Tech Stack:** PowerShell 只读命令、Markdown。

**Spec:** `docs/q4/Q4_review_analysis_design.md`

## Global Constraints

- 只处理 `D:\Model_Q4\modeling-main`。
- 不执行 Git merge、优化、回放、模型训练、依赖安装或大规模实验。
- 不修改 Q1—Q3、Q4 核心 solver、正式参数和 `outputs/q4_final/`。
- 本轮只新增本计划和 `docs/q4/Q4_baseline_audit.md`。

---

### Task 1: 冻结项目清单与证据定位

**Files:**
- Read: `src/`
- Read: `outputs/q4_final/`
- Read: `final_results/`
- Read: `docs/`
- Read: `paper/`
- Create: `docs/q4/Q4_baseline_audit.md`

**Interfaces:**
- Consumes: 当前完整项目快照。
- Produces: 可追溯的入口、配置、结果、账本和验证文件清单。

- [x] **Step 1: 列出指定目录和 Q4 文件**

运行只读文件清单，分别确认 `src/`、不存在或存在的 `src/q4/`、`outputs/`、`outputs/q4_final/`、`final_results/`、`docs/` 和 `paper/`。

- [x] **Step 2: 提取冻结配置与正式结果摘要**

读取 `selection_protocol.json`、`summary.json`、`delivery_verification.json`、正式日账本表头和价格预测记录表头；只输出必要字段，不执行现有求解入口。

- [x] **Step 3: 定位模型与执行流程源码**

从 `q4_solver.py`、`q4_experiments.py`、`q4_forecast.py`、`dispatch_control.py`、`q3_solver.py` 和 Q4 工作簿/验证脚本提取变量、目标、约束、F0 和结算位置。

### Task 2: 撰写基线审计报告

**Files:**
- Create: `docs/q4/Q4_baseline_audit.md`

**Interfaces:**
- Consumes: Task 1 的只读证据。
- Produces: 用户要求的七部分基线审计。

- [x] **Step 1: 写入模型、输入、优化和执行流程**

报告明确区分规划价格和实际结算价格、Q4-2 与 Q4-3、规划变量与 F0 实际流量。

- [x] **Step 2: 写入正式产物和验证证据**

列出 N0/N1 正式工作簿、逐槽账本、日账本、合同版本、预测记录、协议、摘要和验证器。

- [x] **Step 3: 写入评委可能质疑点**

至少覆盖计划储能轨迹未落盘、收益归因缺少全年正交消融、P0p/P1p 无全年经济成本、终端 SOC 敏感性未运行、预测历史窗口字段粒度、现有工作簿验证范围和 2025 因果历史回测边界。

### Task 3: 审计报告验收

**Files:**
- Verify: `docs/q4/Q4_baseline_audit.md`
- Verify unchanged: `src/`, `outputs/q4_final/`, Q1—Q3 正式文件。

**Interfaces:**
- Consumes: 完成的审计报告。
- Produces: 本轮只新增审计材料且冻结文件未变的证据。

- [x] **Step 1: 检查报告完整性**

确认七个要求部分、关键路径、N0/N1 配置、334 天/48,096 槽和证据限制均出现，且没有 `TBD`、`TODO` 或虚构实验数值。

- [x] **Step 2: 比较受保护文件清单**

确认本轮没有更新 `src/`、`outputs/q4_final/`、Q1—Q3、`docs/` 中的既有文件、`final_results/` 或 `paper/` 中的文件时间和内容；新增文件仅为计划、既有设计规格和审计报告。

- [x] **Step 3: 最终只读复核**

重新读取报告和正式摘要，核对路径、费用、参数、结算口径及证据分级后再报告完成状态。
