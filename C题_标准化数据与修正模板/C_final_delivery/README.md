# C题数据标准化交付说明

## 1. 目录结构

```text
C_final_delivery/
├─ README.md
├─ standardized_data.xlsx
├─ templates/
│  ├─ result1.xlsx
│  ├─ result2.xlsx
│  ├─ result3.xlsx
│  ├─ result4-2.xlsx
│  └─ result4-3.xlsx
└─ data_csv/
   ├─ time_dimension_10min.csv
   ├─ typical_day_10min.csv
   ├─ annual_actual_10min.csv
   ├─ pv_forecast_hourly.csv
   ├─ quality_report.csv
   └─ data_dictionary.csv
```

`templates/` 中仅保留题目要求的 Excel 模板，不生成模板的 CSV 版本。

## 2. 时间轴修正

原始结果模板的 10 分钟时间段存在整体错位，例如从 `0:10-0:20` 开始，并在末尾出现 `0:00-0:10+1`。

现已统一为一天 144 个连续区间：

```text
00:00-00:10
00:10-00:20
...
23:40-23:50
23:50-24:00
```

同时定义：

- `interval_id = 0,1,...,143`
- 每个区间长度为 10 min = 1/6 h
- 第 0 个区间对应 `00:00-00:10`
- 第 143 个区间对应 `23:50-24:00`

修正已应用到所有相关模板中的“计划购电量”和“调整购电量”时间表头。

## 3. 附件1标准化

附件1的“时间”列存在 Excel 数值时间与文本时间混合的问题，因此不再直接把原始时间字段作为关联主键，而是按照原始 144 行的固定顺序重建标准时间轴。

保留原始：

- 电价（元/kWh）
- 小区负载（kW）
- 光伏发电预测功率（kW）

并增加 10 分钟电量字段：

```text
energy_kWh = power_kW × 1/6 h
```

对应文件：`data_csv/typical_day_10min.csv`。

## 4. 附件2与附件4标准化

附件2包含 2025 年全年实际小区负载和实际光伏功率；附件4包含全年实际电价。

处理后：

- 日期统一为 `YYYY-MM-DD`
- 每天严格映射到 144 个 10 分钟区间
- 使用 `(date, interval_id)` 作为唯一时间键
- 附件2负载、附件2实际光伏和附件4实际电价合并为统一建模长表
- 原始 kW 数据完整保留
- 新增对应的 10 分钟 kWh 字段

对应文件：`data_csv/annual_actual_10min.csv`，共 `365×144=52560` 条记录。

## 5. 附件3标准化

附件3每天在 `0:00 / 6:00 / 12:00 / 18:00` 发布未来 24 小时光伏预测。

原始文件只有每日 `0:00` 行明确写出日期，随后 `6:00 / 12:00 / 18:00` 行日期单元格为空；部分解析方式可能把这些空共享字符串错误显示为类似 `28` 的值。

处理时按当天 `0:00` 行的真实日期向下补齐，形成完整：

- `issue_date`
- `issue_time`
- `issue_datetime`
- `horizon_hour`
- `forecast_window_start`
- `forecast_window_end`
- `target_date`
- `target_time`
- `pv_forecast_kw`

对应文件：`data_csv/pv_forecast_hourly.csv`，共 `365×4×24=35040` 条预测记录。

当前没有把小时级预测擅自插值成 10 分钟数据，因为插值方法属于后续建模假设，而不是纯数据清洗步骤。

## 6. 数据质量检查

初步审计确认：

- 附件2与附件4日期覆盖 `2025-01-01` 至 `2025-12-31`，共 365 天
- 每日均有 144 个 10 分钟数据点
- 附件3共有 1460 次预测发布，即 `365×4`
- 附件3共有 35040 个小时预测值，即 `1460×24`
- 核心数值字段未发现缺失值
- 未发现负的负载或光伏功率
- 未发现非正电价
- `(date, interval_id)` 主键无重复
- 未对数值做异常值删除、截尾或平滑处理

详细检查结果见 `data_csv/quality_report.csv`。

## 7. 建模使用注意事项

`annual_actual_10min.csv` 中的 `load_kw`、`pv_actual_kw`、`price_yuan_per_kwh` 是实际发生值。

在需要“提前制定计划”的问题中，不能直接读取计划时点之后的真实数据，否则会产生未来信息泄漏。例如在 `2025-03-20 00:00` 制定当天购电计划时，不应把当天 14:00 的实际负载或实际光伏当作已知输入。

建议后续明确区分：

- 预测/计划阶段可用数据
- 实际运行数据
- 事后评价数据

## 8. 文件用途

- `time_dimension_10min.csv`：标准 144 区间时间维表
- `typical_day_10min.csv`：附件1标准化典型日数据
- `annual_actual_10min.csv`：附件2+附件4合并后的全年 10 分钟实际数据
- `pv_forecast_hourly.csv`：附件3小时级滚动光伏预测长表
- `quality_report.csv`：数据质量审计结果
- `data_dictionary.csv`：标准化字段解释
- `standardized_data.xlsx`：上述标准数据的 Excel 核查版
- `templates/*.xlsx`：已修正时间轴、供最终填写结果的原格式模板
