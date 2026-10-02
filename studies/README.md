# 研究结果存档

由脚本生成、提交留档的研究结果(`reports/` 不入库,这里是挑出来保存的)。

## 滞后量化(lag_study.py,2026-10-02)

- `lag_study_binance.md`:币安历史(backfill_vision.py 回填,77 个币,2025-03-01 ~ 2026-10-01,1415 段大涨)。
  生成命令:`python lag_study.py --db --start 2025-03-01`
- `lag_study_live.md`:云端 data 分支 archive.csv.gz(OKX 实时特征,62 个币,2026-09-09 ~ 2026-10-02,93 段大涨)。
  生成命令:`python lag_study.py --archive archive.csv.gz`
- `lag_rallies_*.csv` 每段大涨一行;`lag_rules_*.csv` 各规则在大涨各阶段的触发倾向。

结论摘要:第一次"读起来像在涨"的推送中位出现在启动后约 47 小时,此时涨幅已走完约一半;
只看规则触发(不看推送门槛)中位约 7 小时、走完约 11%,但早期触发多是按绝对值触发、方向不明的规则。
启动前 48 小时的推送只略高于随机窗口(33% vs 23%),多半是低点前的急跌引起的,不能当作提前预警。
