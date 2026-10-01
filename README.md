# CryptoRadar

## 云端版(当前在用)

- **扫描**:`cloud_run.py` 在 GitHub Actions 上运行,用 OKX 公开数据扫描市值前 150 名中有 OKX USDT 永续的代币,外加自选(OP、HYPE、ETH、AAVE、ARB、PUMP、SOL、XRP、FIL、STX)。
  - GitHub 自带的定时触发实际要隔 4–6 小时才跑一次,所以 Claude 定时任务每次推送前会向 `tick` 分支推一次 `tick.txt`,约 3 分钟后拿到新数据。
- **推送**:Claude 定时任务每 1.5 小时一条(UTC 每 3 小时的 :07 和 :37 交替),每天 8:46(新加坡时间)一份日报。
- **推送内容**:
  1. **市场红绿灯**:全市场资金费率分位 × 持仓变化 × 广度(站上 7 日均线的币比例),分为 🔴 去杠杆风险高 / 🟠 杠杆过热 / 🟡 市场转弱 / 🔵 杠杆已出清 / 🟢 正常,附历史上该状态之后 72h 的表现和"何时解除/升级"
  2. **历史概率**:每条预警都附"全市场(或本币)历史上同类情形出现 N 次,之后 72h 上涨概率 vs 基准、中位收益、最差 10%、持有期最大回撤";样本少于 15 次或偏离基准不到 8 个百分点时不给方向
  3. **预警记分卡**:每条预警记录当时的历史概率,24h/72h 后用真实价格核对,统计命中率和"预计 vs 实际"
- **data 分支文件**:`status.md`(网页上直接看)· `signals.json` 当前快照 · `events.json` 最近 7 天事件 · `archive.csv.gz` 全市场小时特征样本库(最近 180 天,历史概率的样本会随时间增长)· `predictions.json` 记分卡 · `state.json`
- 调整阈值、自选、价位/费率提醒、回购数据:修改 `cloud_config.yaml` 并提交,会自动重新扫描
- 局限:历史概率目前只有约 1 个月的样本,而且同一时间很多币一起涨跌,样本并不完全独立,显著性会被高估;记分卡是检验它们是否在样本外仍然成立的唯一办法

离线自检:`python -m tests.cloud_selftest`

---

## 本地版(可选)

监控市值前 200 名、且有币安 USDT 永续合约的代币(另加自选,默认 OP),每 15 分钟扫描一次。
出现 OI 背离、资金费率拥挤、脱离大盘的独立行情等异动时推送到微信;同时把全部小时级数据
存入本地数据库,为以后的"多头机会评分模型"积累样本。

只调用币安和 CoinGecko 的**公开行情接口**,不需要 API Key,也不会下单。

---

## 一、快速开始(Windows)

1. 安装 Python 3.11 或以上版本(python.org,安装时勾选 **Add Python to PATH**)
2. 解压本文件夹,双击 **`setup_windows.bat`**(创建虚拟环境、安装依赖、生成 `config.yaml`)
3. 用记事本打开 `config.yaml`,填写 `pushplus_token`(见第二节)
4. 双击 **`test_push.bat`**,微信收到测试消息即绑定成功
5. 双击 **`start_monitor.bat`** 开始监控。首次运行要补 30 天数据,约需 5–10 分钟

想先不推送、只在屏幕上看结果:`.venv\Scripts\python.exe monitor.py --once --dry-run`

## 二、微信推送绑定(PushPlus)

1. 打开 https://www.pushplus.plus ,用微信扫码登录,并关注"pushplus 推送加"公众号
2. **完成实名认证**(未认证账号不能发送消息)
3. 在"一对一推送"页面复制 token,填到 `config.yaml`:
   ```yaml
   notify:
     channel: pushplus
     pushplus_token: "你的token"
   ```

**额度**:实名认证后微信渠道每天 200 次、每分钟 5 次。超过 200 次当天停发,超过 2000 次会被限制 7 天。
程序默认每天最多推送 150 条(`daily_limit`),一轮扫描的所有异动合并成一条消息,同一代币同一规则 6 小时内不重复推送。

**备选渠道**:如果实名认证办不了,
- Server酱(sct.ftqq.com):免费版每天只有 5 条,不够用,付费版每天 1000 条。填 `serverchan_sendkey`,`channel: serverchan`
- Telegram 机器人:填 `telegram_bot_token` 和 `telegram_chat_id`,`channel: telegram`

## 三、24 小时运行:电脑关机就会停

程序跑在哪台机器上,就依赖哪台机器开着。三种方式:

| 方式 | 做法 | 适合 |
|---|---|---|
| Windows 常开 | 双击 `install_autostart.bat`(登录后自动启动);设置 → 系统 → 电源 → 睡眠改为"从不" | 先试用 |
| Mac mini | 同样的代码直接能跑:`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python monitor.py` | 已有常开设备 |
| 海外 VPS | 约 5 美元/月的 Linux 服务器,**避开美国机房**(币安对受限地区返回 HTTP 451)。用 systemd 常驻,见文末 | 长期积累数据,最推荐 |

**为什么连续运行很重要**:币安实时接口的 OI 和多空比只保留最近 30 天。关机超过 30 天造成的空档,
只能靠 `backfill.py` 从官方数据站补,而且官方数据站通常要到第二天才会发布前一天的文件。
关机几天问题不大,程序重启后会自动补齐。

## 四、推送内容和规则

每条推送包含:大盘状态(BTC、ETH/BTC)、价位提醒,以及每个异动代币的规则说明和关键数据。

| 规则 | 条件 | 含义 |
|---|---|---|
| OI 激增但价格未涨 | OI 24h 变化 z ≥ 2,残差收益 z ≤ 0.5 | 合约在建仓但现货没跟,方向未定(本次 OP 的情形) |
| 放量上涨且 OI 增加 | OI z ≥ 2,残差收益 z ≥ 1 | 新资金顺势加仓 |
| OI 骤降伴随下跌 | OI z ≤ −2,24h 收益 < 0 | 多头去杠杆/被清算 |
| OI 骤降伴随上涨 | OI z ≤ −2,24h 收益 > 0 | 空头回补 |
| 资金费率偏高/偏负 | 费率 z ≥ 2.5 或 ≥ 0.1%/8h(负向同理) | 多头/空头拥挤 |
| 脱离大盘的独立行情 | 残差收益 z ≥ 2.5 | 剔除 BTC beta 后仍显著 |
| 1 小时急涨/急跌 | 1h 收益 z ≥ 3.5 | 短时冲击 |
| 成交额异常放大 | 24h 成交额 z ≥ 2.5 | 关注度上升 |
| 大户多空比极值 | z ≥ 2.5 | 噪音较大,仅作辅助 |

- 所有 z-score 都是和**该代币自己过去 30 天**比较,只用历史数据,不含当前值
- **残差收益** = 代币收益 − β × BTC 收益,β 由过去 30 天小时收益滚动估计
- 普通代币要多条规则同时触发(权重合计 ≥ 2.5)才推送;自选代币门槛是 1
- 每天 9 点发一条日报:自选代币状态 + 独立行情最强的 5 个代币,同时说明程序还在运行

阈值都在 `config.yaml` 的 `signals.thresholds` 里,可以调整。

## 五、历史回填与事件研究(评分模型的第一步)

```
.venv\Scripts\python.exe backfill.py              回填 OP/BTC/ETH,从 2022-06-01 开始(约 15–30 分钟)
.venv\Scripts\python.exe research.py              OP 单币事件研究
.venv\Scripts\python.exe backfill.py --universe   回填整个监控名单(耗时数小时,可中断后续跑)
.venv\Scripts\python.exe research.py --pool       合并所有代币,样本量大得多
```

OI、多空比、主动买卖比来自币安官方数据站 data.binance.vision 的每日文件(5 分钟粒度)。
这样不必从零开始"逐渐积累",马上就有 4 年多的历史样本。

`research.py` 输出每条规则(以及任意两条规则同时成立)之后的表现,结果保存在 `reports/`:

| 列 | 含义 |
|---|---|
| n | 事件数(连续触发只算一次,两次事件至少间隔 72 小时) |
| ret24_mean / ret72_mean | 未来 24h/72h 平均收益 |
| resid72_mean / median | 未来 72h 剔除 BTC 影响后的收益 |
| hit72 | 72h 残差收益为正的比例 |
| t72 | t 值,|t| < 2.5 基本视为噪音(一共检验了几十个条件) |
| mae72_mean / p10 | 72h 内最大不利波动的均值 / 最差的 10% |
| safe_lev | 让 90% 的事件不被强平的杠杆上限近似值(未计手续费和维持保证金) |
| early / late | 前 70% 与后 30% 时间段分别的结果,方向相反就不可信 |

第一行"基准"是任意时点开仓的表现,每个条件都应该和它比。

## 五点五、调参、权重学习与评分模型的样本外检验

```
pip install -r requirements-research.txt                      梯度提升和报告表格需要(云端扫描不需要)
.venv\Scripts\python.exe backfill_vision.py --bases-file syms.txt --start 2024-01-01   币安 API 被限制地区用这个回填
.venv\Scripts\python.exe tune.py                 用库里所有已回填的币
.venv\Scripts\python.exe tune.py --min-safe-lev 3 --pcts 2,5,10 --min-n 30
```

`backfill_vision.py` 的 K 线、资金费率、OI 全部读 data.binance.vision,不访问 fapi.binance.com(部分地区会返回 451);
当月最后几天的资金费率数据站还没有,那段时间资金费率相关规则不触发。

`tune.py` 比较六种触发方式(都只看做多:触发后 72 小时的 BTC 残差收益):基准、当前规则、
**调参规则**(网格搜索阈值)、**学习权重**(规则权重由数据学出,系数为负的规则权重记 0 = 砍掉)、
**逻辑回归**、**梯度提升**。

- **选参统一规则**:训练段内部前 70% 拟合、后 30% 验证,在验证段按"事件数 ≥ min_n、平均残差收益 > 0、
  `safe_lev` ≥ `--min-safe-lev`(默认 3 倍)的前提下 t 值最高"选参数;都不满足就不触发,不勉强给结果。
- **两种检验**:滚动检验(前 40% 起步,后面 4 段逐段检验)和固定的前 70% 训练 / 后 30% 检验。训练与检验之间隔 72 小时防止泄漏。
- 报告 `reports/tune_report.md` 含:汇总、自动判断(事件数够、t ≥ 2.5、高于基准和当前规则、safe_lev 达标才算"值得采用")、
  每轮学出的权重、逻辑回归系数、梯度提升特征重要性。
- 结论是"值得采用"时,把 `reports/suggested_weights.yaml` 的内容并入 `cloud_config.yaml` 的 `signals:` 下
  (`rule_weights` 覆盖权重,权重 ≤ 0 即停用该规则;不配置则用内置权重)。

只有样本外的数字才有参考价值;调参规则如果不如当前规则,说明过拟合,不要采用。

## 五点五五、72 小时机会模型(model.py)

```
.venv\Scripts\python.exe model.py                  评估 + 当前排名(先 backfill.py / backfill_vision.py 回填)
.venv\Scripts\python.exe model.py --threshold 0.05 --topk 8 --cost 0.002
```

对名单里每个币预测未来 72 小时(相对 BTC 超额):上涨概率 P(>+5%)、下跌概率 P(<-5%)、波动幅度、
回撤 q10 及由此得到的"90% 情形不被强平"的杠杆上限。输入是币自身特征 + 大盘行情(BTC 趋势/波动、市场广度、全市场资金费率)。
评估按时间截面做(每 72 小时取一次,互不重叠):Spearman IC、前/后 k 名的真实表现、粗略扣成本后的超额(只是个量级判断,不是含对冲腿/成交时点/资金曲线的可交易回测)、概率校准与 Brier 技能分、
回撤分位数覆盖率、按大盘风格拆分;并与"只看最近波动"和"现行规则得分"两个基线对照。
结果在 `reports/model_report.md`,当前排名在 `reports/model_latest.csv`。

### 云端的机会榜

云端每次扫描会用 `models/opportunity_price.joblib` 给每个币打分,写进 `signals.json` 的 `opportunity`(`status.md` 里也有一节):
预测 72h 波动幅度、回撤 q10 与杠杆上限、上涨/下跌概率、各项排名,以及自选币里排名靠前的"值得留意"项(`watch_highlights`)。
上涨/下跌概率的做法:先预测这个币未来 72h 剔除 BTC 后的波动尺度 σ,再用"收益/σ"的样本外历史分布推出概率,
所以波动越大两头概率越高、榜单顺序与波动榜相同——这不是缺陷,而是现有特征里确实没有方向信息的直接体现;
方向标签(direction)在样本外 t 值不足时一律输出"无明确方向",研究中的档位保留在 `direction_tier_research`。
云端数据来自 OKX,而模型用币安历史训练,所以云端用**价格模型**(只用价格与成交额派生的特征;实测两个交易所的这类特征相关性 0.95~1.00,
而持仓量、资金费率、多空比、主动买卖比差别很大,所以不用)。模型包出错或没装 scikit-learn 时只会跳过这一块,不影响主扫描。
重新训练并导出(需先回填到最新):

```
.venv\Scripts\python.exe model.py --feature-set price --export models/opportunity_price.joblib
```
导出的模型必须和云端的 scikit-learn 版本一致(`radar.yml` 里固定为 1.9.1)。
每 6 小时会把全部币的预测记入 `data` 分支的 `opp_log.csv.gz`,满 72 小时后用真实价格核对,
结果写进 `signals.json` 的 `opportunity_live`(预测概率 vs 实际频率、回撤越界率、波动排序相关性),用来检验模型在实盘是否仍然成立。

## 五点六、实盘信号后验表

云端每次扫描会把推送过的信号追加到 `data` 分支的 `ledger.csv`(永久累积):触发的全部规则、得分、
24h/72h 真实收益(原始和相对 BTC)、持有期最大回撤。`status.md` 里有"实盘信号后验表(按规则)",
列出每条规则的到期数、72h 超额中位、上涨比例、t 值和 safe_lev,样本少于 30 会标注。
实盘样本攒够后,可以和 `research.py` / `tune.py` 的回测结果对照,看规则在实盘有没有失效。

## 六、离线自检

```
.venv\Scripts\python.exe -m tests.selftest
```
用合成数据跑通采集 → 特征 → 规则 → 推送 → 回填 → 事件研究全流程,不联网。
合成数据里埋了一个已知规律,自检会验证事件研究能把它找出来,且触发时点都在事件之前(没有前视偏差)。

## 七、文件结构

```
monitor.py            实时监控入口
backfill.py           历史回填(币安 API + 数据站)
backfill_vision.py    历史回填(只用数据站,地区受限时用)
tune.py               阈值/权重调参与评分模型的样本外检验
model.py              72 小时机会模型(波动/上涨/下跌概率/回撤)的评估、当前排名与导出
horizon_study.py      方向研究:1周/2周/4周下的相对强弱排序(现有特征 + 经典跨币因子 + DefiLlama),Newey-West 校正的 t 值
research.py           事件研究
config.example.yaml   配置模板
cryptoradar/
  binance_api.py      币安公开接口(含限速)
  opportunity.py      72h 机会模型的特征、模型、校准与云端打分(训练评估和云端共用)
  defillama.py        DefiLlama 免费接口:公链/协议 TVL 与费用(解锁/排放数据是付费接口,不用)
  universe.py         市值前 N ∩ 币安永续 的名单映射
  collector.py        增量采集
  features.py         特征计算(监控与研究共用,避免回测和实盘两套代码)
  signals.py          规则定义
  notify.py           微信 / Telegram 推送
  monitor.py          主循环
  storage.py          SQLite 存储(data/cryptoradar.db)
logs/                 运行日志
reports/              事件研究结果
```

## 八、局限

- 信号描述的是**仓位结构异动,不代表方向**。OI 激增可能是多头建仓,也可能是空头建仓
- 规则阈值是经验值,在 `research.py` 验证之前,不要把推送当作开仓信号
- 只覆盖币安永续合约的数据;社交热度、链上交易所流入流出暂未接入(需要付费数据源,计划放在第二阶段)
- 回测里 4 小时结算的合约,资金费率绝对阈值按 8 小时口径会有偏差(z-score 不受影响)

## 附:Linux VPS 常驻(systemd)

```
sudo apt install -y python3-venv
cd ~/CryptoRadar && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml   # 填 token
curl -s -o /dev/null -w "%{http_code}\n" https://fapi.binance.com/fapi/v1/ping   # 200 才能用
```
`/etc/systemd/system/cryptoradar.service`:
```
[Unit]
Description=CryptoRadar
After=network-online.target

[Service]
WorkingDirectory=/root/CryptoRadar
ExecStart=/root/CryptoRadar/.venv/bin/python monitor.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```
然后执行 `sudo systemctl enable --now cryptoradar`,查看日志用 `journalctl -u cryptoradar -f`。
