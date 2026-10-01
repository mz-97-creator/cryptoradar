# CryptoRadar:币圈仓位结构异动监控

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

## 五点五、调参与评分模型的滚动检验

```
.venv\Scripts\python.exe tune.py                 用库里所有已回填的币(先 backfill.py --universe)
.venv\Scripts\python.exe tune.py --folds 4 --top-pct 5 --min-n 30
```

把全部历史切成几段,每一轮"用前面的历史选阈值/训练模型,到紧接着的、没见过的时间段检验"(中间留 72 小时空档防止泄漏),
对比四种触发方式:基准、当前规则、网格调参后的规则、逻辑回归评分模型。
结果在 `reports/tune_report.md`。只有样本外的数字才有参考价值;调参规则如果不如当前规则,说明过拟合,不要采用。

## 六、离线自检

```
.venv\Scripts\python.exe -m tests.selftest
```
用合成数据跑通采集 → 特征 → 规则 → 推送 → 回填 → 事件研究全流程,不联网。
合成数据里埋了一个已知规律,自检会验证事件研究能把它找出来,且触发时点都在事件之前(没有前视偏差)。

## 七、文件结构

```
monitor.py            实时监控入口
backfill.py           历史回填
research.py           事件研究
config.example.yaml   配置模板
cryptoradar/
  binance_api.py      币安公开接口(含限速)
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
