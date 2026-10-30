# Tiger memory 开源参赛准备记录

本页记录 2026-10-12 准备的候选版本：核心 `2.2.0a5`、AML adapter `1.4.0`，
独立入口为 `python -m tiger_memory.aml_open_source`。它是开源文本赛道候选，
已完成下文记录的小规模真实 Embedding API 联调，尚未部署到公网，未取得官方 Smoke / Full 结果。
旧版 BGE 服务的可达性或旧评测不能替代此版本的证据。

## 资格与模型边界

[赛事官网 FAQ](https://agentmemoryleaderboard.ai/competition/) 当前将相关榜单称为“学术榜”，
要求 Embedding 使用 `text-embedding-v4`，LLM 相关组件使用 `gpt-4o-mini`。
本项目按该公开规则准备开源方法榜配置，并在报名材料中如实披露模型与供应方式。
如果实际报名表或当期规则出现不同限制，再针对具体冲突向主办方核实；不把额外确认作为默认前置门槛。
本候选的 Add/Search 不调用生成式 LLM，因此不增加 `gpt-4o-mini` 调用或虚构配对评测门槛；
最终 Answer / Eval 由平台承担。

[阿里云通用文本向量同步接口文档](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
列明 v4 支持 1024 维、每批最多 10 条、每条最多 8192 Token。
这里的 **8192 UTF-8 字节切块**是本候选采用的保守输入策略，与供应商的 Token 上限不是同一单位。

## 已实现的候选配置

| 项目 | 配置与行为 |
| --- | --- |
| Embedding | `text-embedding-v4`，1024 维，OpenAI-compatible 同步接口 |
| 单批及长文本 | 每批最多 10 个分块；按 Unicode 边界切成最多 8192 UTF-8 字节；分块向量等权均值后 L2 归一化，不截掉尾部 |
| 调用边界 | 每个请求的 provider 最多 100 批、每批最多 30 秒 socket 超时、累计 1200 秒批次边界预算，无自动重试 |
| 数值缓存 | 按 `user_id` 独立目录；Add 预热文档向量，Search 复用相同文档向量并编码查询；新查询和缓存未命中仍可能付费调用 |
| 检索 | 词法与向量融合，保留会话 RRF（K=30） |
| 关闭项 | 动态 recency 文本、query rewrite、生成式 Fact projection、reranker |
| 故障 | 候选的向量/检索失败返回 503，不静默降级为词法或成功的空结果；正常无匹配仍可返回空 `data` |
| 数据目录 | 专用新目录，与日常 Tiger memory 库隔离；配置标记不匹配的非空目录拒绝启动 |
| 网络 | 明确启用远程访问；固定官方 HTTPS Embeddings URL；禁用自动重定向和环境/系统代理 |
| 状态 | `/status` 必须鉴权，包含模型配置、配置 SHA-256 及成功/失败计数；不包含 API Key 或记忆正文 |

`/health` 表示 HTTP 服务存活。配置哈希、provider 的 ready 状态或本地模拟测试
都不能证明远程模型实际可调用、检索质量合格或赛事平台已通过验收。
Add 的向量预热失败可能发生在部分原文已持久化之后；503 后应使用同一 request ID 重试，
利用既有幂等键恢复，不把失败请求当作已成功完成。
累计时间使用单调时钟在每批前后检查，单批 socket 超时会收紧到剩余预算。
它不能强制中断 DNS 或缓慢滴流中的 socket，也不覆盖前置数据库写入，
因此不是整个 HTTP 请求的严格 20 分钟上限；部署层仍须设置请求总超时。

原入口 `python -m tiger_memory.aml_adapter` 保留原有词法/BGE 兼容路径。
日常 Skill、日常记忆库和原线上实例不因新增候选入口自动切换。

## 静态检查与配置检查

静态 preflight 只检查源码材料，不访问网络，不查看凭证是否存在，不把 Docker 或生成式 LLM
当作参赛必需条件，也不会把本地 Git origin 的存在当作仓库已公开的证明。

```sh
python3 scripts/aml_preflight.py
```

部署环境需通过安全的配置注入方式提供以下变量。不要把真实密钥写入仓库、终端共享日志或报告：

| 变量 | 要求 |
| --- | --- |
| `TIGER_MEMORY_DIR` | 独立、绝对、规范路径；首次启动选择空目录或不存在的目录；避免日常 `~/.codex/project-memory` |
| `TIGER_MEMORY_EMBEDDING_ENDPOINT` | 对应账户区域的官方完整 HTTPS `/compatible-mode/v1/embeddings` URL，不是仅有主机名或 base URL |
| `DASHSCOPE_API_KEY` | 对应区域和业务空间的可用 Embedding 凭证 |
| `AML_API_KEY` | 此参赛服务独立的访问密钥 |
| `TIGER_MEMORY_ALLOW_REMOTE` | 显式设为 `1` |

只检查配置可运行：

```sh
python3 -m tiger_memory.aml_open_source --check-config
```

此命令不创建数据目录，不调用 API，输出脱敏的配置、哈希及问题列表。
返回 0 仅表示配置字段通过静态检查，返回 2 表示配置不完整或无效；它不会证明凭证权限、余额、
实际目录可写性、远程可用性或官方验收成功。

在授权调用、区域、额度和专用目录确定后，启动候选：

```sh
python3 -m tiger_memory.aml_open_source --host 127.0.0.1 --port 18191
```

启动会初始化专用目录。Add/Search 的缓存未命中会向指定 Embedding 服务发送参赛输入并可能计费。
部署时另行配置 HTTPS、进程管理、持久化目录和合理的并发/资源限制。
Docker 可作为打包选择，当前没有把 Docker 构建成功列为此候选的必经证据。

## 申请接口与费用

2026-10-12 核对官方资料：北京区 `text-embedding-v4` 同步调用为每千输入 Token
0.0005 元（每百万 0.5 元）。该模型新人免费额度为 100 万 Token，通常有效期 90 天；
是否仍有额度，以账户控制台为准。北京以外地域的价格、额度不同，不能混用地域密钥。

在 [百炼北京区 API Key 页面](https://bailian.console.aliyun.com/cn-beijing/model/settings/api-key)
登录并开通，创建通用 API Key。当前候选只需要 Embedding API，无需购买 Coding Plan 或另建知识库。
到免费额度页面查找 `text-embedding-v4`，有可用额度时可开启“免费额度用完即停”；
等待配置生效后再测试。已实名认证账户未开启该开关时，额度耗尽会转为按量计费。
初次联调可先使用免费额度，正式评测前另外核算用量及服务持续可用性。

来源：[接口与单价](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)、
[申请 API Key](https://help.aliyun.com/zh/model-studio/get-api-key)、
[免费额度及用完即停规则](https://help.aliyun.com/zh/model-studio/new-free-quota)。

## 有限规模的参赛方联调

`scripts/aml_smoke.py` 默认仅输出计划，不读取密钥、不访问网络：

```sh
python3 scripts/aml_smoke.py
```

在模型调用已获授权、服务已配置、费用边界已确定后，通过环境注入 `AML_API_KEY`，运行：

```sh
python3 scripts/aml_smoke.py --execute --base-url http://127.0.0.1:18191 \
  --output /absolute/private/report-new.json
```

公网地址必须为 HTTPS，可包含 `/aml` 前缀。脚本禁止自动重定向和环境代理，
只接受字面量 loopback 地址使用 HTTP。报告不记录密钥、URL、模型错误正文或检索正文。
最多 12 个 HTTP 请求，不自动重试，遇首个失败即停；检查服务存活、鉴权、候选配置、
两个用户的 Add/Search 隔离、未知用户空结果和同 ID 重放后原文不变。
报告记录每步状态与耗时、配置哈希及随机合成用户前缀，合成数据会留在该服务的专用库中。

本地 stub 已验证：当前冻结配置与全新合成用户下，该计划触发 4 批 Embedding 请求。
这是代码路径预期，不是供应商计费凭证，也不是可执行的金额硬上限。
每次运行创建新用户，因此重复执行会再次消耗额度；真实费用须在百炼控制台核对。
单请求使用 30 秒 socket 超时，没有整体硬截止时间。
该脚本不测服务重启后持久化、全量质量或容量，且**不等同官方 Smoke**。

## 真实接口联调记录

2026-10-12 21:18（Asia/Shanghai），在源码 Commit
`5c6a2f3b5a393eb6cea0331904789ee107f3c589` 上完成本地 HTTP 服务到百炼北京区
`text-embedding-v4` 的真实调用。12 项客户端检查全部通过，实际 4 批 Embedding 请求，
无自动重试。每批请求 JSON 为 117–158 字节；这些字节数不等于计费 Token 数。
模型返回通过 1024 维、数值有效性及响应索引校验。

- Add 两次分别约 2942 ms、334 ms；首次查询分别约 340 ms、637 ms。
  这些是仅两个合成用户的单次 HTTP 耗时，不代表吞吐、质量或延迟分位数。
- 重复 Add 保留首次原文，重复查询命中缓存；不同用户分别返回自己的原文，未知用户结果为空。
- 随后在新 Python 进程中重启本地 HTTP 服务，两个用户均回读到原文；
  验证时禁止新的 Embedding HTTP 调用，实际新增调用为 0，证明本次数据及缓存落盘可复用。
- 配置 SHA-256：`fbe3880a8ffdfdb107d743a100678830557e1b4afc93489df93bf39b98766d37`。
- 本机 Python 缺少默认 CA 文件，通过 `SSL_CERT_FILE=/etc/ssl/cert.pem` 使用已有 macOS CA 包，
  TLS 证书校验保持开启。其他环境应使用自身有效的信任链。
- 供应商实际账单和免费额度扣减尚未核对，不能把 4 次调用写成已知金额或断言免费。

两个本地服务进程均已停止。该结果不改变旧公网实例，不构成官方 Smoke / Full 成绩。

## 本轮已经取得的 HTTP 证据

在 macOS、Python 3.13.15 上，原 8 worker Add 回归连续运行 30 次有 19 次连接失败。
根因为标准库服务器默认 pending TCP backlog=5，不能容纳突发连接。
纯 HTTP 对照实验暂停 accept 并预建 16 个 `/health` 请求：backlog=5 时只有 5 个成功，
backlog=128 时 16 个成功。修复在服务器调用 listen 前设置 backlog=128。

修复当时 `tests.test_aml_adapter` 的 20 项测试全部通过；增强后的 8 worker Add 回归与
16 个 pending HTTP 连接回归各重复 30 次，60 次全部通过。把 backlog 临时恢复为 5 的负对照
立即复现 ConnectionResetError。这是连接队列修复证据，不是远程 v4 模型或任意负载容量证据。

后续集成检验应在最终候选 Commit 上重新运行，不沿用中途测试次数作为最终验收：

```sh
python3 -m unittest tests.test_aml_preflight tests.test_remote_embeddings tests.test_aml_open_source tests.test_aml_adapter -v
python3 -m unittest discover -s tests -t .
python3 scripts/aml_preflight.py
git diff --check
git rev-parse HEAD
```

## 仍缺少的参赛证据

2026-10-12 本地集成验证：macOS / Python 3.13.15，496 项回归通过；
隔离、离线 sdist 安装验证 9/9 通过，包括安装包内的开源入口和配置资料。
496 项回归中的向量接口测试使用 localhost stub；后续真实接口联调另见上节。
上述结果支持本地候选交付，不代表真实模型质量或平台评测成绩。

1. 提交时复核当期规则，并完整披露开源方法榜、文本赛道、无生成式 LLM 的 v4 配置及供应方式。
2. 公开的最终源码 Commit、许可证、依赖与该 Commit 对应的离线回归记录。
3. 小规模真实 v4 调用、Add/Search 回读、隔离与重启缓存已通过；仍需代表性检索质量、规模化延迟及供应商费用记录。
4. 部署候选后的服务版本、配置哈希、鉴权、TLS、持久化、限流及失败恢复记录。
5. 官方 Smoke 与 Full 的 Run / Job ID、结果和必要的复核记录。

只有这些实际证据才能将状态从“本地候选准备”推进到“正式有效提交”。
