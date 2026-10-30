# tiger_memory：AML 开源组复现与提交说明

> 2026-10-12：本源码已增加 **2.2.0a5 / adapter 1.4.0** 开源参赛候选入口，
> 使用指定的 `text-embedding-v4`，操作方式与能力差异见
> [开源参赛准备与启动](OPEN_SOURCE_PREP.md)。候选已通过小规模真实 v4 联调与本地重启回读，
> 尚未部署到公网，未取得官方 Smoke/Full。本页下方的 BGE 模型、线上容量与耗时均为
> **历史 2.2.0a4 基线披露**，不能用来证明新候选效果或已符合开源榜资格。

本仓库提供公开源码、依赖、运行说明、来源和改动记录，用于复现与审核。
AML 参赛仍需要参赛方自行提供可访问、稳定的 Add/Search 在线服务。
源码交付与在线服务各自验证。当前自托管的 Health 地址为
`https://www.aigcpro.cn/aml/health`；Add 和 Search 分别为同域的 `/aml/add`、`/aml/search`，
需要参赛专用 Memory System Key。在线服务已通过本地与公网可达性检查，
尚未取得 AML 平台的 Smoke、Full 或公榜验收。

## 固定源码版本

- 项目：`tiger_memory`，原名称 Project Long Memory / PLM。
- 当前源码候选：`2.2.0a5`（preview）；AML 适配器：`1.4.0`。
- 本页历史部署基线：`2.2.0a4` / `1.3.4`，Commit `6ab0b870f2f79c458e4f69507b321ba2de4b8396`。
- 原始工作基线 Commit：`cc346f685764a769de3dfb2eb32ba78ae08dbcbc`。
- 复现时使用本次提交给评测方的 **40 位 Commit SHA**，不要用可移动的 `main` 作为版本标识。
  GitHub 的 Commit permalink 与 `git rev-parse HEAD` 可用于核对同一个版本。

```sh
git clone https://github.com/lwf225-source/tiger_memory.git
cd tiger_memory
git checkout --detach <提交给评测方的40位Commit-SHA>
git rev-parse HEAD
```

## 依赖与安装

核心、命令行和 HTTP 适配器使用 Python 标准库，要求 Python 3.9+，包括可用的 SQLite/FTS5。
PEP 517 构建依赖与核心运行依赖都为空，见 `pyproject.toml`。验证脚本另需 Git 和 Python `venv`。
这套基础环境无需 Codex、MCP、远程 API 凭证或模型下载。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --no-deps .
.venv/bin/tiger_memory --help
```

在 Windows 上把 `.venv/bin/` 换成 `.venv/Scripts/`。以下服务示例使用 POSIX shell。

### 可选 BGE 检索环境

完整披露的本地神经检索配置使用：

| 项目 | 固定值或位置 |
| --- | --- |
| 实测环境 | Python 3.12.14，macOS arm64 |
| Python 依赖 | [`repro/model-requirements.lock`](../repro/model-requirements.lock)，逐项固定版本 |
| 主要依赖 | sentence-transformers 5.1.2、transformers 4.57.6、torch 2.8.0 |
| 模型 | `BAAI/bge-small-en-v1.5` |
| 模型 revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| 模型文件校验 | [`repro/bge-small-en-v1.5.json`](../repro/bge-small-en-v1.5.json)，9 个文件的 SHA-256 |

实际参赛服务器为 Ubuntu 24.04、x86_64、Python 3.12.3。服务器安装了官方
`torch==2.8.0+cpu` wheel，其他 30 项依赖与
[`repro/linux-cpu-requirements.lock`](../repro/linux-cpu-requirements.lock)
一致；`pip check` 已通过。这个锁文件与 macOS 锁文件分开，避免 Linux 默认
PyTorch wheel 拉取 NVIDIA/CUDA 依赖。腾讯云服务器安装其余固定包时使用了
[腾讯云官方 PyPI 镜像](https://cloud.tencent.com/document/product/213/8623)；
任何兼容的 PyPI 镜像均应得到同一组声明的版本。

Linux x86_64 环境的安装顺序：

```sh
python3.12 -m venv .venv-model
.venv-model/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
.venv-model/bin/python -m pip install -r repro/linux-cpu-requirements.lock
.venv-model/bin/python -m pip install --no-index --no-deps .
.venv-model/bin/python -m pip check
```

该锁文件是实测环境记录，不代表所有操作系统与硬件已验证。模型文件由使用者另外下载；仓库不包含权重。

```sh
python3.12 -m venv .venv-model
.venv-model/bin/python -m pip install -r repro/model-requirements.lock
.venv-model/bin/python -m pip install --no-index --no-deps .
.venv-model/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

manifest = json.loads(Path('repro/bge-small-en-v1.5.json').read_text())
destination = Path('models/embedding-bge')
snapshot_download(repo_id=manifest['model_id'], revision=manifest['revision'],
                  allow_patterns=list(manifest['files']), local_dir=str(destination))
for name, expected in manifest['files'].items():
    actual = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Model file checksum mismatch: ' + name)
print('All model file checksums matched.')
PY
```

上面的依赖安装和模型下载需要网络。缓存完整后，Add/Search 的默认算法不调用外部生成模型。

## 启动 Add/Search

基础环境可以启动词法检索服务；需要 BGE 混合检索时先完成上面的模型准备，使用
`.venv-model/bin/python`，并设置 `PLM_AML_EMBEDDING_MODEL_DIR`。

```sh
export TIGER_MEMORY_DIR="$PWD/aml_data_store"
# BGE 模式使用下面的路径，并把启动命令中的解释器换成 .venv-model/bin/python：
# export PLM_AML_EMBEDDING_MODEL_DIR="$PWD/models/embedding-bge"
.venv/bin/python -m tiger_memory.aml_adapter --host 127.0.0.1 --port 8080
```

`TIGER_MEMORY_DIR` 应选择独立、持久化的服务目录。旧变量 `PROJECT_LONG_MEMORY_DIR`
仍作为兼容回退，新变量优先。算法开关是 `tiger_memory/aml_adapter.py` 中的常量；
不能通过同名 shell 环境变量任意切换。模型路径实际使用的兼容变量仍是
`PLM_AML_EMBEDDING_MODEL_DIR`。

缺模型或依赖时，适配器会降级为词法检索，部分失败路径没有单独告警。`GET /health` 成功
只说明 HTTP 服务运行，不证明 BGE 已就绪。需要核验模型 revision、文件哈希，
并检查启动模型状态 JSON 中的 `embedding.active` 为 `true`；`degraded_to: "lexical"` 表示已降级。

正式服务应配置 `AML_API_KEY`（由部署环境注入，不提交到 Git），并在部署层配置 HTTPS、
持久化存储和进程管理。客户端可使用 `Authorization: Bearer <token>`、
`Authorization: Token <token>` 或 `X-Api-Key`。未设置 `AML_API_KEY` 时不鉴权；
`/health` 始终可匿名访问。外部地址、TLS、认证方式与平台连通性另行验收。

当前自托管实例使用独立的 `tigermem` 系统账号和持久化目录，服务只监听
`127.0.0.1:18190`，由现有 Nginx 通过 HTTPS 转发。公网 Add/Search 共用并发上限 1，
服务进程限制为 850 MiB 内存和 1 个 CPU 核；超出入口并发上限返回 429 与
`Retry-After`。数值向量的内存热缓存最多保留 4,096 项，磁盘缓存仍供复用。
这些是当前部署参数，最终容量仍须由平台 Smoke 和正式数据验证。

## API 与本地验证示例

在另一个终端执行以下无认证的 localhost 示例。启用服务认证后，各 POST 请求需附带认证头。

```sh
curl --fail http://127.0.0.1:8080/health
curl --fail -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/add \
  -d '{"request_id":"example-1","user_id":"example-user","session_id":"example-session","messages":[{"role":"user","content":"My archive uses violet boxes.","timestamp":1750000000000}]}'
curl --fail -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/search \
  -d '{"query":"violet boxes","user_id":"example-user","top_k":5,"options":[]}'
```

- `POST /add`：`request_id`、`user_id`、`session_id` 是非空字符串；`messages` 是非空数组。
  每条消息包含非空 `role`、`content`；可选 `timestamp` 为 Unix 毫秒整数。
  成功响应包含 `success: true` 和原样回传的三个 ID。
  返回 200 前已写入并可检索；重试键为 `(user_id, request_id, message_index)`，同键以第一次写入为准。
- `POST /search`：`query`、`user_id` 是非空字符串，`top_k` 为必填非负整数，超过 100 时截到 100。
  `options` 可省略，提供时为字符串数组。响应格式为
  `{"data":[{"id":"...","content":"原始证据","score":0.1,"created_at":"..."}]}`。
  系统返回记忆证据，由平台完成后续回答；用户之间按 `user_id` 隔离。
- 空查询结果、`top_k=0` 或不存在的用户返回空 `data`。当前实现也会在 Search 内部异常时返回
  HTTP 200 和空 `data`，因此空结果不能证明相关记忆不存在，应结合服务检查判断故障。
- 请求体上限 32 MiB；JSON 格式错误为 400，契约错误为 422，认证失败为 401，Add 写入故障为 503。

## Add/Search 模型披露

| 阶段 | 默认配置 | 生成模型 |
| --- | --- | --- |
| Add | 原始消息持久化；`AML_FACT_PROJECTION="off"` | N/A |
| Search | 词法召回 + 可选本地 BGE 编码器；缺模型时降级词法 | N/A |
| 查询增强 | `AML_QUERY_REWRITE="recall-template"`，确定性本地模板 | N/A |
| 会话聚合 | session RRF，K=30 | N/A |

BGE 是向量编码器，不是生成回答的模型。代码包含可选远程 LLM 查询改写与 Fact 投影路径，
但上述默认配置未启用它们。若后续启用，须重新披露具体模型、凭证注入方式与配置，并固定新的 Commit。
本地 Reader/QA 评测工具与参赛 Add/Search 服务分开；其结果不视作 AML 官方成绩。

## 原始工作、来源与本次改动

原始工作是本仓库的 PLM 2.2.0a3 基线
`cc346f685764a769de3dfb2eb32ba78ae08dbcbc`。它已包含 Markdown Event、SQLite/FTS5、
Fact 生命周期、项目与用户隔离、AML 1.3.4 适配器、BGE 混合检索、离线查询模板及 session RRF。

本次相对基线的变化：

1. Python 包、命令和 Skill 名称改为 `tiger_memory`，增加新记忆目录变量与旧变量的兼容回退。
2. 补齐公开 README、测试、合成/历史项目文档 fixtures、运行和验证脚本、Skill 参考资料。
3. 修复 wheel/sdist 支持资源、已安装包的资源定位、隔离验收及 `verify` 的退出码。
4. 增补本说明、实测依赖锁、模型 revision/校验清单和公开上传的排除规则。
5. 将已部署的 Linux CPU 依赖锁及安装顺序作为可复现资料加入仓库；Add/Search 算法保持原样。
6. 为共享服务器的长期运行给数值向量热缓存设置 4,096 项上限；缓存淘汰后仍可从同一磁盘缓存读取，不改变检索证据的排序规则。

本次上传没有把 BGE、RRF 或已有检索算法列为新增算法贡献。
代码保留对 FlowGrid、ReFind、Chronicle 等思路的注释归因；外部库、模型与 CodeGraph
的来源见 [`THIRD_PARTY.md`](../THIRD_PARTY.md)，项目许可证见 [`LICENSE`](../LICENSE)。

```sh
git diff --stat cc346f685764a769de3dfb2eb32ba78ae08dbcbc HEAD
git diff --find-renames cc346f685764a769de3dfb2eb32ba78ae08dbcbc HEAD
```

## 验证命令与证据范围

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_install.py
.venv/bin/python scripts/verify_release.py --output output/release-verification
```

测试使用本地临时目录、合成数据和 localhost HTTP，不需要真实记忆、付费模型或外部评测数据。
发布验证涵盖离线 wheel/sdist 安装、重建字节一致性、真实验收退出码及临时 Skill 安装。
它不检查公网服务可用性，不代表平台接入或官方评测通过。

自托管实例另通过 10 个隔离合成用户的本地 HTTPS Add/Search 检查：每人一次写入
20 条消息，并以 `top_k=100` 检索；这组请求的 Add 中位耗时约 0.71 秒、Search
中位耗时约 1.99 秒，没有接口错误，服务内存峰值约 492 MiB。该小样本不能推算正式
Full 评测成绩或吞吐量。

已安装的个人 Codex Skill 可能与公开仓库具有不同能力；复现和参评应使用这里固定 Commit 的源码，
不依赖任何个人全局 Skill。真实记忆、模型权重、私有运行报告、环境文件与本机工作区不随仓库提供。
