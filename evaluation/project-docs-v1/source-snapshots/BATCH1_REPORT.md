# PLM 第一批实施与实验报告

日期：2026-09-11。源码版本：`2.2.0a1`。范围：下一阶段计划中的首批评测、必要正确性修复、可选本地模型实验；不是 A–F 全部完成。

## 结论

第一批源码、隔离评测和真实本地模型对照已交付。81 项 unittest 通过，包含原有 100 并发 writer、安装到临时目录及回滚测试；skill 验证通过。全部破坏性测试在合成临时数据中执行。

已安装的全局 skill **没有替换**，仍是 2.1.0、engine=v2。按 PLM skill 收尾只追加了一条项目验收 Episode；最终只读检查：1,337 Events、3 Fact versions、41 projects，doctor healthy、SQLite integrity=ok。本轮不重提炼真实记忆、不迁移真实数据、不推送 GitHub。新功能已存在于源码和待验收安装包，不等于当前 Agent 自动使用了新功能。

## 已完成

- 证据检查从格式校验升级为项目内实际文件/工具输出 hash、Git 对象及指定版本内容检查。任意 source ID、格式正确的 hash、Agent 自报 confidence 不能成为高风险结论的证明。高风险要求与具体 Fact key/value 绑定的本地 receipt。
- 明确证据边界：receipt 匹配只能证明本地记录与结论一致，**不认证用户身份，也不直接验证远端发布、付款或权限状态**。可信宿主回执/外部状态核验仍需后续接入；不能把本地 receipt 写成“外部事实已认证”。
- 候选整体在 SQL 事务中预校验，写入持久批次标记后提交。失败不激活前缀；SQL 提交丢失时必须重建原批次，不可覆盖标记重新生成 IDs。缺标记/缺已提交源文件会明确失败，不假报健康。
- Fact identity 按项目、scope、scope_id、key 区分；supersede 保留 scope/条件。未来版本生效前，旧版本仍可作为当前事实。候选保留条件、观察时间、生效时间、来源和证据。
- 来源撤回沿派生链传播。当前纠正会使旧来源派生失效；未来替代到期后，检索动态过滤过期祖先。每一跳保持项目/scope 边界，循环或缺来源拒绝作为有效依据。
- 隐私清除写入无正文删除账本，清除 Event/Fact/FTS/向量/实体/候选、压缩 SQLite 并移除受管理的旧重建快照。未提交候选残留也纳入账本。尚无片段级依赖，因此候选暂作为保守清除单元：清除一条会影响同批摘要及兄弟 Facts，避免摘要残留其内容。
- `restore --archive --destination --deletion-ledger`：最新账本必填，只恢复到新建或空目录；拒绝路径逃逸、链接及特殊文件。先隔离解压/重建/重放删除/检查，再发布恢复副本；不修改原存储。无法验证操作者交来的账本是否绝对最新，旧 tar 与未知外部导出仍可能保留历史字节。
- 中文短词、连续问句的词法召回；可选真正的神经 embedding / cross-encoder 精排；来源链硬过滤先于模型；模型缺失有诊断，不冒充成功。核心仍零第三方依赖，模型只在独立环境中使用。
- 上下文整包预算、查询相关摘录、多证据公平配额、条件与否定优先、重复正文合并保留来源。预算采用 UTF-8 字节保守额度，不是假称精确 token。空查询优先 core/当前 Fact。JSON 结果返回 assertion source Event ID；原始结构化 JSON 不是完整文本预算包的替代物。
- 更新源码 skill 的候选提交与恢复规范；没有替换全局 skill。

## 固定实验

80 条合成来源、80 道人工编写中文题，dev/heldout 各 40 题、四个独立主题项目；每组 32 道有答案、8 道无答案。不是 300 道真实项目题，也不是 LongMemEval 官方问答评测。问题/gold 不进入记忆；保留题仅用于最终未调参对照，未向实现者展示逐题正文。

所有基线相同语料、Top-5、1800 保守字节预算、三遍检索。下表为 **heldout**；完整覆盖要求题目需要的全部来源进入上下文。

| 配置 | 证据 Recall | 完整证据覆盖 | 无答案误召回 | 热遍 P95 |
| --- | ---: | ---: | ---: | ---: |
| 无记忆 | 0% | 0% | 0% | 0.001 ms |
| 原始 Episode 字符检索 | 90.625% | 87.5% | 87.5% | 0.067 ms |
| PLM legacy 算法 | 57.8125% | 50% | 37.5% | 1.072 ms |
| 新 lexical | 90.625% | 87.5% | 87.5% | 1.598 ms |
| lexical + embedding | 100% | 100% | 100% | 47.018 ms |
| lexical + reranker | 90.625% | 87.5% | 87.5% | 43.289 ms |
| lexical + embedding + reranker | 100% | 100% | 100% | 95.694 ms |

开发集完整覆盖依次为 0%、96.875%、43.75%、100%、100%、100%、100%。开发集到留出集的回落不应被说成“精排导致退化”：**精排相对同组 lexical 没有带来净提升**。embedding 在留出集提高完整覆盖 12.5 个百分点，但无答案误召回从 87.5% 升到 100%。本轮不采用“模型越多越好”的默认配置。

无答案误召回指无答案问题仍返回任意背景来源，不是错误作答率；回答器可能正确拒答。因此不能直接据此宣布模型会错误回答，也不能忽视没有证据时的决策。**QA 正确率、官方榜单成绩、真实任务成功率均未测。** 无记忆的零误召回也不能掩盖它漏掉全部有答案问题。

时间覆盖指标在这份题集中是 Episode 叙事证据，不等价于完整双时态问答。候选正确性由独立单测覆盖。新上下文 pack 有独立单测；本实验统一使用评测层的完整来源 pack，因此不将实验增益归因于新摘录算法。

## 模型与成本

- [多语言 embedding 模型](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)：revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`，384 维。
- [多语言 cross-encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)：revision `1427fd652930e4ba29e8149678df786c240d8825`。
- 只下载 safetensors、分词器和配置，不下载远程 Python 或 pickle 权重。下载后推理离线；不上传真实记忆。PyPI 原站下载缓慢时，从清华镜像取相同 wheel，并逐一核对 PyPI 公布的 SHA256。
- Python 3.12.14 可选环境；CPU 两线程；模型数值持久缓存关闭。模型约 954 MiB，独立环境约 899 MiB，下载的 wheel 约 169 MiB，均在 git 忽略的 `.codex_work/`。没有启动常驻进程。
- 单独新进程探测：导入/工厂约 548 ms，首次 embedding 推理含加载约 3.71 s，首次 reranker 推理含加载约 0.94 s，进程峰值 RSS 约 1.06 GiB。操作系统页缓存可能已热，不称硬件冷启动；不是大规模语料内存上界。
- 表中 P95 包含查询分派、数据库打开、检索、模型调用、证据包装；不含回答模型、进程启动和建库。每组仅 40 条短来源，建库约 0.95 s；**不能外推为全部真实记忆或 10 万条语料性能**。同进程后续对照可复用已载入模型。

## 可复跑证据

- [评测协议](../evaluation/PROTOCOL.md)、[开发集报告](../evaluation/dev-final.json)、[留出集报告](../evaluation/heldout-final.json)、[首次模型加载](../evaluation/model-runtime.json)、[可选依赖锁定](../evaluation/model-requirements.lock)。
- 数据 SHA256：`c9a00d4758240db557faa1f87aa0e30a9e6db53d01dcc1a5d323af8f019795df`。
- 评测 Python 包指纹：`0d9406d3b729e648a3d6ea03f40108d08bd8ac6841a5eff5ca34cafb955fdb9a`；与安装包指纹算法不同，不混用。模型与配置指纹另存各报告。

```bash
python3 -m unittest discover -q
python3 scripts/plm_eval.py run --split dev --baselines none,raw,legacy,lexical
# 可选模型环境；首次下载由操作者显式运行，正常 context 不执行下载
python3.12 -m venv .codex_work/model-env
.codex_work/model-env/bin/python -m pip install -r evaluation/model-requirements.lock
python3 scripts/download_eval_models.py
.codex_work/model-env/bin/python scripts/plm_eval.py run --split dev \
  --baselines embedding,reranker,provider \
  --provider-factory evaluation.local_models:embedding \
  --reranker-factory evaluation.local_models:reranker
```

恢复必须选择全新目录：

```bash
python3 scripts/plm.py restore --archive /path/backup.tar.gz \
  --destination /path/new-memory-copy \
  --deletion-ledger /path/current-memory/v2/privacy/deletions.json
```

## 下一批按失分优先级

1. 固定回答模型与语义评分协议，测“证据不足时正确拒答／证据充分时答对”；接入官方 LongMemEval 实际运行。当前只有 adapter 和外部预测导入评分，没有官方成绩。
2. 将合成 80 题扩展为约 300 道独立标注、仓库外保存的自然中文项目问题；补长上下文、复杂干扰、罕见同义表述、时间更新与反事实前提。保留新的未参与调参题组。
3. 用开发集校准模型召回阈值与证据充分性，报告 precision/coverage/拒答的共同变化；不能拿全部拒答刷低错误率。本轮暂不启用精排为默认，因为没有独立增益。
4. 再推进原子候选/来源片段、多源去重、完整 valid_at/known_at、关系补检索和采用反馈。批次级清除后续细化为可核验的片段依赖。
5. 新版与当前 2.1.0 的真实查询影子对照、宿主可信回执、安装包一致性及回滚验收完成后才全局安装。本轮临时安装回滚成功不代表已完成真实切换；旧 verify 的标题自检不等于新的发布质量门槛。
