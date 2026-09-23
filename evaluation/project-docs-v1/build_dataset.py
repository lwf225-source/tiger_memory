"""One-shot construction of a frozen, manually annotated PLM-document QA pilot.

This script is an annotation artifact, not a template generator. Do not use its
heldout annotations for retrieval, prompting, source extraction or tuning.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from tiger_memory.evaluation import load_dataset, write_dataset


ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
OBSERVED_AT = "2026-09-11T00:00:00Z"

# One contiguous, verbatim document range per source. Line numbers are one-based.
SOURCE_SPECS = [
    ("pdv1-s01", "plm-storage", "dev", "docs/ARCHITECTURE.md", 5, 7, "Architecture — trust boundary"),
    ("pdv1-s02", "plm-storage", "dev", "docs/ARCHITECTURE.md", 11, 20, "Architecture — storage layout"),
    ("pdv1-s03", "plm-storage", "dev", "docs/ARCHITECTURE.md", 22, 25, "Architecture — source-based rebuilding"),
    ("pdv1-s04", "plm-storage", "dev", "docs/ARCHITECTURE.md", 45, 47, "Architecture — immutable identity"),
    ("pdv1-s05", "plm-evidence", "dev", "docs/BATCH1_REPORT.md", 13, 14, "第一批报告 — 证据核验及其边界"),
    ("pdv1-s06", "plm-evidence", "dev", "docs/BATCH1_REPORT.md", 15, 15, "第一批报告 — 候选批次提交"),
    ("pdv1-s07", "plm-evidence", "dev", "docs/BATCH1_REPORT.md", 16, 16, "第一批报告 — Fact 身份与有效时间"),
    ("pdv1-s08", "plm-evidence", "dev", "docs/BATCH1_REPORT.md", 17, 17, "第一批报告 — 来源与派生链"),
    ("pdv1-s09", "plm-evidence", "dev", "docs/BATCH1_REPORT.md", 18, 18, "第一批报告 — 隐私清除粒度"),
    ("pdv1-s10", "plm-retrieval", "dev", "docs/ARCHITECTURE.md", 29, 36, "Architecture — lexical retrieval and provider boundary"),
    ("pdv1-s11", "plm-retrieval", "dev", "docs/BATCH1_REPORT.md", 20, 20, "第一批报告 — 本地模型与词法回退"),
    ("pdv1-s12", "plm-retrieval", "dev", "docs/BATCH1_REPORT.md", 21, 21, "第一批报告 — 上下文证据包装"),
    ("pdv1-s13", "plm-ops", "heldout", "docs/RUNBOOK.md", 35, 36, "Runbook — invocation audit"),
    ("pdv1-s14", "plm-ops", "heldout", "docs/RUNBOOK.md", 47, 52, "Runbook — rebuilding and moved backup paths"),
    ("pdv1-s15", "plm-ops", "heldout", "docs/RUNBOOK.md", 60, 62, "Runbook — rollback export"),
    ("pdv1-s16", "plm-ops", "heldout", "docs/RUNBOOK.md", 66, 72, "Runbook — install and cutover sequence"),
    ("pdv1-s17", "plm-ops", "heldout", "docs/BATCH1_REPORT.md", 19, 19, "第一批报告 — 受控恢复边界"),
    ("pdv1-s18", "plm-eval", "heldout", "docs/BATCH1_REPORT.md", 26, 28, "第一批报告 — 实验数据及口径"),
    ("pdv1-s19", "plm-eval", "heldout", "docs/BATCH1_REPORT.md", 30, 40, "第一批报告 — 配置对照及取舍"),
    ("pdv1-s20", "plm-eval", "heldout", "docs/BATCH1_REPORT.md", 42, 44, "第一批报告 — 指标解释限制"),
    ("pdv1-s21", "plm-eval", "heldout", "docs/BATCH1_REPORT.md", 48, 53, "第一批报告 — 模型实验资源与计时边界"),
    ("pdv1-s22", "plm-eval", "heldout", "docs/BATCH1_REPORT.md", 84, 88, "第一批报告 — 后续优先级与发布条件"),
]


def point(identity, description, *alternatives):
    return {"id": identity, "description": description, "any_of": list(alternatives)}


def forbid(identity, description, *patterns):
    return {"id": identity, "description": description, "patterns": list(patterns)}


QUESTIONS = [
    ("d01", "plm-storage", "principle", "这些记忆真正依赖哪些源文件保存？SQLite 在其中承担什么角色？", "记忆的源记录是不可变 Markdown Event；SQLite 保存目录和当前事实、检索等派生视图。", ["s02"],
     [point("source", "不可变 Markdown Event 是源记录", "Markdown", "Markdown Event", "Markdown 事件"), point("derived", "SQLite 保存当前事实和检索视图", "派生", "当前事实", "current facts")],
     [forbid("sqlite_only", "不能说 SQLite 是唯一记忆源", "SQLite.{0,8}(唯一|仅有).{0,5}(源|记忆)")]),
    ("d02", "plm-storage", "recovery_reason", "索引数据库意外丢失后，重建时还要准备哪些原始材料？", "需要项目 manifests、Markdown Events、candidate 文件和迁移 checksum manifest。", ["s03"],
     [point("manifests", "项目身份 manifests", "manifest", "项目清单"), point("events", "Markdown Events", "Markdown", "事件文件"), point("candidates", "候选文件", "candidate", "候选"), point("checksums", "迁移校验清单", "checksum", "校验", "迁移清单")],
     [forbid("lost_forever", "不能宣称删数据库等于记忆永久丢失", "(记忆|记录).{0,5}(全部|永久).{0,5}(丢失|无法恢复)")]),
    ("d03", "plm-storage", "location", "Agent 刚提交、还等着处理的候选内容，会以什么文件形式保存？", "保存在 candidates/*.json 持久候选队列中。", ["s02"],
     [point("queue", "candidates JSON 文件", "candidates/*.json", "candidates", "候选 JSON")],
     [forbid("volatile", "候选不是只留在进程内存", "仅.{0,5}(进程内存|内存中)")]),
    ("d04", "plm-storage", "identity", "项目改了名字或迁了目录，存储布局里哪个文件负责保留身份和路径别名？", "项目目录下的 manifest.json 保存稳定身份和路径 aliases。", ["s02"],
     [point("file", "manifest.json", "manifest.json"), point("alias", "保存路径别名", "alias", "别名")],
     [forbid("new_id", "不能把每次移动都说成新项目", "(必须|一定).{0,8}(新建|生成新).{0,5}项目")]),
    ("d05", "plm-storage", "immutability", "同一个 Event ID 重试时，如果正文已经变了，系统应该接受还是拒绝？", "拒绝；只有内容完全相同的重试才被接受。", ["s04"],
     [point("reject", "内容变化时拒绝", "拒绝", "不接受", "不能覆盖"), point("identical", "相同内容可以幂等重试", "相同", "一致", "identical")],
     [forbid("overwrite", "不能允许覆盖同 ID 正文", "(允许|可以|直接).{0,4}覆盖")]),
    ("d06", "plm-storage", "scope_boundary", "知道另一个项目的记忆 ID，能否从当前工作目录直接修改它？", "不能；修改命令从 cwd 确定当前项目，不能操作另一个项目的 ID。", ["s04"],
     [point("deny", "拒绝跨项目修改", "不能", "不可以", "拒绝"), point("cwd", "以 cwd 确定项目", "cwd", "当前工作目录")],
     [forbid("id_bypass", "知道 ID 不能绕过项目边界", "知道.{0,6}ID.{0,6}就能")]),
    ("d07", "plm-storage", "authority", "记忆里写的结论与真实项目证据冲突时，应该以什么为准？", "以项目文件、Git 提交、测试和部署证据为最终依据；记忆只提供有来源的工作上下文。", ["s01"],
     [point("authority", "实际项目证据优先", "项目文件", "Git", "测试", "部署证据"), point("role", "记忆作为有来源上下文", "上下文", "context", "辅助")],
     [forbid("memory_wins", "不能让记忆自动覆盖事实证据", "记忆.{0,6}(最高权威|优先于真实|覆盖实际)")]),
    ("d08", "plm-storage", "multi_source", "重建丢失的目录索引时，能顺手把旧 Event 改写后沿用原 ID 吗？为什么？", "不能。重建应从原始记录恢复目录；原 Event ID 不能承载不同内容。", ["s03", "s04"],
     [point("rebuild", "从源记录恢复", "原始记录", "源记录", "Markdown"), point("immutable", "同 ID 内容不可更改", "不能", "拒绝", "不可改写")],
     [forbid("rewrite_ok", "恢复不允许复用旧 ID 改内容", "(可以|允许).{0,12}(改写|覆盖)")]),
    ("d09", "plm-storage", "unanswerable", "这套存储服务按每千条记忆收多少钱？", "资料没有提供按条计费价格，无法据此确定。", [],
     [point("unknown", "价格未提供", "没有提供", "未说明", "无法确定", "不知道")],
     [forbid("invent_price", "不得编造价格", "每千条.{0,8}\\d+.{0,2}元")]),
    ("d10", "plm-evidence", "evidence_requirement", "提交一个格式正确的 hash，或把 confidence 写得很高，能直接证明高风险状态吗？", "不能；需要实际核验，并有与具体 Fact key/value 绑定的本地 receipt。", ["s05"],
     [point("insufficient", "hash 格式和自报信心不足", "不能", "不够", "不足"), point("receipt", "需要绑定具体 Fact 的回执", "receipt", "回执"), point("binding", "绑定具体 key/value", "key/value", "具体 Fact", "键和值", "具体结论")],
     [forbid("confidence_proof", "自报信心不是高风险证明", "confidence.{0,12}(就能证明|即可通过)")]),
    ("d11", "plm-evidence", "trust_limit", "本地 receipt 校验通过，是否等于用户身份和远端付款都已被系统认证？", "不等于；receipt 只能证明本地记录与结论一致，不认证身份，也不直接核验远端付款等状态。", ["s05"],
     [point("local", "仅证实本地记录一致", "本地", "记录一致"), point("not_auth", "身份和外部状态仍未认证", "不认证", "不等于", "不能证明", "不直接")],
     [forbid("remote_verified", "不能夸大为已认证外部事实", "(用户身份|远端付款).{0,5}已被认证")]),
    ("d12", "plm-evidence", "identity", "两个不同 session 都有同名 Fact key，它们应当算同一条事实吗？", "不应合并；Fact identity 同时由项目、scope、scope_id 和 key 区分。", ["s07"],
     [point("scope", "scope 参与身份", "scope"), point("scope_id", "scope_id 参与身份", "scope_id", "session ID"), point("separate", "不同作用域不合并", "不应", "不同", "区分", "不合并")],
     [forbid("key_only", "不能只凭 key 合并", "只.{0,4}(看|按).{0,4}key.{0,6}(合并|相同)")]),
    ("d13", "plm-evidence", "temporal_condition", "新 Fact 的生效日还没到，原来的值会立即从当前上下文消失吗？", "不会；未来版本生效之前，原版本仍可作为当前事实。", ["s07"],
     [point("not_early", "旧值不提前失效", "不会", "仍可", "仍然", "继续有效"), point("until", "等未来版本生效", "生效前", "生效之前", "生效日", "未来版本生效")],
     [forbid("early_drop", "不能提前挤掉旧值", "(立即|提前).{0,6}(删除|失效|消失)")]),
    ("d14", "plm-evidence", "atomicity", "一个候选包含多条 Fact，后面的校验失败时，前面已处理的部分能先算激活成功吗？", "不能；整个候选先在事务内预校验，失败不激活前缀，故障恢复必须重建原批次。", ["s06"],
     [point("atomic", "不能激活成功前缀", "不能", "不激活前缀", "整体"), point("transaction", "事务预校验", "事务", "SQL"), point("batch", "恢复原批次", "原批次", "原 IDs", "不能重新生成")],
     [forbid("partial", "不能称前缀已正式激活", "前.{0,5}(仍然|可以|先).{0,4}激活")]),
    ("d15", "plm-evidence", "multi_source", "若旧 Fact 要到未来才被替代，依赖它的说明在前后两个阶段应怎样处理？", "替代生效前旧 Fact 仍有效；到期后检索应沿来源链过滤依赖过期祖先的派生说明。", ["s07", "s08"],
     [point("before", "生效前旧 Fact 有效", "生效前", "生效之前", "到期前"), point("after", "到期后排除派生说明", "过滤", "排除", "失效"), point("provenance", "沿来源祖先处理", "来源链", "祖先", "派生")],
     [forbid("forever", "派生说明不能无限沿用旧值", "派生.{0,10}永久有效")]),
    ("d16", "plm-evidence", "privacy_condition", "为什么清除一个候选里的某条内容，可能连同同批摘要和兄弟 Facts 一起受影响？", "目前没有片段级依赖，候选暂作为保守清除单元，以避免摘要残留被清除的内容。", ["s09"],
     [point("granularity", "尚无片段级依赖", "没有片段级", "尚无片段级", "未实现片段级"), point("unit", "按候选或批次保守清除", "候选", "同批", "批次"), point("purpose", "防止残留", "残留", "泄漏")],
     [forbid("precise_fragment", "不能声称已能精确片段清除", "已经.{0,4}(精确|片段级).{0,6}清除")]),
    ("d17", "plm-evidence", "multi_source", "候选里既有生效条件又有高风险结论，正式写入时需要同时保住哪两类信息？", "保留 scope、条件、观察与生效时间等限定，同时保留来源和与具体 Fact 绑定的可核验 receipt 证据。", ["s05", "s07"],
     [point("conditions", "限定和时间保留", "条件", "限定", "生效时间"), point("source", "保留来源", "来源", "source"), point("receipt", "高风险证据绑定具体 Fact", "receipt", "回执")],
     [forbid("discard_conditions", "不能为了短摘要舍弃条件", "(可以|应当).{0,6}(丢弃|忽略|省略)条件")]),
    ("d18", "plm-evidence", "unanswerable", "系统用哪一种数字签名算法认证提交回执的真实用户？", "现有材料没有提供用户身份认证算法；本地 receipt 不承担身份认证。", [],
     [point("unknown", "不提供认证算法", "没有提供", "未提供", "不承担", "不认证")],
     [forbid("invent_algorithm", "不能猜测认证签名算法", "(采用|使用|基于).{0,6}(RSA|Ed25519|ECDSA).{0,8}(认证|签名)")]),
    ("d19", "plm-retrieval", "dependency_boundary", "要运行基本记忆检索，必须先装神经网络模型环境吗？", "不必；核心保持零第三方依赖，模型是独立环境中的可选能力。", ["s11"],
     [point("optional", "模型不是核心必需", "不必", "不需要", "可选"), point("core", "核心零第三方依赖", "零第三方依赖", "零依赖", "标准库"), point("separate", "模型在独立环境", "独立环境", "可选环境")],
     [forbid("mandatory", "不能把模型当作默认强依赖", "必须.{0,6}(安装|下载).{0,6}模型")]),
    ("d20", "plm-retrieval", "fallback", "本地模型没有准备好时，检索能否假装语义模型成功运行？", "不能；模型缺失必须有明确诊断或降级说明，不冒充语义运行成功。", ["s11"],
     [point("diagnosis", "明确记录模型缺失", "诊断", "降级", "缺失"), point("no_false_success", "不假报成功", "不能", "不冒充", "不得", "不假报")],
     [forbid("silent_success", "缺模型不能仍报模型成功", "(静默|直接).{0,8}(报告|标记).{0,4}成功")]),
    ("d21", "plm-retrieval", "budget", "上下文预算显示 1800，这是否就是某个回答模型会计费的精确 Token 数？", "不是；这里采用 UTF-8 字节的保守额度，不是模型 tokenizer 的精确 Token 数。", ["s12"],
     [point("bytes", "UTF-8 字节计量", "UTF-8", "字节"), point("estimate", "保守而非精确 token", "保守", "不是精确", "非精确")],
     [forbid("billing_exact", "不能当成精确计费 Token", "就是.{0,5}精确.{0,5}Token", "等于.{0,6}计费")]),
    ("d22", "plm-retrieval", "context_selection", "一条记忆特别长、另一条含有关键限制时，包装上下文怎样避免后者被吞掉？", "采用多证据公平配额和查询相关摘录，优先保留条件与否定信息。", ["s12"],
     [point("fair", "多证据公平配额", "公平配额", "公平分配", "多证据"), point("excerpt", "查询相关摘录", "摘录", "相关片段"), point("conditions", "优先限定和否定", "条件", "否定", "限制")],
     [forbid("first_only", "不能永远只保留第一条长记忆", "只保留第一条")]),
    ("d23", "plm-retrieval", "multi_source", "原来 512 维本地向量与第一批增加的 embedding 是同一种能力吗？增加模型会迫使源记录换格式吗？", "不同：原向量是字符 n-gram 的词法向量，新 embedding 是可选神经语义能力；provider 边界允许保留原来的源记录。", ["s10", "s11"],
     [point("lexical", "512 维是词法向量", "词法", "lexical", "字符 n-gram"), point("neural", "新能力是神经 embedding", "神经", "语义"), point("source", "源记录无需替换", "保留", "不改变", "无需", "不用改")],
     [forbid("old_neural", "不能把旧哈希向量说成神经向量", "原来.{0,12}就是神经")]),
    ("d24", "plm-retrieval", "unanswerable", "检索模块规定每条记忆最多永久保留多少天？", "给定资料没有规定统一的最大保留天数。", [],
     [point("unknown", "没有统一保留天数", "没有规定", "未说明", "无法确定", "不知道")],
     [forbid("invent_retention", "不能编造统一保留天数", "(保留|保存).{0,5}\\d+天")]),
    ("h01", "plm-ops", "recovery_reason", "把传统备份放到另一个记忆根目录后，为什么不能直接拿旧索引做健康检查？", "目录中的来源路径是绝对路径，需要先从恢复的 Markdown Events 重建索引，再运行 doctor。", ["s14"],
     [point("absolute", "旧索引含绝对路径", "绝对路径", "absolute"), point("rebuild", "先 rebuild-index", "rebuild-index", "先重建", "重建索引"), point("doctor", "之后 doctor", "doctor", "健康检查")],
     [forbid("direct_health", "不能跳过重建直接认定恢复健康", "无需重建.{0,8}(健康|doctor)")]),
    ("h02", "plm-ops", "rollout_condition", "执行安装命令成功后，新安装包首先进入哪个模式？这一步是否已经完成正式切换？", "首先进入 shadow 模式，安装本身不是正式 cutover。", ["s16"],
     [point("shadow", "初始为 shadow", "shadow", "影子"), point("not_cutover", "安装不等于正式切换", "不是", "不等于", "尚未", "还没有")],
     [forbid("immediate_v2", "不能把安装后状态说成立即正式 v2", "安装.{0,6}(立即|自动).{0,6}正式")]),
    ("h03", "plm-ops", "release_gate", "什么样的验收报告才可以用于正式切换，切换后还要确认哪些运行状态？", "verify 执行随包测试并绑定代码指纹及 Event 数；所有门禁通过后 cutover，再确认 engine=v2、write_mode=v2 并运行 doctor。", ["s16"],
     [point("binding", "报告绑定代码及事件数", "代码指纹", "code fingerprint", "Event 数"), point("all_gates", "所有门禁通过", "所有", "全部", "every"), point("runtime", "确认读写 v2", "engine=v2", "write_mode=v2", "读写"), point("doctor", "运行 doctor", "doctor")],
     [forbid("some_gates", "部分门禁通过不足切换", "部分.{0,5}通过.{0,5}即可")]),
    ("h04", "plm-ops", "rollback", "回滚前新增的 v2 数据会如何处理，什么时候才需要 materialize-v1 选项？", "先把切换后新增 Event 导出为 v1 兼容 Markdown；只有要立即放进旧项目目录时才加 --materialize-v1。", ["s15"],
     [point("export", "先导出新增数据", "导出", "export"), point("compat", "v1 兼容 Markdown", "v1", "Markdown"), point("immediate", "选项用于立即进入旧目录", "立即", "旧项目目录", "legacy")],
     [forbid("discard", "回滚不能直接丢掉新增事件", "直接.{0,5}(丢弃|删除).{0,8}新增")]),
    ("h05", "plm-ops", "privacy_recovery_limit", "恢复工具要求提供删除账本，是否就能证明操作者拿来的账本绝对最新、旧导出也都已清干净？", "不能；账本虽必填，但无法验证它绝对最新，旧 tar 和未知外部导出仍可能保留历史字节。", ["s17"],
     [point("ledger", "账本必填但新鲜度不能保证", "必填", "必须", "无法验证", "不能保证"), point("copies", "旧归档和外部导出可能残留", "旧 tar", "外部导出", "历史字节", "残留")],
     [forbid("all_erased", "不能承诺所有外部副本都已清除", "所有.{0,6}(外部副本|旧备份).{0,6}(都已|全部).{0,4}清除")]),
    ("h06", "plm-ops", "restore_boundary", "恢复副本的目标目录和归档成员有什么限制，会直接覆盖原存储吗？", "只恢复到新建或空目录，拒绝路径逃逸、链接和特殊文件；隔离验证后发布副本，不修改原存储。", ["s17"],
     [point("empty", "目标必须新建或为空", "新建", "空目录"), point("unsafe_members", "拒绝链接等不安全成员", "链接", "路径逃逸", "特殊文件"), point("original", "不修改原存储", "不修改", "不覆盖", "原存储保持")],
     [forbid("overwrite_original", "不能直接覆盖原记忆目录", "直接覆盖原")]),
    ("h07", "plm-ops", "observability", "usage 能看到哪些调用信息，它会保留原始提问和返回的记忆正文吗？", "可以看调用次数、结果数、成功和延迟；只存查询 SHA-256 hash，不存提问原文或记忆正文。", ["s13"],
     [point("counts", "次数与结果数量", "次数", "结果数"), point("latency", "成功和耗时信息", "延迟", "耗时", "成功"), point("hash", "仅查询 hash", "SHA-256", "hash", "哈希"), point("no_plaintext", "不保存原文", "不存", "不保存", "never")],
     [forbid("full_query_log", "不能把审计说成完整查询日志", "(保存|记录).{0,4}完整提问")]),
    ("h08", "plm-ops", "unanswerable", "生产记忆库每天自动备份的定时时刻是几点几分？", "现有文档没有给出每天自动备份的具体定时。", [],
     [point("unknown", "没有定时信息", "没有给出", "未说明", "无法确定", "不知道")],
     [forbid("invent_time", "不能猜测自动备份时刻", "每天.{0,5}\\d{1,2}[:：]\\d{2}")]),
    ("h09", "plm-eval", "measurement_boundary", "第一批能报告证据召回表现，是否也已经测出了问答正确率和官方榜单成绩？", "没有；QA 正确率、官方榜单成绩和真实任务成功率都未测。", ["s20"],
     [point("unmeasured", "明确未测 QA 和榜单", "未测", "没有", "尚未"), point("qa", "问答正确率", "QA", "问答"), point("official", "官方成绩", "榜单", "官方")],
     [forbid("official_score", "不能将证据召回100%冒充官方问答成绩", "官方.{0,6}(正确率|成绩).{0,4}100%")]),
    ("h10", "plm-eval", "tradeoff", "在留出对照中，增加 embedding 带来了什么收益，又伴随什么无答案问题上的代价？", "完整证据覆盖提高 12.5 个百分点，但无答案误召回从 87.5% 升到 100%。", ["s19"],
     [point("coverage_gain", "覆盖增益12.5个百分点", "12.5", "十二点五"), point("false_before", "此前87.5%", "87.5"), point("false_after", "之后100%", "100%", "百分之百")],
     [forbid("all_better", "不能称所有指标同步改善", "所有指标.{0,4}(改善|提升)")]),
    ("h11", "plm-eval", "configuration_choice", "第一批有没有证据支持把精排默认打开？开发集到留出集的变化应怎样解释？", "没有独立净增益，暂不默认启用精排；开发集到留出集回落不等于精排导致退化，应比较同组 lexical。", ["s19"],
     [point("no_gain", "同组精排无净提升", "没有净", "无净", "没有独立", "未带来净"), point("not_default", "不默认启用", "不默认", "暂不", "不采用"), point("comparison", "同组比较", "同组", "lexical", "不能归因")],
     [forbid("reranker_caused", "不能把跨数据集回落直接归因精排", "(?<!不等于)(?<!不是)(?<!说成)精排导致退化")]),
    ("h12", "plm-eval", "latency_scope", "报告里的热遍 P95 覆盖了哪些工作，为什么不能据此估算十万条真实记忆的端到端性能？", "计入分派、数据库打开、检索、模型和证据包装；不含回答模型、启动或建库，且每组只有40条短来源，不能外推大规模。", ["s21"],
     [point("included", "涵盖检索和包装", "检索", "证据包装", "数据库打开"), point("excluded", "不含回答器或启动建库", "不含", "不包括"), point("scale", "仅40条短来源", "40", "四十", "短来源"), point("no_extrapolate", "不能外推", "不能外推", "无法外推", "不代表十万")],
     [forbid("all_e2e", "不能说包含回答模型的全部端到端耗时", "包含回答模型.{0,6}全部")]),
    ("h13", "plm-eval", "multi_source", "这次已完成的数据集与后续约300题目标有什么区别？", "已完成的是80条合成来源和80道人工作题；约300道独立自然中文项目题是后续目标，尚不能说已完成。", ["s18", "s22"],
     [point("existing", "已有80道合成题", "80", "八十"), point("synthetic", "明确合成性质", "合成"), point("target", "300是后续目标", "300", "三百"), point("future", "目标未全部完成", "后续", "目标", "尚未", "待")],
     [forbid("already_300", "不能说已经完成300道真实题", "已(经)?完成.{0,4}300.{0,4}真实")]),
    ("h14", "plm-eval", "runtime_limit", "首次模型探测是不是硬件冷启动测试？实验运行时是否有常驻服务？", "不是严格硬件冷启动，操作系统页缓存可能已热；没有启动常驻进程。", ["s21"],
     [point("cache", "页缓存可能已热", "页缓存", "缓存可能", "不称硬件冷启动"), point("no_daemon", "没有常驻进程", "没有启动常驻", "无常驻", "不是常驻")],
     [forbid("hardware_cold", "不能称严格硬件冷启动", "(这是|属于|完成了).{0,4}严格.{0,4}硬件冷启动")]),
    ("h15", "plm-eval", "multi_source", "要把检索实验推进为可比较的问答能力，下一步需要增加什么固定环节？", "需要固定回答模型与语义评分协议，测充分证据下答对和不足证据下正确拒答，并实际运行官方 LongMemEval。", ["s20", "s22"],
     [point("reader", "固定回答模型", "固定回答模型", "固定 reader", "固定读者"), point("scoring", "固定评分协议", "评分协议", "语义评分", "评审协议"), point("abstention", "同时测答对和拒答", "拒答", "依据不足"), point("official_run", "实际运行官方评测", "LongMemEval", "官方实际运行", "运行官方")],
     [forbid("retrieval_equals_qa", "不能直接将召回率当问答正确率", "召回率就是问答正确率")]),
    ("h16", "plm-eval", "unanswerable", "LongMemEval 当前官方榜单第一名的准确率是多少？", "给定项目资料没有提供当前官方榜单第一名的得分。", [],
     [point("unknown", "榜首得分未提供", "没有提供", "未提供", "无法确定", "不知道")],
     [forbid("invent_rank", "不得编造榜首分数", "第一名.{0,8}\\d+(\\.\\d+)?%")]),
]


def build():
    if (FOLDER / "manifest.json").exists():
        raise SystemExit("Frozen dataset exists; construct a new version instead.")
    snapshots = FOLDER / "source-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    documents = {}
    for relative in sorted({spec[3] for spec in SOURCE_SPECS}):
        original = ROOT / relative
        snapshot = snapshots / original.name
        if snapshot.exists():
            raise SystemExit("Snapshot already exists; inspect it before constructing a dataset.")
        shutil.copyfile(original, snapshot)
        documents[relative] = {
            "original_path": relative,
            "snapshot_path": str(snapshot.relative_to(FOLDER)),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "line_count": len(snapshot.read_text(encoding="utf-8").splitlines()),
        }
    sources = []
    provenance = []
    for source_id, project, split, original_path, start, end, title in SOURCE_SPECS:
        document = documents[original_path]
        lines = (FOLDER / document["snapshot_path"]).read_text(encoding="utf-8").splitlines()
        body = "\n".join(lines[start - 1:end]).strip()
        sources.append({"source_id": source_id, "project": project, "split": split, "title": title,
                        "body": body, "observed_at": OBSERVED_AT, "scope": "project", "scope_id": "", "tags": []})
        provenance.append({"source_id": source_id, "original_path": original_path,
                           "snapshot_path": document["snapshot_path"], "document_sha256": document["sha256"],
                           "line_start": start, "line_end": end,
                           "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                           "extraction": "contiguous original lines joined with LF, outer whitespace stripped"})
    questions = []
    for qid, project, category, question, answer, gold, required, forbidden in QUESTIONS:
        split = "dev" if qid.startswith("d") else "heldout"
        questions.append({"question_id": "pdv1-" + qid, "project": project, "split": split,
                          "question": question, "answer": answer, "acceptable_answers": [answer],
                          "category": category, "required_source_ids": ["pdv1-" + identity for identity in gold],
                          "unanswerable": not gold, "scope": "project", "scope_id": "", "view": "current",
                          "required_points": required, "forbidden_claims": forbidden})
    write_dataset(FOLDER, sources, questions, {
        "dataset_id": "plm-project-docs-zh-v1-40", "origin": "manually_annotated_questions_from_one_real_project_document_snapshot",
        "splits": ["dev", "heldout"], "freeze_date": "2026-09-11",
        "source_count": len(sources), "question_count": len(questions),
        "split_policy": "One PLM project, five non-overlapping topic groups; dev=storage/evidence/retrieval; heldout=operations/evaluation. Topic keys are not independent real projects.",
        "official_benchmark_score": False, "real_user_query_sample": False, "real_project_count": 1,
        "annotation_policy": "Human-style manual annotation by a separate agent; no model tuning on heldout questions. Not an external secret test.",
        "snapshot_semantics": "Answers describe the frozen 2026-09-11 documentation, not the future installed runtime.",
    })
    provenance_payload = {"schema_version": 1, "dataset_id": "plm-project-docs-zh-v1-40", "observed_at": OBSERVED_AT,
                          "documents": list(documents.values()), "sources": provenance}
    (FOLDER / "provenance.json").write_text(json.dumps(provenance_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
    sidecar = {"schema_version": 1, "provenance_sha256": hashlib.sha256((FOLDER / "provenance.json").read_bytes()).hexdigest(),
               "dataset_manifest_sha256": hashlib.sha256((FOLDER / "manifest.json").read_bytes()).hexdigest(),
               "scope": "Document provenance sidecar is not ingested as memory; main manifest remains loader-compatible."}
    (FOLDER / "provenance-manifest.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for split in manifest["splits"]:
        loaded = load_dataset(FOLDER, split)
        print(json.dumps({"split": split, "sources": len(loaded["sources"]), "questions": len(loaded["questions"]),
                          "unanswerable": sum(item["unanswerable"] for item in loaded["questions"]),
                          "multi_source": sum(len(item["required_source_ids"]) > 1 for item in loaded["questions"])}, ensure_ascii=False))


if __name__ == "__main__":
    build()
