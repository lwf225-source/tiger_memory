"""Independent, manually specified scoring points for the frozen synthetic pilot.

Only additive rubric sidecars are written. Original manifests, questions and
sources are never changed, and none of this file is reader input.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


DATASET = Path(__file__).resolve().parent.parent / "synthetic-v1"


# Each inner list describes alternative surface forms of one indispensable point.
DEV_POINTS = {
    "dev_library-q01": [["七本", "7本", "7 本"]],
    "dev_library-q02": [["关闭自动续借", "暂停自动续借", "停用自动续借"], ["导出借阅", "导出清单", "导出借书"], ["核对后", "核对完成", "对照书架完成", "确认后"], ["再开启", "恢复续借", "重新开启", "再打开"]],
    "dev_library-q03": [["七本", "7本", "7 本"], ["两位志愿者", "两名志愿者", "2位志愿者", "2名志愿者"], ["影响借阅", "借阅窗口", "无人接待", "借阅服务"]],
    "dev_library-q04": [["橙色三角", "橙色的三角", "橙色三角形"]],
    "dev_library-q05": [["蓝色圆点", "蓝色的圆点", "蓝色圆形"]],
    "dev_library-q06": [["不可以", "不能直接", "不得直接", "不能自动"], ["馆员", "图书管理员"], ["逐本", "逐一", "单独判断", "一本一本"]],
    "dev_library-q07": [["两位志愿者", "两名志愿者", "2位志愿者", "2名志愿者"], ["借阅窗口", "无人接待", "影响借阅", "借阅服务"]],
    "dev_library-q09": [["四秒", "4秒", "4 秒"]],
    "dev_orchard-q01": [["低于百分之二十四", "低于24%", "小于24%", "24%以下", "不足24%"]],
    "dev_orchard-q02": [["末端堵塞", "检查堵塞", "检查管道"], ["东侧阀", "东阀"], ["十二分钟", "12分钟", "12 分钟"], ["关阀", "关闭阀门"], ["登记流量", "记录流量", "流量读数"]],
    "dev_orchard-q03": [["十五厘米", "15厘米", "15 cm", "15cm"], ["根系", "根部"], ["波动", "不稳定"]],
    "dev_orchard-q04": [["六点", "6点", "06:00", "6:00"]],
    "dev_orchard-q05": [["十五点", "15点", "15:00", "下午三点", "下午3点"]],
    "dev_orchard-q06": [["不会", "不自动", "暂停自动", "停止自动"], ["人工检查", "人工核查", "人工判断", "人工确认"]],
    "dev_orchard-q07": [["波动", "不稳定"], ["十五厘米", "15厘米", "15cm", "15 cm"], ["根系", "根部"]],
    "dev_orchard-q09": [["八分钟", "8分钟", "8 分钟"]],
    "dev_museum-q01": [["前六件", "前6件", "前 6 件", "最前面的六件"]],
    "dev_museum-q02": [["淡出旧", "淡出上一", "旧讲解淡出"], ["识别新编号", "识别新展品", "新编号识别"], ["标题卡片", "标题卡", "展品标题"], ["确认后", "点击确认", "观众确认"]],
    "dev_museum-q03": [["编号识别后", "识别编号后", "识别新编号后", "识别后"], ["标题卡片", "标题卡"], ["确认", "点一下"], ["反射", "识别跳变", "识别不稳定", "来回跳"]],
    "dev_museum-q04": [["二十四像素", "24像素", "24 像素", "24px", "24 px"]],
    "dev_museum-q05": [["十八像素", "18像素", "18 像素", "18px", "18 px"]],
    "dev_museum-q06": [["不可以", "不得", "不能", "禁止"], ["人脸", "面部照片"]],
    "dev_museum-q07": [["玻璃", "展柜"], ["反射"], ["跳变", "来回跳", "识别不稳定", "识别切换"]],
    "dev_museum-q09": [["九十秒", "90秒", "90 秒", "一分半", "1分30秒"]],
    "dev_workshop-q01": [["四十五分钟", "45分钟", "45 分钟"], ["十分钟", "10分钟", "10 分钟"]],
    "dev_workshop-q02": [["开工前", "开始前", "切割前"], ["开抽风", "开启抽风", "启动抽风"], ["两分钟", "2分钟", "2 分钟"], ["关闭", "停抽风"]],
    "dev_workshop-q03": [["十分钟", "10分钟", "10 分钟"], ["透镜"], ["边角料", "残料"], ["焦痕", "烧焦"]],
    "dev_workshop-q04": [["十八点三十分", "18:30", "18点30分", "下午六点半", "晚上六点半"]],
    "dev_workshop-q05": [["十九点", "19:00", "19点", "晚上七点", "晚上7点"]],
    "dev_workshop-q06": [["暂停", "停机", "停止设备"], ["不能交", "不得交", "不允许", "不能让", "不可以"], ["未培训", "没培训", "未经培训"]],
    "dev_workshop-q07": [["残料", "边角料"], ["焦痕", "烧焦"]],
    "dev_workshop-q09": [["三十分钟", "30分钟", "30 分钟"]],
}


DEV_FORBIDDEN = {
    "dev_library-q04": ["(?<!不是)(?<!而非)蓝色圆点"],
    "dev_library-q05": ["(?<!不是)(?<!而非)橙色三角"],
    "dev_library-q06": ["(?<!不)可以直接.{0,5}(撤下|移除|下架)", "自动.{0,3}(撤下|下架)"],
    "dev_orchard-q01": ["高于.{0,3}(24%|百分之二十四)"],
    "dev_orchard-q06": ["(?<!不)会自动开阀", "继续自动滴灌"],
    "dev_museum-q06": ["(?<!不)可以.{0,4}(采集|拍摄).{0,3}人脸"],
    "dev_workshop-q02": ["结束后立即关闭"],
    "dev_workshop-q06": ["(?<!不)可以.{0,8}(朋友|访客).{0,8}(代看|继续)"],
}


HELDOUT_POINTS = {
    "hold_observatory-q01": [["超过三点二像素", "超过3.2像素", "大于3.2像素", "3.2像素以上"]],
    "hold_observatory-q02": [["曝光前", "拍目标前"], ["十张暗场", "10张暗场", "10 张暗场"], ["曝光后", "曝光完成后", "拍完后"], ["平场"]],
    "hold_observatory-q03": [["超过三点二", "超过3.2", "大于3.2", "3.2像素以上"], ["跟踪抖动", "追踪抖动"]],
    "hold_observatory-q04": [["观测日期", "观察日期", "观测当天日期"]],
    "hold_observatory-q05": [["目标名称", "天体名称", "观测目标名"]],
    "hold_observatory-q06": [["不可以", "不能", "不得"], ["保留", "保存原始", "留下原始"], ["复核", "复查", "重新核对"]],
    "hold_observatory-q07": [["无法区分", "不能区分", "分不清"], ["跟踪抖动", "追踪抖动"], ["薄云"]],
    "hold_observatory-q09": [["二十五秒", "25秒", "25 秒"]],
    "hold_buoy-q01": [["连续四次", "连续4次", "连续 4 次"], ["超过一点五", "超过1.5", "大于1.5"]],
    "hold_buoy-q02": [["起吊前", "吊起前"], ["暂停采样", "停止采样"], ["日志落盘", "日志写盘", "日志保存"], ["检查渗水", "检查进水", "核查渗水"], ["再拷贝", "然后拷贝", "再复制", "后复制日志"]],
    "hold_buoy-q03": [["连续四次", "连续4次", "连续 4 次"], ["海浪", "波浪"], ["短暂露出", "暂时露出", "暴露于空气", "露出空气"]],
    "hold_buoy-q04": [["两小时", "2小时", "2 小时"]],
    "hold_buoy-q05": [["四小时", "4小时", "4 小时"]],
    "hold_buoy-q06": [["不能", "不可以", "不得"], ["潮汐"], ["不能自动清除", "不得清除", "保留历史轨迹", "不能擦除"]],
    "hold_buoy-q07": [["海浪", "波浪"], ["探头"], ["露出空气", "短暂露出", "接触空气", "暴露于空气"]],
    "hold_buoy-q09": [["七十秒", "70秒", "70 秒"]],
    "hold_theater-q01": [["十四个汉字", "14个汉字", "十四字", "14字", "14 个汉字"], ["停顿处", "停顿位置", "对白停顿"], ["两行", "2行", "2 行"]],
    "hold_theater-q02": [["同步时钟", "时钟同步"], ["测试对白", "对白测试"], ["返听"], ["操作员确认", "人工确认"], ["首句落点", "第一句落点", "首句位置"]],
    "hold_theater-q03": [["停顿处", "停顿位置", "对白停顿"], ["呼吸节奏", "对白节奏"], ["避免把否定词", "不能拆开否定词", "否定词和结论", "否定词与结论"]],
    "hold_theater-q04": [["暖白色", "暖白"]],
    "hold_theater-q05": [["浅绿色", "浅绿"]],
    "hold_theater-q06": [["不可以", "不能", "不得"], ["未说出", "尚未说出", "还没说", "未说出口"]],
    "hold_theater-q07": [["否定词和结论", "否定词与结论"], ["节奏"]],
    "hold_theater-q09": [["三百毫秒", "300毫秒", "300 毫秒", "300ms", "0.3秒"]],
    "hold_seedbank-q01": [["低于百分之八十二", "低于82%", "小于82%", "82%以下"], ["复检", "复测"], ["复检前不改", "复测前不改", "评级不变", "暂不改评级"]],
    "hold_seedbank-q02": [["密封回温", "密封状态回温"], ["四十分钟", "40分钟", "40 分钟"], ["称量", "称重"], ["开袋分装", "打开分装"], ["封口登记", "封口并登记", "登记流转"], ["放回余样", "余样放回", "余样归还"]],
    "hold_seedbank-q03": [["密封回温", "密封状态回温"], ["四十分钟", "40分钟", "40 分钟"], ["结露", "凝露", "冷凝"]],
    "hold_seedbank-q04": [["批次序号", "批次编号"], ["历史标签保留", "保留旧标签", "保留历史标签"], ["关联条码", "关联的条码"]],
    "hold_seedbank-q05": [["采集地", "采集地点", "产地"], ["年份", "采集年"]],
    "hold_seedbank-q06": [["不能", "不可以", "不得"], ["等待复检", "等待复测", "复检结果", "复测结果"], ["不能直接销毁", "不得销毁", "不应销毁"]],
    "hold_seedbank-q07": [["湿空气", "潮湿空气"], ["结露", "凝露", "冷凝"]],
    "hold_seedbank-q09": [["十一天", "11天", "11 天"]],
}
HELDOUT_FORBIDDEN = {
    "hold_observatory-q01": ["低于.{0,3}(3.2|三点二)"],
    "hold_observatory-q06": ["(?<!不)可以.{0,6}(删除|清掉|清除).{0,6}原始"],
    "hold_buoy-q01": ["一次.{0,6}就.{0,4}(提示|报警)"],
    "hold_buoy-q06": ["(?<!不能)自动清除历史轨迹", "(?<!不)可以.{0,6}擦除历史"],
    "hold_theater-q06": ["(?<!不)可以.{0,6}(预放|提前显示)"],
    "hold_seedbank-q01": ["高于.{0,3}(82%|百分之八十二)"],
    "hold_seedbank-q06": ["(?<!不)可以立即.{0,6}(报废|销毁)", "直接报废整个批次"],
}


def build(split):
    question_path = DATASET / ("questions." + split + ".json")
    destination = DATASET / ("rubric." + split + ".json")
    if destination.exists():
        raise SystemExit("Rubric sidecar already exists; version revisions explicitly.")
    points_by_id = DEV_POINTS if split == "dev" else HELDOUT_POINTS
    forbidden_by_id = DEV_FORBIDDEN if split == "dev" else HELDOUT_FORBIDDEN
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    rubrics = []
    for question in questions:
        qid = question["question_id"]
        absent = bool(question["unanswerable"])
        point_forms = [["资料不足", "没有提供", "未提供", "未说明", "无法确定", "不知道", "不清楚", "缺少依据", "无法回答"]] if absent else points_by_id[qid]
        points = [{"id": "p%02d" % (index + 1), "description": "明确资料不足，不编造具体值" if absent else "必要要点：" + forms[0], "any_of": forms}
                  for index, forms in enumerate(point_forms)]
        prohibited = [{"id": "f%02d" % (index + 1), "description": "不得将相冲突或越界的结论作为答案", "patterns": [pattern]}
                      for index, pattern in enumerate(forbidden_by_id.get(qid, []))]
        rubrics.append({"question_id": qid, "unanswerable": absent,
                        "acceptable_answers": ["资料未提供，无法确定。"] if absent else [question["answer"]],
                        "required_points": points, "forbidden_claims": prohibited})
    payload = {"schema_version": 1, "dataset_id": "plm-synthetic-zh-v1-80", "split": split,
               "questions_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
               "sources_sha256": hashlib.sha256((DATASET / "sources.json").read_bytes()).hexdigest(),
               "scoring_kind": "local_lexical_rubric_not_semantic_judge",
               "annotation_method": "Independently reviewed manual answer points; no model hypotheses used. Unanswerable refusal forms are shared, not generated gold facts.",
               "limitations": ["Surface forms can miss valid paraphrases.", "Forbidden-claim regexes can misread quoted or negated examples.", "Not official LongMemEval semantic evaluation."],
               "rubrics": rubrics}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"split": split, "rubrics": len(rubrics), "required_points": sum(len(item["required_points"]) for item in rubrics), "path": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    build(parser.parse_args().split)
