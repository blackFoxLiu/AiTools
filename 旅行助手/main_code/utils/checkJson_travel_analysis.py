import json
from typing import Any, Dict, List, Set, Union


def check_travel_analysis(json_input: Union[str, Dict[str, Any]]) -> bool:
    """
    检查 JSON 数据是否符合要求的格式。

    参数:
        json_input: JSON 字符串或已经解析的字典。

    返回:
        bool: 符合要求返回 True，否则返回 False。
    """
    # 解析 JSON（如果是字符串）
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return False
    else:
        data = json_input

    errors: List[str] = []

    # ---------- 顶级字段检查 ----------
    # provincial
    if "provincial" not in data:
        errors.append("缺少顶级字段: provincial")
    else:
        if not isinstance(data["provincial"], str):
            errors.append("provincial 类型错误，应为字符串")
        elif data["provincial"].strip() == "":
            errors.append("provincial 不能为空")

    # main_scenic_list
    if "main_scenic_list" not in data:
        errors.append("缺少顶级字段: main_scenic_list")
    else:
        main_list = data["main_scenic_list"]
        if not isinstance(main_list, list):
            errors.append("main_scenic_list 类型错误，应为列表")
        elif len(main_list) == 0:
            errors.append("main_scenic_list 至少包含一个元素")
        else:
            for i, scenic_item in enumerate(main_list):
                if not isinstance(scenic_item, dict):
                    errors.append(f"main_scenic_list[{i}] 类型错误，应为字典")
                    continue

                # main_scenic
                if "main_scenic" not in scenic_item:
                    errors.append(f"main_scenic_list[{i}] 缺少字段: main_scenic")
                else:
                    ms = scenic_item["main_scenic"]
                    if not isinstance(ms, str):
                        errors.append(f"main_scenic_list[{i}].main_scenic 类型错误，应为字符串")
                    elif ms.strip() == "":
                        errors.append(f"main_scenic_list[{i}].main_scenic 不能为空")

                # journeys (可选)
                if "journeys" in scenic_item:
                    journeys = scenic_item["journeys"]
                    if not isinstance(journeys, list):
                        errors.append(f"main_scenic_list[{i}].journeys 类型错误，应为列表")
                    else:
                        for j, journey_item in enumerate(journeys):
                            if not isinstance(journey_item, dict):
                                errors.append(f"main_scenic_list[{i}].journeys[{j}] 类型错误，应为字典")
                                continue

                            # scenic
                            if "scenic" not in journey_item:
                                errors.append(f"main_scenic_list[{i}].journeys[{j}] 缺少字段: scenic")
                            else:
                                sc = journey_item["scenic"]
                                if not isinstance(sc, str):
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].scenic 类型错误，应为字符串")
                                elif sc.strip() == "":
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].scenic 不能为空")

                            # season
                            if "season" not in journey_item:
                                errors.append(f"main_scenic_list[{i}].journeys[{j}] 缺少字段: season")
                            else:
                                season_val = journey_item["season"]
                                if not isinstance(season_val, list):
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].season 类型错误，应为列表")
                                else:
                                    valid_seasons: Set[str] = {"春季", "夏季", "秋季", "冬季"}
                                    for k, s in enumerate(season_val):
                                        if not isinstance(s, str):
                                            errors.append(f"main_scenic_list[{i}].journeys[{j}].season[{k}] 类型错误，应为字符串")
                                        elif s not in valid_seasons:
                                            errors.append(
                                                f"main_scenic_list[{i}].journeys[{j}].season[{k}] 值 '{s}' 无效，"
                                                f"必须为 {valid_seasons}"
                                            )

                            # suit_months_range
                            if "suit_months_range" not in journey_item:
                                errors.append(f"main_scenic_list[{i}].journeys[{j}] 缺少字段: suit_months_range")
                            else:
                                months_val = journey_item["suit_months_range"]
                                if not isinstance(months_val, list):
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].suit_months_range 类型错误，应为列表")
                                else:
                                    valid_months: Set[str] = {str(m) for m in range(1, 13)}
                                    for k, m in enumerate(months_val):
                                        if not isinstance(m, str):
                                            errors.append(
                                                f"main_scenic_list[{i}].journeys[{j}].suit_months_range[{k}] 类型错误，应为字符串"
                                            )
                                        elif m not in valid_months:
                                            errors.append(
                                                f"main_scenic_list[{i}].journeys[{j}].suit_months_range[{k}] 值 '{m}' 无效，"
                                                f"必须为 1-12 的字符串"
                                            )

                            # tendency_label_1
                            if "tendency_label_1" not in journey_item:
                                errors.append(f"main_scenic_list[{i}].journeys[{j}] 缺少字段: tendency_label_1")
                            else:
                                t1 = journey_item["tendency_label_1"]
                                if not isinstance(t1, str):
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].tendency_label_1 类型错误，应为字符串")
                                elif t1.strip() == "":
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].tendency_label_1 不能为空")

                            # tendency_label_2
                            if "tendency_label_2" not in journey_item:
                                errors.append(f"main_scenic_list[{i}].journeys[{j}] 缺少字段: tendency_label_2")
                            else:
                                t2 = journey_item["tendency_label_2"]
                                if not isinstance(t2, str):
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].tendency_label_2 类型错误，应为字符串")
                                elif t2.strip() == "":
                                    errors.append(f"main_scenic_list[{i}].journeys[{j}].tendency_label_2 不能为空")

                # other_recommend 字段可选，不强制检查

    # copewriting_type
    if "copewriting_type" not in data:
        errors.append("缺少顶级字段: copewriting_type")
    else:
        ct = data["copewriting_type"]
        if not isinstance(ct, str):
            errors.append("copewriting_type 类型错误，应为字符串")
        elif ct not in ["经验分享", "主观感受"]:
            errors.append("copewriting_type 值必须为 '经验分享' 或 '主观感受'")

    # 输出所有错误
    for err in errors:
        print(err)

    return len(errors) == 0


# 示例用法（可取消注释进行测试）
if __name__ == "__main__":
    # 正确示例
    valid_json = """
    {
        "provincial": "四川省",
        "main_scenic_list": [
            {
                "main_scenic": "九寨沟",
                "journeys": [
                    {
                        "scenic": "五花海",
                        "season": [],
                        "suit_months_range": [],
                        "scenic_intro": "美丽的湖泊",
                        "time_required": "2小时",
                        "recommand": "必去",
                        "tendency_label_1": "自然风光",
                        "tendency_label_2": "摄影天堂"
                    }
                ],
                "other_recommend": ["黄龙", "若尔盖"]
            }
        ],
        "copewriting_type": "主观感受"
    }
    """
    print("检查结果:", check_travel_analysis(valid_json))  # 应为 True
