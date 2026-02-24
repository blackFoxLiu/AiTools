from typing import Any, List, Tuple

# ---------- 字段规则定义 ----------
SEASON_VALUES = {"春季", "夏季", "秋季", "冬季"}
MONTH_VALUES = {str(i) for i in range(1, 13)}   # "1" ~ "12"
SCENIC_TYPE_VALUES = {"经验分享", "主观感受"}

# 必需的顶级字段及其类型约束（所有字段均为必需）
TOP_LEVEL_FIELDS = {
    "provincial": str,
    "main_scenic": str,
    "journeys": list,
    "other_recommend": list,
    "scheduling": list,
    "copewriting_type": str,
}

# journeys 中每个对象必须包含的字段及其类型约束
JOURNEY_ITEM_FIELDS = {
    "scenic": str,
    "season": list,
    "suit_months_range": list,
    "scenic_intro": str,
    "recommand": str,
}

# scheduling 中每个对象必须包含的字段及其类型约束
SCHEDULING_ITEM_FIELDS = {
    "scenic": str,
    "time_required": str,
}


def _add_error(errors: List[str], path: str, msg: str) -> None:
    """添加格式统一的错误信息"""
    errors.append(f"字段 '{path}': {msg}")


def _check_type(value: Any, expected_type: type, path: str, errors: List[str]) -> bool:
    """类型检查，失败时记录错误"""
    if not isinstance(value, expected_type):
        _add_error(errors, path, f"应为 {expected_type.__name__} 类型，实际为 {type(value).__name__}")
        return False
    return True


def _check_list_of_strings(lst: List[Any], path: str, errors: List[str]) -> bool:
    """检查列表中的每个元素是否为字符串（允许空列表）"""
    valid = True
    for idx, item in enumerate(lst):
        if not isinstance(item, str):
            _add_error(errors, f"{path}[{idx}]", "应为字符串")
            valid = False
    return valid


def _validate_season(season_list: List[str], path: str, errors: List[str]) -> bool:
    """
    校验 season 字段：
    - 必须为列表（已在外部检查）
    - 允许空列表
    - 非空时每个元素必须为 SEASON_VALUES 中的值
    """
    valid = True
    for idx, season in enumerate(season_list):
        if season not in SEASON_VALUES:
            _add_error(errors, f"{path}[{idx}]", f"季节值 '{season}' 不在允许集合 {SEASON_VALUES} 中")
            valid = False
    return valid


def _validate_suit_months(month_list: List[str], path: str, errors: List[str]) -> bool:
    """
    校验 suit_months_range 字段：
    - 必须为列表（已在外部检查）
    - 允许空列表
    - 非空时每个元素必须为 MONTH_VALUES 中的值
    """
    valid = True
    for idx, month in enumerate(month_list):
        if month not in MONTH_VALUES:
            _add_error(errors, f"{path}[{idx}]", f"月份值 '{month}' 不在允许集合 1-12 中")
            valid = False
    return valid


def _validate_scenic_type(copewriting_type: str, path: str, errors: List[str]) -> bool:
    """
    校验 copewriting_type 字段：
    - 必须为字符串（已在外部检查）
    - 允许空字符串
    - 非空时必须在 SCENIC_TYPE_VALUES 中
    """
    if copewriting_type == "":
        return True
    if copewriting_type not in SCENIC_TYPE_VALUES:
        _add_error(errors, path, f"取值应为 {SCENIC_TYPE_VALUES} 之一或空字符串，实际为 '{copewriting_type}'")
        return False
    return True


def check_travel_analysis(data: Any) -> Tuple[bool, List[str]]:
    """
    校验旅行推荐JSON格式的主函数（基于最新模板）
    :param data: 已解析的JSON数据（通常为dict）
    :return: (是否通过, 错误信息列表)
    """
    errors = []

    # 1. 根节点必须是对象
    if not isinstance(data, dict):
        _add_error(errors, "<root>", f"根节点应为对象(dict)，实际为 {type(data).__name__}")
        return False, errors

    # 2. 检查所有必需的顶级字段是否存在
    for field, expected_type in TOP_LEVEL_FIELDS.items():
        if field not in data:
            _add_error(errors, field, "字段缺失")
            continue  # 缺失则跳过后续类型校验

        value = data[field]
        if not _check_type(value, expected_type, field, errors):
            continue  # 类型错误，跳过该字段的深度校验

        # 3. 按字段进行深度校验
        if field == "provincial" or field == "main_scenic":
            # 普通字符串字段，允许空字符串，无额外取值限制
            pass

        elif field == "journeys":
            # journeys 是列表，每个元素是对象
            for i, journey in enumerate(value):
                path_prefix = f"journeys[{i}]"
                if not isinstance(journey, dict):
                    _add_error(errors, path_prefix, f"应为对象，实际为 {type(journey).__name__}")
                    continue

                # 检查 journey 对象必需的字段
                for jf, jf_type in JOURNEY_ITEM_FIELDS.items():
                    if jf not in journey:
                        _add_error(errors, f"{path_prefix}.{jf}", "字段缺失")
                        continue
                    jv = journey[jf]
                    if not _check_type(jv, jf_type, f"{path_prefix}.{jf}", errors):
                        continue

                    # 特殊字段值校验
                    if jf == "season":
                        _validate_season(jv, f"{path_prefix}.season", errors)
                    elif jf == "suit_months_range":
                        _validate_suit_months(jv, f"{path_prefix}.suit_months_range", errors)
                    # 其他字符串字段（scenic, scenic_intro, recommand）无额外限制，允许空字符串

        elif field == "other_recommend":
            # other_recommend 是字符串数组
            _check_list_of_strings(value, "other_recommend", errors)

        elif field == "scheduling":
            # scheduling 是列表，每个元素是对象
            for i, item in enumerate(value):
                path_prefix = f"scheduling[{i}]"
                if not isinstance(item, dict):
                    _add_error(errors, path_prefix, f"应为对象，实际为 {type(item).__name__}")
                    continue

                for sf, sf_type in SCHEDULING_ITEM_FIELDS.items():
                    if sf not in item:
                        _add_error(errors, f"{path_prefix}.{sf}", "字段缺失")
                        continue
                    sv = item[sf]
                    _check_type(sv, sf_type, f"{path_prefix}.{sf}", errors)
                    # 无额外值约束，允许空字符串

        elif field == "copewriting_type":
            # copewriting_type 字符串校验
            _validate_scenic_type(value, "copewriting_type", errors)

    # 4. 返回校验结果
    return len(errors) == 0, errors