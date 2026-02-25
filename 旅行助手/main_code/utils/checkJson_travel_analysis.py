import json
import logging
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 允许的季节和月份
ALLOWED_SEASONS: Set[str] = {"春季", "夏季", "秋季", "冬季"}
ALLOWED_MONTHS: Set[str] = {str(i) for i in range(1, 13)}


def check_travel_analysis(data: Dict[str, Any]) -> bool:
    """
    验证给定的字典是否符合 JSON 格式要求。
    返回 True 表示通过，False 表示失败，同时打印所有错误日志。
    """
    errors: List[str] = []          # 收集严重错误
    warnings: List[str] = []        # 收集警告（不影响最终结果）

    def add_error(msg: str) -> None:
        errors.append(msg)

    def add_warning(msg: str) -> None:
        warnings.append(msg)

    # 辅助函数：检查必填字符串字段
    def check_required_str(obj: Any, field: str, path: str) -> bool:
        """检查字段是否存在、类型为 str 且非空。返回 True 表示通过，否则向 errors 添加错误并返回 False。"""
        if field not in obj:
            add_error(f"{path} 缺少字段 '{field}'")
            return False
        val = obj[field]
        if not isinstance(val, str):
            add_error(f"{path}['{field}'] 必须是字符串，实际是 {type(val).__name__}")
            return False
        if not val.strip():
            add_error(f"{path}['{field}'] 不能为空")
            return False
        return True

    # 辅助函数：检查可选列表字段（如果存在，则必须是列表）
    def check_optional_list(obj: Any, field: str, path: str) -> bool:
        """如果字段存在，检查其类型是否为 list。返回 True 表示通过或字段不存在，否则添加错误并返回 False。"""
        if field in obj:
            if not isinstance(obj[field], list):
                add_error(f"{path}['{field}'] 必须是列表，实际是 {type(obj[field]).__name__}")
                return False
        return True

    # 辅助函数：检查列表中的每个元素是否为指定类型（可选）
    def check_list_elements(lst: List, expected_type: type, item_path_prefix: str) -> bool:
        """检查列表 lst 中每个元素是否为 expected_type。返回 True 表示所有元素符合，否则添加错误并返回 False。"""
        valid = True
        for idx, elem in enumerate(lst):
            if not isinstance(elem, expected_type):
                add_error(f"{item_path_prefix}[{idx}] 必须是 {expected_type.__name__}，实际是 {type(elem).__name__}")
                valid = False
        return valid

    # 辅助函数：检查集合包含关系
    def check_allowed_values(value: Any, allowed_set: Set[str], path: str) -> bool:
        if value not in allowed_set:
            add_error(f"{path} 值 '{value}' 不在允许集合 {allowed_set} 中")
            return False
        return True

    # ---------- 顶层字段检查 ----------
    # provincial
    check_required_str(data, "provincial", "顶层")
    # main_scenic_list
    if "main_scenic_list" not in data:
        add_error("顶层缺少字段 'main_scenic_list'")
    elif not isinstance(data["main_scenic_list"], list):
        add_error("'main_scenic_list' 必须是列表")
    elif len(data["main_scenic_list"]) == 0:
        add_error("'main_scenic_list' 至少包含一个元素")

    # 如果顶层已有严重错误，直接返回 False（避免后续因缺少字段而崩溃）
    if errors:
        for err in errors:
            logger.error(f"错误:", err)
        for warn in warnings:
            logger.error(f"警告:", warn)
        return False

    # ---------- 遍历 main_scenic_list ----------
    main_list = data["main_scenic_list"]
    for idx, scenic_dict in enumerate(main_list):
        base_path = f"main_scenic_list[{idx}]"

        if not isinstance(scenic_dict, dict):
            add_error(f"{base_path} 必须是字典")
            continue   # 跳过后续对该元素的检查

        # main_scenic
        check_required_str(scenic_dict, "main_scenic", base_path)
        # tendency_label_1
        check_required_str(scenic_dict, "tendency_label_1", base_path)
        # tendency_label_2
        check_required_str(scenic_dict, "tendency_label_2", base_path)

        # other_recommend (可选列表)
        if "other_recommend" in scenic_dict:
            if check_optional_list(scenic_dict, "other_recommend", base_path):
                other_list = scenic_dict["other_recommend"]
                check_list_elements(other_list, str, f"{base_path}['other_recommend']")

        # journeys (可选列表)
        if "journeys" in scenic_dict:
            if not check_optional_list(scenic_dict, "journeys", base_path):
                continue
            journeys = scenic_dict["journeys"]
            # 即使 journeys 为空列表，也无需检查内部元素
            for jdx, journey in enumerate(journeys):
                journey_path = f"{base_path}['journeys'][{jdx}]"
                if not isinstance(journey, dict):
                    add_error(f"{journey_path} 必须是字典")
                    continue

                # scenic
                check_required_str(journey, "scenic", journey_path)

                # season
                if "season" not in journey:
                    add_error(f"{journey_path} 缺少字段 'season'")
                elif not isinstance(journey["season"], list):
                    add_error(f"{journey_path}['season'] 必须是列表")
                else:
                    season_list = journey["season"]
                    # 检查每个元素类型
                    for s_idx, s in enumerate(season_list):
                        s_path = f"{journey_path}['season'][{s_idx}]"
                        if not isinstance(s, str):
                            add_error(f"{s_path} 必须是字符串")
                        else:
                            check_allowed_values(s, ALLOWED_SEASONS, s_path)

                # suit_months_range
                if "suit_months_range" not in journey:
                    add_error(f"{journey_path} 缺少字段 'suit_months_range'")
                elif not isinstance(journey["suit_months_range"], list):
                    add_error(f"{journey_path}['suit_months_range'] 必须是列表")
                else:
                    months_list = journey["suit_months_range"]
                    for m_idx, m in enumerate(months_list):
                        m_path = f"{journey_path}['suit_months_range'][{m_idx}]"
                        if not isinstance(m, str):
                            add_error(f"{m_path} 必须是字符串")
                        else:
                            check_allowed_values(m, ALLOWED_MONTHS, m_path)

                # 可选字段的类型检查（仅警告）
                optional_fields = ["scenic_intro", "time_required", "recommand"]
                for field in optional_fields:
                    if field in journey and not isinstance(journey[field], str):
                        add_warning(f"{journey_path}['{field}'] 应该是字符串，实际是 {type(journey[field]).__name__}")

    # ---------- 输出结果 ----------
    if errors:
        for err in errors:
            logger.error(f"错误:", err)
        for warn in warnings:
            logger.error(f"警告:", warn)
        logger.error("JSON 格式验证失败！")
        return False
    else:
        if warnings:
            for warn in warnings:
                logger.error(f"警告:", warn)
        logger.info("JSON 格式验证通过！")
        return True


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
                        "recommand": "必去"
                    }
                ],
                "other_recommend": ["黄龙", "若尔盖"],
                "tendency_label_1": "自然风光",
                "tendency_label_2": "摄影天堂"
            }
        ]
    }
    """
    print("检查结果:", check_travel_analysis(json.loads(valid_json)))  # 应为 True
