import json
from typing import Dict, Any

from tqdm import tqdm


def read_json_file(file_path):
    """读取JSON文件并处理每条数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []
    except json.JSONDecodeError:
        print(f"错误: {file_path} 不是有效的JSON文件")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return []


def get_statistics_structure(json_file):
    travel_dict = read_json_file(json_file)

    save_structure = {}
    for travel_info in tqdm(travel_dict, desc="处理中"):
        # 拿到省会 val
        for_provincial_val = travel_info.get("provincial", "")
        # 拿到景点 val
        for_main_scenic_val = travel_info.get("main_scenic", "")
        if len(for_main_scenic_val) == 0:
            continue
        # 打卡点列表 val
        journeys = travel_info.get("journeys", list)

        use_provincial_val = save_structure.get(for_provincial_val, {})
        save_structure[for_provincial_val] = use_provincial_val

        for journey in journeys:
            for_scenic = journey.get("scenic", "")
            if len(for_scenic) == 0:
                continue

            for_season = journey.get("season", [])
            for_suit_months_range = journey.get("suit_months_range", [])
            for_scenic_intro = journey.get("scenic_intro", [])
            for_recommand = journey.get("recommand", [])
            for_tendency_label_1 = journey.get("tendency_label_1", [])
            for_tendency_label_2 = journey.get("tendency_label_2", [])

            # 检查景点是否存在，如果存在，对信息进行重设
            use_sub_scenic = use_provincial_val.get(for_scenic, {})
            use_provincial_val[for_scenic] = use_sub_scenic
            cnt_season = use_sub_scenic.get("cnt_season", {})
            use_sub_scenic["cnt_season"] = cnt_season
            for tmp_season in for_season:
                cnt_season_nums = cnt_season.get(tmp_season, 0)
                cnt_season[tmp_season] = cnt_season_nums+1

            cnt_suit_months_range = use_sub_scenic.get("suit_months_range", {})
            use_sub_scenic["suit_months_range"] = cnt_suit_months_range
            for tmp_suit_months_range in for_suit_months_range:
                cnt_suit_months_range_nums = cnt_suit_months_range.get(tmp_suit_months_range, 0)
                cnt_suit_months_range[tmp_suit_months_range] = cnt_suit_months_range_nums+1

            use_scenic_intro = use_provincial_val.get("scenic_intro", "")
            if len(for_scenic_intro) != 0:
                use_scenic_intro = for_scenic_intro+";"+use_scenic_intro
                use_sub_scenic["scenic_intro"] = use_scenic_intro

            if len(for_recommand) != 0:
                cnt_recommand = use_provincial_val.get("recommand", {})
                use_sub_scenic["recommand"] = cnt_recommand
                cnt_recommand[for_recommand] = cnt_recommand.get(for_recommand, 0) + 1

            if isinstance(for_tendency_label_1, list) or isinstance(for_tendency_label_2, list):
                continue
            cnt_tendency_label_1 = use_provincial_val.get("tendency_label_1", {})
            use_sub_scenic["tendency_label_1"] = cnt_tendency_label_1
            cnt_tendency_label_1[for_tendency_label_1] = cnt_tendency_label_1.get(for_tendency_label_1, 0) + 1

            cnt_tendency_label_2 = use_provincial_val.get("tendency_label_2", {})
            use_sub_scenic["tendency_label_2"] = cnt_tendency_label_2
            cnt_tendency_label_2[for_tendency_label_2] = cnt_tendency_label_2.get(for_tendency_label_2, 0) + 1
    return save_structure


# 查找到最大的类型
def find_max_cnt(op_dict: Dict[str, Any]):
    if op_dict == {}:
        return ""
    max_num = 0
    max_type = ""
    for key in op_dict.keys():
        tmp_cnt = op_dict[key]
        if tmp_cnt > max_num:
            max_num = op_dict[key]
            max_type = key
    return max_type


# 获取信息
def get_travel_info(json_file):
    rst_structure = get_statistics_structure(json_file)
    travel_dict = dict()
    for dict_key in rst_structure.keys():
        for sub_dict_key in rst_structure[dict_key].keys():
            tmp_travel_info = rst_structure[dict_key][sub_dict_key]
            travel_dict[sub_dict_key] = {
                "provincial": dict_key,
                "season": find_max_cnt(tmp_travel_info["cnt_season"]),
                "suit_months_range": find_max_cnt(tmp_travel_info["suit_months_range"]),
                "recommand": find_max_cnt(tmp_travel_info.get("recommand", {})),
                "tendency_label_1": find_max_cnt(tmp_travel_info.get("tendency_label_1", {})),
                "tendency_label_2": find_max_cnt(tmp_travel_info.get("tendency_label_2", {}))
            }
    return travel_dict


if __name__ == '__main__':
    # 替换为你的JSON文件路径
    json_file = "C:/Users/13187/Desktop/travel_analysis_1.json"
    travel_list = get_travel_info(json_file)