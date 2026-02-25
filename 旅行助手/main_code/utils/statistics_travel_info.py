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
        for_main_scenic_list = travel_info.get("main_scenic_list", [])
        if len(for_main_scenic_list) == 0:
            continue

        for main_scenic in for_main_scenic_list:
            # 打卡点列表 val
            for_main_scenic = main_scenic.get("main_scenic", "")
            journeys = main_scenic.get("journeys", [])
            if not journeys:
                journeys.append({
                "scenic": for_main_scenic,
                "season": [],
                "suit_months_range": [],
                "scenic_intro": "",
                "time_required": "",
                "other_recommend": ""
            })

            use_main_scenic_val = save_structure.get(for_main_scenic, {})
            save_structure[for_main_scenic] = use_main_scenic_val
            use_main_scenic_val["provincial"] = for_provincial_val

            other_recommend = main_scenic.get("other_recommend", "")
            use_main_scenic_val["other_recommend"] = use_main_scenic_val.get("other_recommend", "") + ";".join(other_recommend)

            use_scenic_dict = use_main_scenic_val.get("scenic_dict", {})
            use_main_scenic_val["scenic_dict"] = use_scenic_dict

            for_tendency_label_1 = main_scenic.get("tendency_label_1", "")
            for_tendency_label_2 = main_scenic.get("tendency_label_2", "")

            for journey in journeys:
                for_scenic = journey.get("scenic", "")
                if len(for_scenic) == 0:
                    continue

                use_scenic_val = use_scenic_dict.get(for_scenic, {})
                use_scenic_dict[for_scenic] = use_scenic_val

                for_season = journey.get("season", [])
                for_suit_months_range = journey.get("suit_months_range", [])
                for_scenic_intro = journey.get("scenic_intro", [])

                cnt_season = use_scenic_val.get("cnt_season", {})
                use_scenic_val["cnt_season"] = cnt_season
                for tmp_season in for_season:
                    cnt_season_nums = cnt_season.get(tmp_season, 0)
                    cnt_season[tmp_season] = cnt_season_nums+1

                cnt_suit_months_range = use_scenic_val.get("suit_months_range", {})
                use_scenic_val["suit_months_range"] = cnt_suit_months_range
                for tmp_suit_months_range in for_suit_months_range:
                    cnt_suit_months_range_nums = cnt_suit_months_range.get(tmp_suit_months_range, 0)
                    cnt_suit_months_range[tmp_suit_months_range] = cnt_suit_months_range_nums+1

                use_scenic_intro = use_scenic_val.get("scenic_intro", "")
                if len(for_scenic_intro) != 0:
                    use_scenic_intro = for_scenic_intro+";"+use_scenic_intro
                    use_scenic_val["scenic_intro"] = use_scenic_intro

                if isinstance(for_tendency_label_1, list) or isinstance(for_tendency_label_2, list):
                    continue
                cnt_tendency_label_1 = use_scenic_val.get("tendency_label_1", {})
                use_scenic_val["tendency_label_1"] = cnt_tendency_label_1
                cnt_tendency_label_1[for_tendency_label_1] = cnt_tendency_label_1.get(for_tendency_label_1, 0) + 1

                cnt_tendency_label_2 = use_scenic_val.get("tendency_label_2", {})
                use_scenic_val["tendency_label_2"] = cnt_tendency_label_2
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
        tmp_travel_info = rst_structure[dict_key]
        tmp_scenic_dict = tmp_travel_info.get("scenic_dict", [])

        for scenic_name in tmp_scenic_dict:
            scenic_dict = tmp_scenic_dict.get(scenic_name)

            travel_dict[scenic_name] = {
                "provincial": tmp_travel_info.get("provincial", ""),
                "main_scenic": dict_key,
                "scenic": scenic_name,
                "season": find_max_cnt(scenic_dict.get("cnt_season", {})),
                "suit_months_range": find_max_cnt(scenic_dict.get("suit_months_range", {})),
                "tendency_label_1": find_max_cnt(scenic_dict.get("tendency_label_1", {})),
                "tendency_label_2": find_max_cnt(scenic_dict.get("tendency_label_2", {})),
                "other_recommend": tmp_travel_info.get("other_recommend", "")
            }
    return travel_dict


if __name__ == '__main__':
    # 替换为你的JSON文件路径
    json_file = "C:/Users/13187/Desktop/travelAnalysis.txt"
    travel_list = get_travel_info(json_file)
    print(str(travel_list))