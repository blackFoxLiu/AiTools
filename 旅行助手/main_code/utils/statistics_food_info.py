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
    food_list = read_json_file(json_file)

    save_structure = {}
    for food_dict in tqdm(food_list, desc="处理中"):
        food_list = food_dict.get("foods", [])
        if len(food_list) == 0:
            continue
        for food in food_list:
            main_scenic = food.get("main_scenic", "")
            food_name = food.get("food_name", "")
            rst_food_name = main_scenic + food_name
            use_food_dict = save_structure.get(rst_food_name, {})
            save_structure[rst_food_name] = use_food_dict

            use_node = food.get("note", "")
            if len(use_node) != 0:
                use_food_dict["note"] = use_node + ";" + use_food_dict.get("note", "")

            use_food_price = food.get("food_price", "")
            if len(use_food_price) != 0:
                use_food_dict["food_price"] = use_food_price + ";" + use_food_dict.get("food_price", "")

            use_location = food.get("location", "")
            if len(use_location) != 0:
                use_food_dict["location"] = use_location + ";" + use_food_dict.get("location", "")

            use_food_dict["main_scenic"] = food.get("main_scenic", "")

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
def get_food_info(json_file):
    return get_statistics_structure(json_file)


if __name__ == '__main__':
    # 替换为你的JSON文件路径
    json_file = "C:/Users/13187/Desktop/food.json"
    travel_list = get_food_info(json_file)
    print(str(travel_list))