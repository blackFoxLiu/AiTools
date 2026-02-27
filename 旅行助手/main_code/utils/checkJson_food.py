import json
from typing import Any, Dict, List


def checkJson_food(data: Dict[str, Any]) -> bool:
    """
    验证美食推荐 JSON 数据是否符合预期结构。
    要求：
      - 根字段必须包含 "foods"，且为列表
      - 列表内每个对象必须包含字段：main_scenic, food_name, food_price, location, note
      - main_scenic 和 food_name 不能为空字符串或 None
      - 其他字段可为空字符串
    """
    # ---------- 类型检查辅助函数 ----------
    def _check_type(value: Any, expected: str) -> bool:
        """检查类型，str/list/dict 均接受空值。"""
        if expected == 'str':
            return isinstance(value, str)
        elif expected == 'list':
            return isinstance(value, list)
        elif expected == 'dict':
            return isinstance(value, dict)
        return False

    # 1. 根字段存在性及类型
    if "foods" not in data:
        print("缺少根字段: foods")
        return False
    if not _check_type(data["foods"], "list"):
        print("根字段 foods 类型错误，应为 list")
        return False

    # 2. 获取 foods 列表
    foods: List[Dict] = data["foods"]

    # 新增：检查 foods 不能为空
    if not foods:
        print("foods 列表不能为空")
        return False

    # 3. 遍历 foods 数组
    item_schema = {
        "main_scenic": "str",
        "food_name": "str",
        "food_price": "str",
        "location": "str",
        "note": "str"
    }

    for idx, item in enumerate(foods):
        if not isinstance(item, dict):
            print(f"foods[{idx}] 不是对象")
            return False

        # 检查每个必要字段是否存在且类型正确
        for field, typ in item_schema.items():
            if field not in item:
                print(f"foods[{idx}] 缺少字段: {field}")
                return False
            if not _check_type(item[field], typ):
                print(f"foods[{idx}] 字段 {field} 类型错误，应为 {typ}")
                return False

        # 特殊非空校验：main_scenic 和 food_name 不能为空字符串
        if item["main_scenic"] == "":
            print(f"foods[{idx}].main_scenic 不能为空")
            return False
        if item["food_name"] == "":
            print(f"foods[{idx}].food_name 不能为空")
            return False

        # 注意：空字符串对于其他字段是允许的，此处无需额外检查

    return True


def main(file_path: str) -> bool:
    """读取 JSON 文件并进行格式校验，返回布尔值。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取或解析 JSON 失败: {e}")
        return False

    return checkJson_food(data)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = main(sys.argv[1])
        print(result)
    else:
        # 无参数时，使用内置演示数据（符合要求）
        demo_json = """
            {
              "foods": [
                {
                  "main_scenic": "九寨沟",
                  "food_name": "烤牦牛肉串",
                  "food_price": "1元/串",
                  "location": "景区门口",
                  "note": "肉质鲜美"
                },
                {
                  "main_scenic": "长沙",
                  "food_name": "臭豆腐",
                  "food_price": "10元/份",
                  "location": "步行街",
                  "note": "外焦里嫩"
                }
              ]
            }
        """
        demo_data = json.loads(demo_json)
        print(checkJson_food(demo_data))