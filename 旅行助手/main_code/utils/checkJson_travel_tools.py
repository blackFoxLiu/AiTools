import json
from typing import Any, Dict


def checkJson_travel_tools(data: Dict[str, Any]) -> bool:
    """
    验证解析后的 JSON 字典是否符合预期结构。
    允许空列表和空字符串。
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

    # ---------- 根字段验证 ----------
    root_schema = {
        'provincial': 'str',
        'transportation': 'list',
        'hotels': 'list',
        'node_id': 'str'
    }
    for field, typ in root_schema.items():
        if field not in data:
            print(f"缺少根字段: {field}")
            return False
        if not _check_type(data[field], typ):
            print(f"根字段类型错误: {field} 应为 {typ}")
            return False

    # ---------- transportation 数组验证 ----------
    transport_schema = {
        'departure': 'str',
        'destination': 'str',
        'transportation_mode': 'list',
        'transportation_cost': 'str',
        'transportation_time': 'str',
        'transportation_diss': 'str',
        'notes': 'str'
    }
    for idx, item in enumerate(data['transportation']):
        if not isinstance(item, dict):
            print(f"transportation[{idx}] 不是对象")
            return False
        for field, typ in transport_schema.items():
            if field not in item:
                print(f"transportation[{idx}] 缺少字段: {field}")
                return False
            if not _check_type(item[field], typ):
                print(f"transportation[{idx}] 字段 {field} 类型应为 {typ}")
                return False
        # 检查 transportation_mode 数组元素类型
        for mode_idx, mode in enumerate(item['transportation_mode']):
            if not isinstance(mode, str):
                print(f"transportation[{idx}].transportation_mode[{mode_idx}] 不是字符串")
                return False

    # ---------- hotels 数组验证 ----------
    hotel_schema = {
        'hotel_name': 'str',
        'location': 'str',
        'price_range': 'str',
        'nearby_attractions': 'list',
        'price_notes': 'str'
    }
    for idx, item in enumerate(data['hotels']):
        if not isinstance(item, dict):
            print(f"hotels[{idx}] 不是对象")
            return False
        for field, typ in hotel_schema.items():
            if field not in item:
                print(f"hotels[{idx}] 缺少字段: {field}")
                return False
            if not _check_type(item[field], typ):
                print(f"hotels[{idx}] 字段 {field} 类型应为 {typ}")
                return False
        # 检查 nearby_attractions 数组元素类型
        for attr_idx, attr in enumerate(item['nearby_attractions']):
            if not isinstance(attr, str):
                print(f"hotels[{idx}].nearby_attractions[{attr_idx}] 不是字符串")
                return False

    return True


def main(file_path: str) -> bool:
    """读取 JSON 文件并进行格式校验。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取或解析 JSON 失败: {e}")
        return False

    if checkJson_travel_tools(data):
        print("✅ JSON 格式正确")
        return True
    else:
        print("❌ JSON 格式不正确")
        return False


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # 演示：使用允许空值的样例 JSON
        demo_json = """
{
  "provincial": "乌鲁木齐",
  "transportation": [
    {
      "departure": "喀纳斯机场",
      "destination": "喀纳斯湖",
      "transportation_mode": ["大巴"],
      "transportation_cost": "50元",
      "transportation_time": "5小时",
      "transportation_diss": "13公里",
      "notes": ""
    }
  ],
  "hotels": [],
  "node_id": "697c9561000000000a02e5a4"
}
        """
        demo_data = json.loads(demo_json)
        if checkJson_travel_tools(demo_data):
            print("✅ 演示 JSON 校验通过")
        else:
            print("❌ 演示 JSON 校验失败")