#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手整合数据脚本（适配新数据模型）
"""

import logging
import os
from typing import Optional, Dict, List, Set
import configparser

from py2neo import Graph

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
# 文件信息配置
SCENIC_INFO_FILE = config['model_output']['SCENIC_INFO_FILE']
rag_file_path = config['file_path']['rag_file_path']
TOOLS_INFO_FILE = config['model_output']['TOOLS_INFO_FILE']
HOTELS_INFO_FILE = config['model_output']['HOTELS_INFO_FILE']
neo4j_uri = config['neo4j']['neo4j_uri']
account = config['neo4j']['account']
password = config['neo4j']['password']

# ==================== 配置信息 ====================
NEO4J_CONFIG = {
    "uri": neo4j_uri,
    "user": account,
    "password": password
}

PAGE_SIZE = 20  # 分页大小


# ==================== 数据库连接 ====================
class Neo4jConnection:
    def __init__(self, config: Dict[str, str]):
        self.graph = Graph(config["uri"], auth=(config["user"], config["password"]))

    def run_query(self, cql: str, **params):
        """执行Cypher查询并返回Cursor（流式迭代）"""
        try:
            return self.graph.run(cql, **params)
        except Exception as e:
            logging.error(f"查询执行失败: {e}, CQL: {cql}, 参数: {params}")
            raise


# ==================== 分页查询 Main_Scenic 基本信息 ====================
def fetch_main_scenic_page(db: Neo4jConnection, skip: int, limit: int):
    """
    获取一页 Main_Scenic 节点及其所属省份名称
    返回列表，每个元素为 (scenic_node, provincial_name)
    """
    cql = """
        MATCH (s:Main_Scenic)
        OPTIONAL MATCH (s)-[:belong_to]->(p:Provincial)
        RETURN s, p.name AS provincial_name
        ORDER BY s.name
        SKIP $skip
        LIMIT $limit
    """
    cursor = db.run_query(cql, skip=skip, limit=limit)
    results = []
    for record in cursor:
        results.append((record["s"], record["provincial_name"]))
    return results


# ==================== 查询 Scenic 节点 ====================
def fetch_scenic_for_main_scenic(db: Neo4jConnection, main_scenic_name: str):
    cql = """
        MATCH (s:Main_Scenic {name: $name})<-[:include]-(sc:Scenic)
        RETURN sc.name as spot_name
    """
    cursor = db.run_query(cql, name=main_scenic_name)
    # 直接遍历游标，提取每个记录的 spot_name
    result = [record["spot_name"] for record in cursor]
    return result


# ==================== 批量查询 ScenicHotel 节点 ====================
def fetch_hotel_for_main_scenic(db: Neo4jConnection, main_scenic_names: List[str]):
    """
    根据 Main_Scenic 的 name 列表，查询每个 Main_Scenic 关联的所有 ScenicHotel 节点
    返回字典：{ scenic_name: [hotel_node, ...] }
    """
    if not main_scenic_names:
        return {}
    cql = """
        MATCH (s:Main_Scenic)-[:exists]->(h:ScenicHotel)
        WHERE s.name IN $names
        RETURN s.name AS scenic_name, collect(h) AS hotel_list
    """
    cursor = db.run_query(cql, names=main_scenic_names)
    result_dict = {}
    for record in cursor:
        result_dict[record["scenic_name"]] = record["hotel_list"]
    return result_dict


# ==================== 批量查询 Scenic_From_To 及 TravelTool ====================
def fetch_from_to_for_main_scenic(db: Neo4jConnection, main_scenic_names: List[str]):
    """
    根据 Main_Scenic 的 name 列表，查询与之关联的所有 Scenic_From_To 节点及其 TravelTool
    返回列表，每个元素为 {"from_to": node, "tools": [tool_node, ...]}
    注意：已通过 DISTINCT 对 from_to 去重，每个 from_to 只返回一次
    """
    if not main_scenic_names:
        return []
    cql = """
        MATCH (s:Main_Scenic)-[:from_to]->(f:Scenic_From_To)
        WHERE s.name IN $names
        WITH DISTINCT f
        OPTIONAL MATCH (f)-[:tools]-(t:TravelTool)
        RETURN f, collect(t) AS tools
    """
    cursor = db.run_query(cql, names=main_scenic_names)
    results = []
    for record in cursor:
        results.append({
            "from_to": record["f"],
            "tools": record["tools"]
        })
    return results


# ==================== 获取 Main_Scenic 总数 ====================
def get_scenic_count(db: Neo4jConnection) -> int:
    """获取 Main_Scenic 总数"""
    result = db.run_query("MATCH (s:Main_Scenic) RETURN count(s) AS total")
    return result.evaluate()


# ==================== 信息生成函数（保持不变） ====================
def format_scenic_info(scenic_node, provincial_name: Optional[str], spopt_nodes) -> str:
    """生成景点基本信息描述"""
    props = dict(scenic_node)
    name = props.get("name")
    if not name:
        return ""

    parts = []
    if provincial_name:
        parts.append(f"{name}位于{provincial_name}")

    season = props.get("season")
    suit_months = props.get("suit_months_range")
    tendency1 = props.get("tendency_label_1")
    tendency2 = props.get("tendency_label_2")
    other_recommend = props.get("other_recommend")
    spot_str = "包含景点包括" + "、".join(spopt_nodes)
    parts.append(spot_str)

    if season:
        parts.append(f"适合季节为{season}")
    if suit_months:
        parts.append(f"适合月份为{suit_months}")
    if tendency1:
        parts.append(f"小红书上推荐景点倾向标签为{tendency1}")
    if tendency2:
        parts.append(f"小红书上推荐景点倾向次级标签为{tendency2}")
    if other_recommend:
        parts.append(f"其他推荐信息：{other_recommend}")

    return "，".join(parts) + "。" if parts else ""


def format_tools_info(from_to_list: List[Dict], processed_from_to: Set[str]) -> List[str]:
    """
    生成交通信息，每条from_to生成一条描述（如果尚未处理过）
    processed_from_to: 用于全局去重的集合（存储from_to节点name）
    返回描述字符串列表
    """
    lines = []
    for item in from_to_list:
        ft_node = item["from_to"]
        ft_name = ft_node.get("name")
        if not ft_name or ft_name in processed_from_to:
            continue
        processed_from_to.add(ft_name)

        # 解析 from_to 名称，格式应为 "A=B"
        places = ft_name.split("=")
        if len(places) != 2:
            continue
        departure, destination = places[0], places[1]

        tools = item["tools"]
        if not tools:
            continue

        line = f"景点{departure}到达{destination}"
        tool_parts = []
        for tool_node in tools:
            tool = dict(tool_node)
            tool_name = tool.get("name")
            if not tool_name:
                continue
            desc = f"交通方式{tool_name}"
            cost = tool.get("trans_cost")
            if cost:
                desc += f"，花费{cost}"
            time = tool.get("trans_time")
            if time:
                desc += f"，需要时间{time}"
            diss = tool.get("trans_diss")
            if diss:
                desc += f"，距离约为{diss}"
            notes = tool.get("trans_notes")
            if notes:
                desc += f"，其他备注{notes}"
            tool_parts.append(desc)
        if tool_parts:
            line += "，" + "；".join(tool_parts)
            lines.append(line)
    return lines


def format_hotels_info(scenic_name: str, hotels: List) -> List[str]:
    """生成酒店信息，每个酒店一条描述"""
    lines = []
    for hotel_node in hotels:
        hotel = dict(hotel_node)
        hotel_name = hotel.get("name")
        if not hotel_name:
            continue
        parts = [f"景点【{scenic_name}】附近存在旅店【{hotel_name}】"]
        location = hotel.get("location")
        if location:
            parts.append(f"旅店位于【{location}】")
        nearby = hotel.get("nearby")
        if nearby:
            parts.append(f"靠近景点包括【{nearby}】")
        price_range = hotel.get("price_range")
        if price_range:
            parts.append(f"其价格区间为【{price_range}】")
        lines.append("，".join(parts) + "。")
    return lines


# ==================== 主流程 ====================
def main():
    # 初始化数据库连接
    db = Neo4jConnection(NEO4J_CONFIG)

    # 获取总景点数，计算页数
    total = get_scenic_count(db)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    logging.info(f"总 Main_Scenic 数: {total}, 分页数: {total_pages}")

    # 打开输出文件（覆盖模式）
    with open(os.path.join(rag_file_path, SCENIC_INFO_FILE), 'w', encoding='utf-8') as f_scenic, \
         open(os.path.join(rag_file_path, TOOLS_INFO_FILE), 'w', encoding='utf-8') as f_tools, \
         open(os.path.join(rag_file_path, HOTELS_INFO_FILE), 'w', encoding='utf-8') as f_hotels:

        # 用于全局去重的from_to名称集合
        processed_from_to: Set[str] = set()

        for page in range(total_pages):
            skip = page * PAGE_SIZE
            logging.info(f"处理第 {page+1}/{total_pages} 页 (skip={skip})")

            # 1. 获取当前页的 Main_Scenic 基本信息
            main_scenic_page = fetch_main_scenic_page(db, skip, PAGE_SIZE)
            if not main_scenic_page:
                continue

            # 收集当前页所有 Main_Scenic 的 name
            main_scenic_names = [node.get("name") for node, _ in main_scenic_page if node.get("name")]

            # 3. 批量查询 ScenicHotel 节点
            hotel_dict = fetch_hotel_for_main_scenic(db, main_scenic_names)

            # 4. 批量查询 Scenic_From_To 节点（已去重）
            from_to_list = fetch_from_to_for_main_scenic(db, main_scenic_names)

            # 5. 处理交通信息（全局去重）
            tools_lines = format_tools_info(from_to_list, processed_from_to)
            for line in tools_lines:
                f_tools.write(line + '\n')

            # 6. 遍历每个 Main_Scenic，写入景点基本信息和酒店信息
            for scenic_node, provincial_name in main_scenic_page:
                scenic_name = scenic_node.get("name", "")
                if not scenic_name:
                    continue

                # 2. 批量查询 Scenic 节点
                spopt_nodes = fetch_scenic_for_main_scenic(db, scenic_name)

                # 写入景点基本信息
                scenic_info = format_scenic_info(scenic_node, provincial_name, spopt_nodes)
                if scenic_info:
                    f_scenic.write(scenic_info + '\n')

                # 写入酒店信息（从 hotel_dict 中取出对应列表）
                hotels = hotel_dict.get(scenic_name, [])
                hotels_lines = format_hotels_info(scenic_name, hotels)
                for line in hotels_lines:
                    f_hotels.write(line + '\n')

    logging.info("所有数据写入完成。")


if __name__ == "__main__":
    main()