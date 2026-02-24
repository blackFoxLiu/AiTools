#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手整合数据脚本
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


# ==================== 批量查询（每页一次） ====================
def fetch_scenic_page_data(db: Neo4jConnection, skip: int, limit: int):
    """
    执行复杂查询，返回一页景点的完整数据（景点、所属城市、酒店、from_to及其工具）
    返回一个Cursor，每条记录包含：
        - s: 景点节点
        - main_city: 所属城市名称（可能为None）
        - hotels: 酒店节点列表
        - from_to_list: 列表，每个元素为 {from_to: 节点, tools: 工具节点列表}
    """
    cql = """
        MATCH (s:Scenic)
        OPTIONAL MATCH (s)-[:belong_to]->(m:Main_City)
        WITH s, m.name AS main_city
        RETURN s, main_city,
               [(s)-[:exists]->(h:ScenicHotel) | h] AS hotels,
               [(s)-[:from_to]-(f:Scenic_From_To) | 
                 {
                   from_to: f,
                   tools: [(f)-[:tools]-(t:TravelTool) | t]
                 }
               ] AS from_to_list
        ORDER BY s.name   // 按名称排序保证分页稳定
        SKIP $skip
        LIMIT $limit
    """
    return db.run_query(cql, skip=skip, limit=limit)


def get_scenic_count(db: Neo4jConnection) -> int:
    """获取景点总数"""
    result = db.run_query("MATCH (s:Scenic) RETURN count(s) AS total")
    return result.evaluate()


# ==================== 信息生成函数（基于批量查询的数据） ====================
def format_scenic_info(scenic_node, main_city: Optional[str]) -> str:
    """生成景点基本信息描述"""
    props = dict(scenic_node)
    name = props.get("name")
    if not name:
        return ""

    parts = []
    if main_city:
        parts.append(f"{name}位于{main_city}")

    season = props.get("season")
    suit_months = props.get("suit_months_range")
    tendency1 = props.get("tendency_label_1")
    tendency2 = props.get("tendency_label_2")

    if season:
        parts.append(f"适合季节为{season}")
    if suit_months:
        parts.append(f"适合月份为{suit_months}")
    if tendency1:
        parts.append(f"小红书上推荐景点倾向标签为{tendency1}")
    if tendency2:
        parts.append(f"小红书上推荐景点倾向次级标签为{tendency2}")

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
    logging.info(f"总景点数: {total}, 分页数: {total_pages}")

    # 打开输出文件（覆盖模式）
    with open(os.path.join(rag_file_path, SCENIC_INFO_FILE), 'w', encoding='utf-8') as f_scenic, \
         open(os.path.join(rag_file_path, TOOLS_INFO_FILE), 'w', encoding='utf-8') as f_tools, \
         open(os.path.join(rag_file_path, HOTELS_INFO_FILE), 'w', encoding='utf-8') as f_hotels:

        # 用于全局去重的from_to名称集合
        processed_from_to: Set[str] = set()

        for page in range(total_pages):
            skip = page * PAGE_SIZE
            logging.info(f"处理第 {page+1}/{total_pages} 页 (skip={skip})")

            # 执行批量查询，返回游标（流式处理）
            cursor = fetch_scenic_page_data(db, skip, PAGE_SIZE)

            for record in cursor:
                scenic_node = record["s"]
                main_city = record["main_city"]
                hotels = record["hotels"]          # 列表
                from_to_list = record["from_to_list"]  # 列表

                scenic_name = dict(scenic_node).get("name", "")

                # 写入景点基本信息
                scenic_info = format_scenic_info(scenic_node, main_city)
                if scenic_info:
                    f_scenic.write(scenic_info + '\n')

                # 写入交通信息（去重）
                tools_lines = format_tools_info(from_to_list, processed_from_to)
                for line in tools_lines:
                    f_tools.write(line + '\n')

                # 写入酒店信息
                hotels_lines = format_hotels_info(scenic_name, hotels)
                for line in hotels_lines:
                    f_hotels.write(line + '\n')

    logging.info("所有数据写入完成。")


if __name__ == "__main__":
    main()