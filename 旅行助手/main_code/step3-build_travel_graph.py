#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手构建知识图谱脚本
"""

import json
import logging
import configparser

from py2neo import Graph, Node, Subgraph

# 尝试导入自定义验证函数，若失败则提供占位函数
try:
    # 假设这些自定义函数存在
    from utils.statistics_travel_info import get_travel_info
except ImportError:
    print("警告：未找到 statistics_travel_info，使用默认验证（始终通过）")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# 文件信息配置
itinerary_output_path = config['model_output']['itinerary_output_path']
neo4j_uri = config['neo4j']['neo4j_uri']
account = config['neo4j']['account']
password = config['neo4j']['password']
travel_analysis_output_path = config['model_output']['travel_analysis_output_path']


class MedicalGraph:
    """
    知识图谱构建器：从JSON数据创建旅游相关的Neo4j图数据库
    """

    def __init__(self,
                 neo4j_uri=neo4j_uri,
                 neo4j_auth=None):
        """
        初始化图数据库连接和路径
        :param json_path: JSON数据文件路径
        :param neo4j_uri: Neo4j连接URI
        :param neo4j_auth: 认证元组 (user, password)
        """
        # 配置数据路径（如果未提供则使用默认）
        self.data_path = itinerary_output_path

        # Neo4j连接（推荐使用环境变量或配置文件传递密码）
        self.g = Graph(neo4j_uri, auth=neo4j_auth or (account, password))

        # 从统计数据获取景点推荐信息
        self.travel_scenic_info = get_travel_info(travel_analysis_output_path)

        # 初始化数据容器（全部使用实例变量，避免类变量共享）
        self.main_city_set = set()                 # 所有省会名称（唯一）
        self.main_scenic_set = set()               # 所有景区名称（唯一）
        self.scenic_set = set()                    # 所有景点名称（唯一）
        self.scenic_ft_set = set()                 # 出发地-目的地对（格式 "A=B"）
        self.scenic_ft_mode_set = set()            # 出行方式对（格式 "A=使用=M=到=B"）

        # 关系列表（去重后使用集合，最后转换为列表用于批量创建）
        self.rels_scenic_from_to = set()        # (出发地, 出行对)
        self.rels_ft_mode = set()                # (出行对, 出行方式对)
        self.rels_m_s = set()                     # (景区, 景点)
        self.rels_m_p = set()                     # (景区, 省会)
        self.rels_hotel_scenic = set()             # (景点, 酒店)

        # 酒店节点缓存，避免重复创建
        self.hotel_nodes = {}                     # name -> Node

    def _parse_hotels(self, data_dict):
        """解析酒店信息，创建酒店节点并记录景点-酒店关系"""
        hotels = data_dict.get("hotels", [])
        for hotel_data in hotels:
            hotel_name = hotel_data.get("hotel_name", "").strip()
            if not hotel_name:
                continue

            location = hotel_data.get("location", [])
            price_range = hotel_data.get("price_range", [])
            nearby_attractions = hotel_data.get("nearby_attractions", [])

            judge_exist = False
            # 记录酒店与附近景点的关系
            for scenic in nearby_attractions:
                if scenic.strip() and scenic in self.main_scenic_set:
                    self.rels_hotel_scenic.add((scenic.strip(), hotel_name))
                    self.main_scenic_set.add(scenic.strip())  # 确保景点存在
                    judge_exist = True

            # 如果酒店节点尚未创建，则缓存（稍后批量创建）
            if hotel_name not in self.hotel_nodes and judge_exist:
                props = {
                    "name": hotel_name,
                    "location": location,
                    "price_range": price_range,
                    "nearby": ",".join(nearby_attractions)
                }
                self.hotel_nodes[hotel_name] = Node("ScenicHotel", **props)

    def _parse_transportation(self, data_dict):
        """解析交通信息，创建出行相关节点和关系"""
        main_city = data_dict.get("provincial", "").strip()
        if main_city:
            self.main_city_set.add(main_city)

        transportation_list = data_dict.get("transportation", [])
        for trans in transportation_list:
            departure = trans.get("departure", "").strip()
            destination = trans.get("destination", "").strip()

            if departure not in self.travel_scenic_info and \
                    destination not in self.travel_scenic_info:
                continue
            trans_modes = trans.get("transportation_mode", [])

            if not (departure and destination and trans_modes):
                continue

            # 景点-主要城市关系
            self.rels_m_p.add((departure, main_city))
            self.rels_m_p.add((destination, main_city))
            self.main_scenic_set.update([departure, destination])

            # 出发地-目的地对
            scenic_ft = f"{departure}={destination}"
            self.scenic_ft_set.add(scenic_ft)

            # 景点与出行对的关系
            self.rels_scenic_from_to.add((departure, scenic_ft))
            self.rels_scenic_from_to.add((destination, scenic_ft))

            # 构建出行方式节点名称
            trans_modes_str = ','.join(trans_modes)
            scenic_ft_mode = f"【{departure}】使用{trans_modes_str}达到【{destination}】"
            self.scenic_ft_mode_set.add(scenic_ft_mode)

            # 出行方式与出行对的关系
            self.rels_ft_mode.add((scenic_ft, scenic_ft_mode))

            # 获取出行方式的其他属性
            trans_cost = trans.get("transportation_cost", "")
            trans_time = trans.get("transportation_time", "")
            trans_distance = trans.get("transportation_diss", "")
            trans_notes = trans.get("notes", "")

            # 创建出行方式节点（稍后批量创建）
            # 临时存储节点属性，稍后统一创建节点
            # 这里可以使用一个字典缓存，避免重复创建
            # 为了方便，在创建节点阶段直接使用Node，但为了批量，我们先记录属性
            # 简单起见，我们创建节点后立即添加到一个集合中，但这样会破坏批量性。
            # 改进：使用一个字典来暂存节点属性，然后在create_graphnodes中批量创建。
            # 由于这里需要记录属性，我们使用一个列表暂存节点。
            # 为了简化，我们在解析过程中直接创建节点并放入列表，但这样会增加内存并可能重复。
            # 更好的方法是收集所有节点属性，最后批量创建。
            # 此处我们采用在解析过程中创建Node对象，并存储到列表中，最后通过subgraph批量创建。
            # 注意：重复节点会多次创建，因此需要去重。
            # 我们将出行方式节点也缓存起来。
            node_key = ("TravelTool", scenic_ft_mode)
            props = {
                "name": scenic_ft_mode,
                "trans_cost": trans_cost,
                "trans_time": trans_time,
                "trans_diss": trans_distance,
                "trans_notes": trans_notes
            }
            # 临时存储，但稍后创建节点时仍可能重复，因为不同data_dict可能产生相同scenic_ft_mode。
            # 所以使用一个字典去重。
            self._travel_tool_nodes_cache[node_key] = Node("TravelTool", **props)

    def read_nodes(self):
        """
        读取JSON文件，解析所有节点和关系数据，存储到实例变量中
        """
        # 清空已有数据（如果重复调用）
        self.scenic_set.clear()
        self.main_city_set.clear()
        self.main_scenic_set.clear()
        self.scenic_ft_set.clear()
        self.scenic_ft_mode_set.clear()
        self.rels_scenic_from_to.clear()
        self.rels_ft_mode.clear()
        self.rels_m_s.clear()
        self.rels_m_p.clear()
        self.rels_hotel_scenic.clear()
        self.hotel_nodes.clear()
        self._travel_tool_nodes_cache = {}  # (label, name) -> Node

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败: {e}")
            raise

        for scenic_name in self.travel_scenic_info:
            self.main_scenic_set.add(scenic_name)
            scenic_info_dict = self.travel_scenic_info.get(scenic_name, "")
            provincial = scenic_info_dict.get("provincial", "")
            if len(provincial) != 0:
                self.rels_m_p.add((scenic_name, provincial))
                self.main_city_set.add(provincial)
            main_scenic = scenic_info_dict.get("main_scenic", "")

            self.rels_m_s.add((scenic_name, main_scenic))
            self.scenic_set.add(scenic_name)

        for record in data:
            self._parse_hotels(record)
            self._parse_transportation(record)

    # 将集合转换为列表以便后续使用
    # （实际上在批量创建时会直接使用集合迭代）
    def _batch_create_nodes(self, label, nodes_iter, properties_func=None):
        """
        批量创建节点，使用UNWIND提高性能
        :param label: 节点标签
        :param nodes_iter: 节点名称的可迭代对象
        :param properties_func: 可选函数，接收节点名称返回额外属性字典
        """
        if not nodes_iter:
            return
        # 构建参数列表
        params = []
        for name in nodes_iter:
            node_data = {"name": name}
            if properties_func:
                node_data.update(properties_func(name))
            params.append(node_data)

        # 使用UNWIND批量创建
        query = f"""
        UNWIND $params AS row
        CREATE (n:{label})
        SET n = row
        RETURN count(n)
        """
        try:
            self.g.run(query, parameters={"params": params}).data()
            logger.info(f"批量创建 {label} 节点 {len(params)} 个")
        except Exception as e:
            logger.error(f"批量创建节点 {label} 失败: {e}")
            # 可以尝试逐个创建作为回退，但这里简单抛出
            raise

    def _batch_create_relationships(self, rels, start_label, end_label, rel_type, rel_name):
        """
        批量创建关系，使用UNWIND
        :param rels: 关系三元组集合，每个元素为 (start_name, end_name)
        :param start_label: 起始节点标签
        :param end_label: 结束节点标签
        :param rel_type: 关系类型
        :param rel_name: 关系属性name
        """
        if not rels:
            return
        # 转换为列表
        rels_list = list(rels)
        query = f"""
        UNWIND $pairs AS pair
        MATCH (a:{start_label} {{name: pair[0]}})
        MATCH (b:{end_label} {{name: pair[1]}})
        CREATE (a)-[r:{rel_type} {{name: $rel_name}}]->(b)
        RETURN count(r)
        """
        try:
            self.g.run(query, parameters={"pairs": rels_list, "rel_name": rel_name}).data()
            logger.info(f"批量创建关系 {rel_type} 共 {len(rels_list)} 条")
        except Exception as e:
            logger.error(f"批量创建关系 {rel_type} 失败: {e}")
            raise

    def create_graphnodes(self):
        """创建所有节点（除酒店节点已在解析时创建）"""
        self.read_nodes()

        # 创建主要城市节点
        self._batch_create_nodes("Provincial", self.main_city_set)

        self._batch_create_nodes("Scenic", self.scenic_set)

        # 创建景点节点，附加推荐信息
        def main_scenic_props(name):
            info = self.travel_scenic_info.get(name, {})
            return {
                "season": info.get("season", ""),
                "suit_months_range": info.get("suit_months_range", ""),
                "other_recommend": info.get("other_recommend", ""),
                "tendency_label_1": info.get("tendency_label_1", ""),
                "tendency_label_2": info.get("tendency_label_2", "")
            }
        # 创建主要景点
        self._batch_create_nodes("Main_Scenic", self.main_scenic_set, properties_func=main_scenic_props)

        # 创建出行对节点 (Scenic_From_To)
        self._batch_create_nodes("Scenic_From_To", self.scenic_ft_set)

        # 创建出行方式节点 (TravelTool) - 使用缓存的节点对象
        if self._travel_tool_nodes_cache:
            # 从缓存中提取节点列表
            tool_nodes = list(self._travel_tool_nodes_cache.values())
            # 去重（按name去重）
            unique_tools = {}
            for node in tool_nodes:
                name = node["name"]
                if name not in unique_tools:
                    unique_tools[name] = node
            # 批量创建：使用Subgraph一次性提交
            subgraph = Subgraph(list(unique_tools.values()))
            self.g.create(subgraph)
            logger.info(f"批量创建 TravelTool 节点 {len(unique_tools)} 个")

        # 创建酒店节点 (ScenicHotel) - 使用缓存的节点对象
        if self.hotel_nodes:
            subgraph = Subgraph(list(self.hotel_nodes.values()))
            self.g.create(subgraph)
            logger.info(f"批量创建 ScenicHotel 节点 {len(self.hotel_nodes)} 个")

    def create_graphrels(self):
        """创建所有关系"""
        self._batch_create_relationships(self.rels_scenic_from_to, "Main_Scenic", "Scenic_From_To", "from_to", "关联")
        self._batch_create_relationships(self.rels_ft_mode, "Scenic_From_To", "TravelTool", "tools", "可选工具")
        self._batch_create_relationships(self.rels_m_s, "Scenic", "Main_Scenic", "include", "包含")
        self._batch_create_relationships(self.rels_m_p, "Main_Scenic", "Provincial", "belong_to", "属于")
        self._batch_create_relationships(self.rels_hotel_scenic, "Main_Scenic", "ScenicHotel", "exists", "存在")


if __name__ == '__main__':
    handler = MedicalGraph()
    handler.create_graphnodes()
    handler.create_graphrels()