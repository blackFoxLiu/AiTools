import json
import re
import tiktoken
from typing import List, Dict, Any, Generator, Optional
import os


class MedicalRAGProcessor:
    """
    医疗知识图谱RAG处理器（流式版本，输出JSON Lines格式）
    每个疾病生成三个独立片段的文本块，每个块存储为一个JSON对象
    """

    def __init__(self, max_tokens: int = 800, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_by_tokens(self, text: str) -> List[str]:
        """将文本分割为不超过max_tokens的块，尽量在句子边界处分割"""
        if self.count_tokens(text) <= self.max_tokens:
            return [text]

        sentences = re.split(r'(?<=[。!！?？\n])', text)
        chunks = []
        current_chunk = ""
        for sent in sentences:
            test_chunk = current_chunk + sent
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if self.count_tokens(sent) > self.max_tokens:
                    # 极长句子强制截断
                    chunks.append(sent[:self.max_tokens * 4])
                else:
                    current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def generate_desc_blocks(self, disease_name: str, disease_data: Dict[str, Any]) -> List[Dict]:
        """生成疾病描述（desc）的文本块，返回JSON对象列表"""
        desc = disease_data.get('desc', '')
        if desc:
            text = f"疾病：{disease_name}\n疾病描述：{desc}"
        else:
            text = f"疾病：{disease_name}\n疾病描述：暂无详细描述"
        chunks = self.split_by_tokens(text)
        blocks = []
        for idx, chunk_text in enumerate(chunks):
            blocks.append({
                "disease_name": disease_name,
                "chunk_type": "desc",
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text)
            })
        return blocks

    def generate_cause_blocks(self, disease_name: str, disease_data: Dict[str, Any]) -> List[Dict]:
        """生成病因（cause）的文本块，没有病因则返回空列表"""
        cause = disease_data.get('cause', '')
        if not cause:
            return []
        text = f"疾病：{disease_name}\n病因：{cause}"
        chunks = self.split_by_tokens(text)
        blocks = []
        for idx, chunk_text in enumerate(chunks):
            blocks.append({
                "disease_name": disease_name,
                "chunk_type": "cause",
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text)
            })
        return blocks

    def generate_other_blocks(self, disease_name: str, disease_data: Dict[str, Any]) -> List[Dict]:
        """生成除 desc 和 cause 之外的所有其他信息，返回JSON对象列表"""
        parts = [f"疾病：{disease_name}"]

        # 症状
        if symptoms := disease_data.get('symptom', []):
            parts.append(f"主要症状：{'、'.join(symptoms)}")
        # 并发症
        if acompany := disease_data.get('acompany', []):
            parts.append(f"可能并发症：{'、'.join(acompany)}")
        # 宜吃食物
        if do_eat := disease_data.get('do_eat', []):
            parts.append(f"宜吃食物：{'、'.join(do_eat)}")
        # 忌吃食物
        if not_eat := disease_data.get('not_eat', []):
            parts.append(f"忌吃食物：{'、'.join(not_eat)}")
        # 推荐食谱
        if recommand_eat := disease_data.get('recommand_eat', []):
            parts.append(f"推荐食谱：{'、'.join(recommand_eat)}")
        # 常用药品
        if common_drug := disease_data.get('common_drug', []):
            parts.append(f"常用药品：{'、'.join(common_drug)}")
        # 推荐药品
        if recommand_drug := disease_data.get('recommand_drug', []):
            parts.append(f"推荐药品：{'、'.join(recommand_drug)}")
        # 检查项目
        if checks := disease_data.get('check', []):
            parts.append(f"检查项目：{'、'.join(checks)}")
        # 就诊科室
        if depts := disease_data.get('cure_department', []):
            parts.append(f"就诊科室：{'、'.join(depts)}")
        # 药品厂商
        if drug_details := disease_data.get('drug_detail', []):
            producer_map = {}
            for detail in drug_details:
                if '(' in detail and ')' in detail:
                    producer = detail.split('(')[0].strip()
                    drug_name = detail.split('(')[1].replace(')', '').strip()
                else:
                    producer = detail
                    drug_name = detail
                producer_map.setdefault(producer, []).append(drug_name)
            producer_lines = [f"{p} 生产了：{'、'.join(d)}" for p, d in producer_map.items()]
            parts.append("药品厂商信息：" + "；".join(producer_lines))
        # 科室层级
        if (category := disease_data.get('category', [])) and len(category) >= 2:
            hier = [f"{category[i + 1]} 属于 {category[i]}" for i in range(len(category) - 1)]
            parts.append("科室层级：" + "；".join(hier))
        # 预防措施
        if prevent := disease_data.get('prevent', ''):
            parts.append(f"预防措施：{prevent}")
        # 治疗方式
        if cure_way := disease_data.get('cure_way', []):
            parts.append(f"治疗方式：{'、'.join(cure_way)}")
        # 治疗周期
        if cure_lasttime := disease_data.get('cure_lasttime', ''):
            parts.append(f"治疗周期：{cure_lasttime}")
        # 治愈率
        if cured_prob := disease_data.get('cured_prob', ''):
            parts.append(f"治愈率：{cured_prob}")
        # 易感人群
        if easy_get := disease_data.get('easy_get', ''):
            parts.append(f"易感人群：{easy_get}")
        # 传播途径
        if get_way := disease_data.get('get_way', ''):
            parts.append(f"传播途径：{get_way}")
        # 医保状态
        if yibao := disease_data.get('yibao_status', ''):
            parts.append(f"医保状态：{yibao}")

        if len(parts) == 1:  # 只有疾病名称，无其他信息
            return []

        full_text = "\n".join(parts)
        chunks = self.split_by_tokens(full_text)
        blocks = []
        for idx, chunk_text in enumerate(chunks):
            blocks.append({
                "disease_name": disease_name,
                "chunk_type": "other",
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text)
            })
        return blocks

    def process_disease(self, disease_data: Dict[str, Any]) -> List[Dict]:
        """处理单个疾病，返回该疾病生成的所有文本块（JSON对象列表）"""
        name = disease_data.get('name', '未知疾病')
        blocks = []
        blocks.extend(self.generate_desc_blocks(name, disease_data))
        blocks.extend(self.generate_cause_blocks(name, disease_data))
        blocks.extend(self.generate_other_blocks(name, disease_data))
        return blocks

    def process_file_streaming(self, input_file: str, output_file: str,
                               show_progress: bool = True) -> int:
        """
        流式处理大文件，输出JSON Lines格式
        返回生成的文本块总数
        """
        total_blocks = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        if show_progress:
            try:
                from tqdm import tqdm
                with open(input_file, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
                pbar = tqdm(total=total_lines, desc="Processing diseases")
            except ImportError:
                pbar = None
        else:
            pbar = None

        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    if pbar: pbar.update(1)
                    continue
                try:
                    disease = json.loads(line)
                except json.JSONDecodeError:
                    print(f"跳过无效JSON行 {line_num}")
                    if pbar: pbar.update(1)
                    continue

                blocks = self.process_disease(disease)
                for block in blocks:
                    outfile.write(json.dumps(block, ensure_ascii=False) + "\n")
                total_blocks += len(blocks)

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()
        return total_blocks

    @staticmethod
    def read_blocks_from_file(output_file: str) -> Generator[Dict, None, None]:
        """从输出文件中读取JSON块，返回生成器"""
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


if __name__ == "__main__":
    processor = MedicalRAGProcessor(max_tokens=800)
    input_path = "input/medical.json"
    output_path = "output/rag_blocks.jsonl"

    if os.path.exists(input_path):
        total = processor.process_file_streaming(input_path, output_path, show_progress=True)
        print(f"处理完成，共生成 {total} 个文本块，保存至 {output_path}")

        # 预览前3个块
        print("\n--- 预览前3个文本块 ---")
        for i, block in enumerate(processor.read_blocks_from_file(output_path)):
            if i >= 3:
                break
            print(json.dumps(block, ensure_ascii=False, indent=2))
            print("-" * 50)
    else:
        print(f"文件 {input_path} 不存在")