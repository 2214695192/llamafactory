from typing import List, Dict, Any
import json
import re
import requests
from dataclasses import dataclass
from config import KNOWLEDGE_GRAPH_CONFIG
import time
from tqdm import tqdm

@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    subject_type: str  # 主体类型
    object_type: str   # 客体类型

class KnowledgeExtractor:
    def __init__(self, api_key: str, model_name: str = "Pro/deepseek-ai/DeepSeek-V3"):
        """Initialize knowledge extractor
        
        Args:
            api_key: API key
            model_name: Model name
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"✅ Initialized knowledge extractor with model: {model_name}")
        print(f"API URL: {self.api_url}")
    
    def _normalize_text(self, text: str) -> str:
        """规范化文本，处理特殊字符和格式"""
        # 替换中文引号为英文引号
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _chunk_text(self, text: str, max_chunk_size: int = 5000) -> List[str]:
        """将长文本分割成较小的块"""
        # 按句子分割
        sentences = re.split(r'([。！？!?])', text)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

    def _generate_prompt(self, text: str) -> str:
        """Generate prompt for knowledge extraction"""
        return f"""Please extract knowledge triples from the following medical text. Each triple should contain subject, relation, object, and their respective types.

Requirements:
1. Subject and object must be medical concepts (diseases, symptoms, drugs, treatments, etc.)
2. Relation must be one of the predefined types
3. Each triple must contain 5 elements: subject, subject_type, relation, object, object_type
4. Output must be in JSON array format, each element being an object with 5 fields
5. All text must be in English, including entity names and types

Valid node types (use EXACTLY these types, no variations allowed):
{node_types}

Predefined relation types (use EXACTLY these types, no variations allowed):
{relationship_types}

Example input:
"Hypertension is a common chronic disease, often accompanied by headache and dizziness. Antihypertensive drugs like Captopril can effectively control blood pressure, but attention should be paid to interactions with certain medications."

Example output:
[
    {{"subject": "Hypertension", "subject_type": "Disease", "relation": "has_symptom", "object": "Headache", "object_type": "Symptom"}},
    {{"subject": "Hypertension", "subject_type": "Disease", "relation": "has_symptom", "object": "Dizziness", "object_type": "Symptom"}},
    {{"subject": "Hypertension", "subject_type": "Disease", "relation": "treated_by", "object": "Captopril", "object_type": "Medication"}},
    {{"subject": "Captopril", "subject_type": "Medication", "relation": "interacts_with", "object": "Other Medications", "object_type": "Medication"}}
]

Please extract knowledge triples from the following text:

{text}

Please ensure:
1. Extracted triples accurately reflect medical knowledge from the text
2. Subject and object types must be one of the valid node types listed above, used EXACTLY as shown
3. Relation types must be selected from the predefined list, used EXACTLY as shown
4. Output must be valid JSON format
5. All text should be in English
6. Do not make up or infer relationships that are not explicitly stated in the text
7. Focus on extracting clear, factual relationships
"""

    def _extract_triples_from_text(self, text: str) -> List[Triple]:
        """Extract knowledge triples from text"""
        print("\n=== 开始三元组提取 ===")
        prompt = self._generate_prompt(text)
        
        messages = [
            {
                "role": "system", 
                "content": """You are a medical knowledge graph expert. Your task is to extract accurate knowledge triples from medical text.
                You must strictly follow the required JSON format and ensure all fields are correctly filled.
                If no valid triples can be extracted from the text, return an empty array [].
                All output must be in English."""
            },
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"发送API请求... (第 {attempt + 1}/{max_retries} 次尝试)")
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 3000,  # 增加token数量
                    "temperature": 0.1,
                    "top_p": 0.95,  # 略微提高采样概率
                    "top_k": 100,  # 增加采样范围
                    "frequency_penalty": 0.2,  # 降低频率惩罚
                    "n": 1,
                    "response_format": {"type": "text"}
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=180  # 增加超时时间到180秒
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                print(f"API响应内容:\n{content}")
                
                # Parse triples from the response
                return self._parse_triples(content)
                
            except requests.exceptions.Timeout:
                print(f"❌ API请求超时 (第 {attempt + 1}/{max_retries} 次尝试)")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                continue
                
            except requests.exceptions.RequestException as e:
                print(f"❌ API请求失败: {str(e)}")
                if hasattr(e.response, 'text'):
                    print(f"API响应内容: {e.response.text}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                continue
                
            except Exception as e:
                print(f"❌ 处理失败: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                continue
        
        print("❌ 所有重试都失败了")
        return []

    def extract_triples(self, text: str) -> List[Triple]:
        """Extract knowledge triples from text"""
        print("\n=== 开始提取知识三元组 ===")
        print(f"输入文本长度: {len(text)} 字符")
        
        # Normalize text
        text = self._normalize_text(text)
        print(f"规范化后文本长度: {len(text)} 字符")
        
        # Process text in chunks
        chunks = self._chunk_text(text)
        total_chunks = len(chunks)
        print(f"文本分块数量: {total_chunks} 个")
        
        all_triples = []
        with tqdm(total=total_chunks, desc="正在处理文本块", unit="块") as pbar:
            for chunk in chunks:
                chunk_triples = self._extract_triples_from_text(chunk)
                all_triples.extend(chunk_triples)
                pbar.update(1)
                pbar.set_postfix({"三元组数": len(chunk_triples)})
        
        # Remove duplicates
        unique_triples = self._deduplicate_triples(all_triples)
        print(f"\n提取出的唯一三元组数量: {len(unique_triples)} 个")
        return unique_triples
    
    def _deduplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """去除重复的三元组"""
        seen = set()
        unique_triples = []
        
        for triple in triples:
            key = (triple.subject, triple.relation, triple.object)
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        return unique_triples

    def _normalize_relation(self, relation: str) -> str:
        """Normalize relation type to match configuration format"""
        # 关系类型映射表
        relation_mapping = {
            'prevents': 'prevented_by',
            'treats': 'treated_by',
            'diagnoses': 'diagnosed_by',
            'measures': 'measured_by',
            'indicates': 'indicated_for',
            'locates': 'located_in',
            'associates': 'associated_with',
            'classifies': 'classified_as',
            'has_symptom': 'has_symptom',  # 保持不变的关系
            'leads_to': 'leads_to',
            'part_of': 'part_of',
            'type_of': 'type_of',
            'stage_of': 'stage_of',
            'causes': 'causes',
            'contraindicates': 'contraindicates',
            'interacts_with': 'interacts_with'
        }
        
        # 转换为小写并移除下划线
        relation = relation.lower().replace('_', '')
        
        # 查找映射关系
        for key, value in relation_mapping.items():
            if relation == key.lower().replace('_', ''):
                return value
                
        # 如果没有找到映射，返回原始关系
        return relation

    def _parse_triples(self, output_text: str) -> List[Triple]:
        """Parse triples from model output"""
        print("\n=== 开始解析三元组 ===")
        print(f"原始输出文本:\n{output_text}")
        
        triples = []
        try:
            # Try direct JSON parsing
            if output_text.strip().startswith('['):
                data = json.loads(output_text)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and all(k in item for k in ['subject', 'subject_type', 'relation', 'object', 'object_type']):
                            # 验证节点类型
                            if (item['subject_type'] not in KNOWLEDGE_GRAPH_CONFIG["node_types"] or
                                item['object_type'] not in KNOWLEDGE_GRAPH_CONFIG["node_types"]):
                                print(f"❌ 无效的节点类型: {item}")
                                continue
                                
                            # 验证关系类型
                            if item['relation'] not in KNOWLEDGE_GRAPH_CONFIG["relationship_types"]:
                                print(f"❌ 无效的关系类型: {item['relation']}")
                                continue
                                
                            triple = Triple(
                                subject=item['subject'],
                                subject_type=item['subject_type'],
                                relation=item['relation'],
                                object=item['object'],
                                object_type=item['object_type']
                            )
                            triples.append(triple)
                            print(f"✅ 成功解析三元组: {triple}")
            
            # Validate triples
            valid_triples = []
            for triple in triples:
                # 检查节点和关系类型是否在配置中
                if (triple.subject_type in KNOWLEDGE_GRAPH_CONFIG["node_types"] and
                    triple.object_type in KNOWLEDGE_GRAPH_CONFIG["node_types"] and
                    triple.relation in KNOWLEDGE_GRAPH_CONFIG["relationship_types"]):
                    # 检查关系约束
                    if triple.relation in KNOWLEDGE_GRAPH_CONFIG["relation_constraints"]:
                        constraints = KNOWLEDGE_GRAPH_CONFIG["relation_constraints"][triple.relation]
                        if ("allowed_subjects" in constraints and
                            triple.subject_type not in constraints["allowed_subjects"]):
                            print(f"❌ 主体类型不满足关系约束: {triple}")
                            continue
                        if ("allowed_objects" in constraints and
                            triple.object_type not in constraints["allowed_objects"]):
                            print(f"❌ 客体类型不满足关系约束: {triple}")
                            continue
                    valid_triples.append(triple)
                else:
                    print(f"❌ 无效的三元组类型: {triple}")
                    print(f"   - 主体类型: {triple.subject_type} {'有效' if triple.subject_type in KNOWLEDGE_GRAPH_CONFIG['node_types'] else '无效'}")
                    print(f"   - 客体类型: {triple.object_type} {'有效' if triple.object_type in KNOWLEDGE_GRAPH_CONFIG['node_types'] else '无效'}")
                    print(f"   - 关系类型: {triple.relation} {'有效' if triple.relation in KNOWLEDGE_GRAPH_CONFIG['relationship_types'] else '无效'}")
            
            print(f"\n总共解析出 {len(valid_triples)} 个有效三元组")
            return valid_triples
            
        except Exception as e:
            print(f"❌ 解析三元组失败: {str(e)}")
            return [] 