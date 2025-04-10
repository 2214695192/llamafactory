from typing import Dict, List, Any
import re

class MarkdownProcessor:
    def __init__(self):
        self.section_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^[-*]\s+(.+)$', re.MULTILINE)
        
    def process_markdown(self, md_content: str) -> Dict[str, Any]:
        """处理 Markdown 内容，提取结构化信息"""
        sections = self._extract_sections(md_content)
        processed_data = {}
        
        for section in sections:
            title = section['title']
            content = section['content']
            
            # 提取列表项
            items = self._extract_list_items(content)
            if items:
                processed_data[title] = items
            else:
                processed_data[title] = content.strip()
                
        return processed_data
    
    def _extract_sections(self, md_content: str) -> List[Dict[str, Any]]:
        """提取 Markdown 中的章节"""
        sections = []
        current_section = None
        current_content = []
        
        for line in md_content.split('\n'):
            if self.section_pattern.match(line):
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content)
                    })
                current_section = self.section_pattern.match(line).group(1)
                current_content = []
            elif current_section:
                current_content.append(line)
                
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content)
            })
            
        return sections
    
    def _extract_list_items(self, content: str) -> List[str]:
        """提取列表项"""
        items = self.list_pattern.findall(content)
        return [item.strip() for item in items] 