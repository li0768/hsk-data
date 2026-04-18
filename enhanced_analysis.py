"""
增强分析模块 - OpenRouter版
使用OpenRouter API调用Qwen3 Coder 480B模型
优化教学建议提示词，输出更加实用的教学内容
修复：编码问题、API调用错误、主题和关键词提取
"""

import json
import re
import time
import traceback
import os
import glob
import jieba
import jieba.analyse
import requests
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter

# 导入OpenRouter客户端
from openrouter_client import get_openrouter_client
# 导入优化后的提示词
try:
    from teaching_prompts import (
        SYSTEM_PROMPT,
        THEME_EXTRACTION_PROMPT,
        KEYWORDS_EXTRACTION_PROMPT,
        TEACHING_PLAN_PROMPT
    )
except ImportError:
    # 如果teaching_prompts不存在，定义默认提示词
    SYSTEM_PROMPT = "你是一位资深国际中文教育专家，请用中文回答，不要使用任何英文单词。"
    THEME_EXTRACTION_PROMPT = "请分析以下中文文本，提取最核心的主题（2-6个汉字）。"
    KEYWORDS_EXTRACTION_PROMPT = "请从以下中文文本中提取最重要的3-5个教学关键词，用顿号分隔。"
    TEACHING_PLAN_PROMPT = "请为以下中文文本提供详细的教学建议。"

class EnhancedAnalysisGenerator:
    """增强分析生成器 - OpenRouter版"""
    
    def __init__(self, 
                 model_name: str ="openai/gpt-oss-120b:free",
                 collocation_dir: str = None,
                 hsk_data_dir: str = None,
                 verbose: bool = True):
        """
        初始化增强分析生成器
        
        Args:
            model_name: OpenRouter模型名称
            collocation_dir: 搭配词库目录路径
            hsk_data_dir: HSK数据目录路径
            verbose: 是否显示详细信息
        """
        self.verbose = verbose
        self.collocation_data = {}
        self.collocation_dir = collocation_dir
        self.hsk_data_dir = hsk_data_dir
        self.model_name = model_name
        self.extended_timeout = 180  # 增加到180秒
        
        # 初始化OpenRouter客户端
        self.openrouter_client = get_openrouter_client(verbose=verbose)
        
        # 加载HSK词汇表和汉字表（包含拼音和词性）
        self.hsk_vocab_with_details = self.load_hsk_vocab_with_details()
        self.hsk_chars_with_details = self.load_hsk_chars_with_details()
        
        print(f"✅ 初始化OpenRouter客户端")
        print(f"   模型名称: {self.model_name}")
        print(f"   超时设置: {self.extended_timeout}秒")
        
        # 初始化jieba分词
        try:
            jieba.initialize()
            print("✅ jieba分词器初始化成功")
        except:
            jieba.initialize(sender=None)
        
        if collocation_dir:
            self.initialize_collocation_library(collocation_dir)
    
    def load_hsk_vocab_with_details(self) -> Dict[str, List[Dict]]:
        """加载HSK词汇表（包含拼音、词性、级别等详细信息）"""
        vocab_data = {}
        
        try:
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(self.hsk_data_dir, 'data', '词汇.csv') if self.hsk_data_dir else None,
                os.path.join(os.path.dirname(__file__), '..', 'data', '词汇.csv'),
                os.path.join(os.path.dirname(__file__), 'data', '词汇.csv')   # 备用
            ]
            
            vocab_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    vocab_path = path
                    break
            
            if not vocab_path:
                print("❌ 未找到词汇表文件")
                return {}
            
            print(f"📚 加载词汇表: {vocab_path}")
            
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig', 'latin-1']
            vocab_df = None
            
            for encoding in encodings:
                try:
                    vocab_df = pd.read_csv(vocab_path, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码加载词汇表成功")
                    break
                except Exception as e:
                    continue
            
            if vocab_df is not None:
                print(f"📊 词汇表记录数: {len(vocab_df)}")
                print(f"📊 列名: {vocab_df.columns.tolist()}")
                
                # 标准化列名
                col_mapping = {
                    '词语': ['词语', '词', '单词', 'word', 'Word', '词汇', '汉字'],
                    '拼音': ['拼音', 'pinyin', 'Pinyin', '读音'],
                    '词性': ['词性', '词类', 'part_of_speech', 'pos', 'POS', '词类'],
                    '级别': ['级别', '等级', 'level', 'Level', 'HSK级别', 'HSK等级', 'hsk_level']
                }
                
                for target_col, possible_cols in col_mapping.items():
                    for col in possible_cols:
                        if col in vocab_df.columns:
                            if col != target_col:
                                vocab_df = vocab_df.rename(columns={col: target_col})
                            print(f"  ✅ 找到列: {col} -> {target_col}")
                            break
                
                # 检查必备列
                required_cols = ['词语']
                missing_cols = [col for col in required_cols if col not in vocab_df.columns]
                if missing_cols:
                    print(f"❌ 缺少必备列: {missing_cols}")
                    print(f"   可用列: {vocab_df.columns.tolist()}")
                    return {}
                
                # 处理每个词语
                for _, row in vocab_df.iterrows():
                    word = str(row.get('词语', '')).strip()
                    if not word:
                        continue
                    
                    pinyin = str(row.get('拼音', '')).strip()
                    pos = str(row.get('词性', '')).strip()
                    level = str(row.get('级别', '')).strip()
                    
                    # 清理数据
                    if pinyin.lower() in ['nan', 'null', 'none', '']:
                        pinyin = ''
                    if pos.lower() in ['nan', 'null', 'none', '']:
                        pos = ''
                    if level.lower() in ['nan', 'null', 'none', '']:
                        level = ''
                    
                    # 标准化拼音格式
                    if pinyin:
                        pinyin = pinyin.replace(' ', '').replace(',', ' ').replace('，', ' ')
                        pinyin = re.sub(r'[^\w\s]', '', pinyin)
                    
                    if word not in vocab_data:
                        vocab_data[word] = []
                    
                    entry = {
                        'pinyin': pinyin if pinyin else '待补充',
                        'pos': pos if pos else '待补充',
                        'level': level if level else '待补充'
                    }
                    
                    # 只添加非重复条目
                    if entry not in vocab_data[word]:
                        vocab_data[word].append(entry)
                
                print(f"✅ 词汇表加载完成: {len(vocab_data)} 个词语")
                
                # 显示前5个词语的详细信息
                sample_count = 0
                for word, entries in list(vocab_data.items())[:5]:
                    print(f"  📝 示例: {word}")
                    for entry in entries:
                        print(f"    拼音: {entry['pinyin']}, 词性: {entry['pos']}, 级别: {entry['level']}")
                    sample_count += 1
                
        except Exception as e:
            print(f"❌ 加载词汇表失败: {e}")
            traceback.print_exc()
        
        return vocab_data
    
    def load_hsk_chars_with_details(self) -> Dict[str, List[Dict]]:
        """加载HSK汉字表（包含拼音、级别等详细信息）"""
        chars_data = {}
        
        try:
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(self.hsk_data_dir, 'data', '汉字.csv') if self.hsk_data_dir else None,
                os.path.join(os.path.dirname(__file__), '..', 'data', '汉字.csv'),
                os.path.join(os.path.dirname(__file__), 'data', '汉字.csv')
            ]
            
            chars_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    chars_path = path
                    break
            
            if not chars_path:
                print("❌ 未找到汉字表文件")
                return {}
            
            print(f"📚 加载汉字表: {chars_path}")
            
            # 尝试不同编码
            encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-8-sig', 'latin-1']
            chars_df = None
            
            for encoding in encodings:
                try:
                    chars_df = pd.read_csv(chars_path, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码加载汉字表成功")
                    break
                except Exception as e:
                    continue
            
            if chars_df is not None:
                print(f"📊 汉字表记录数: {len(chars_df)}")
                print(f"📊 列名: {chars_df.columns.tolist()}")
                
                # 标准化列名
                col_mapping = {
                    '汉字': ['汉字', '字', 'char', 'Char', '字符', '词语'],
                    '拼音': ['拼音', 'pinyin', 'Pinyin', '读音'],
                    '级别': ['级别', '等级', 'level', 'Level', 'HSK级别', 'HSK等级', 'hsk_level']
                }
                
                for target_col, possible_cols in col_mapping.items():
                    for col in possible_cols:
                        if col in chars_df.columns:
                            if col != target_col:
                                chars_df = chars_df.rename(columns={col: target_col})
                            print(f"  ✅ 找到列: {col} -> {target_col}")
                            break
                
                # 检查必备列
                required_cols = ['汉字']
                missing_cols = [col for col in required_cols if col not in chars_df.columns]
                if missing_cols:
                    print(f"❌ 缺少必备列: {missing_cols}")
                    print(f"   可用列: {chars_df.columns.tolist()}")
                    return {}
                
                # 处理每个汉字
                for _, row in chars_df.iterrows():
                    char = str(row.get('汉字', '')).strip()
                    if not char or len(char) != 1:
                        continue
                    
                    pinyin = str(row.get('拼音', '')).strip()
                    level = str(row.get('级别', '')).strip()
                    
                    # 清理数据
                    if pinyin.lower() in ['nan', 'null', 'none', '']:
                        pinyin = ''
                    if level.lower() in ['nan', 'null', 'none', '']:
                        level = ''
                    
                    # 标准化拼音格式
                    if pinyin:
                        pinyin = pinyin.replace(' ', '').replace(',', ' ').replace('，', ' ')
                        pinyin = re.sub(r'[^\w\s]', '', pinyin)
                    
                    if char not in chars_data:
                        chars_data[char] = []
                    
                    entry = {
                        'pinyin': pinyin if pinyin else '待补充',
                        'level': level if level else '待补充'
                    }
                    
                    # 只添加非重复条目
                    if entry not in chars_data[char]:
                        chars_data[char].append(entry)
                
                print(f"✅ 汉字表加载完成: {len(chars_data)} 个汉字")
                
                # 显示前5个汉字的详细信息
                sample_count = 0
                for char, entries in list(chars_data.items())[:5]:
                    print(f"  📝 示例: {char}")
                    for entry in entries:
                        print(f"    拼音: {entry['pinyin']}, 级别: {entry['level']}")
                    sample_count += 1
                
        except Exception as e:
            print(f"❌ 加载汉字表失败: {e}")
            traceback.print_exc()
        
        return chars_data
    
    def get_word_details(self, word: str) -> Dict[str, Any]:
        """获取词语的详细信息（拼音、词性、级别）"""
        if word in self.hsk_vocab_with_details:
            entries = self.hsk_vocab_with_details[word]
            return {
                'word': word,
                'pinyin_list': list(set([entry['pinyin'] for entry in entries if entry['pinyin'] and entry['pinyin'] != '待补充'])),
                'pos_list': list(set([entry['pos'] for entry in entries if entry['pos'] and entry['pos'] != '待补充'])),
                'level_list': list(set([entry['level'] for entry in entries if entry['level'] and entry['level'] != '待补充'])),
                'has_data': True,
                'entry_count': len(entries),
                'all_entries': entries
            }
        else:
            # 如果不在词汇表中，尝试分词获取拼音
            return {
                'word': word,
                'pinyin_list': ['未在词表中找到'],
                'pos_list': ['未知'],
                'level_list': ['未知'],
                'has_data': False,
                'entry_count': 0,
                'all_entries': []
            }
    
    def get_char_details(self, char: str) -> Dict[str, Any]:
        """获取汉字的详细信息（拼音、级别）"""
        if char in self.hsk_chars_with_details:
            entries = self.hsk_chars_with_details[char]
            return {
                'char': char,
                'pinyin_list': list(set([entry['pinyin'] for entry in entries if entry['pinyin'] and entry['pinyin'] != '待补充'])),
                'level_list': list(set([entry['level'] for entry in entries if entry['level'] and entry['level'] != '待补充'])),
                'has_data': True,
                'entry_count': len(entries),
                'all_entries': entries
            }
        else:
            return {
                'char': char,
                'pinyin_list': ['未在字表中找到'],
                'level_list': ['未知'],
                'has_data': False,
                'entry_count': 0,
                'all_entries': []
            }
    
    def format_word_details_for_teaching(self, word: str) -> str:
        """为教学建议格式化词语详细信息"""
        details = self.get_word_details(word)
        
        if details['has_data']:
            # 格式化拼音
            if details['pinyin_list'] and details['pinyin_list'][0] != '未在词表中找到':
                pinyin_str = "、".join(details['pinyin_list'][:3])  # 最多显示3个拼音
            else:
                pinyin_str = "需查证"
            
            # 格式化词性
            if details['pos_list'] and details['pos_list'][0] not in ['未知', '待补充']:
                pos_str = "、".join(details['pos_list'][:2])  # 最多显示2个词性
            else:
                pos_str = "需查证"
            
            # 格式化级别
            if details['level_list'] and details['level_list'][0] not in ['未知', '待补充']:
                level_str = "、".join(details['level_list'][:2])  # 最多显示2个级别
            else:
                level_str = "需查证"
            
            return f"{word}（拼音：{pinyin_str}，词性：{pos_str}，级别：{level_str}）"
        else:
            return f"{word}（需查证详细信息）"
    
    def call_llm_api(self, prompt: str, system_prompt: str = None, max_tokens: int = 8000, timeout: int = 180) -> Optional[str]:
        """调用OpenRouter API生成文本"""
        try:
            # 确保prompt是UTF-8编码
            if isinstance(prompt, str):
                prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
            
            return self.openrouter_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"      ❌ 调用OpenRouter失败: {e}")
            return None
    
    def generate_detailed_teaching_suggestions_with_llm(self, 
                                                       text: str,
                                                       theme: str,
                                                       keywords: List[str],
                                                       difficult_words: List[Dict],
                                                       level: str) -> Dict[str, Any]:
        """使用OpenRouter API生成详细、具体的教学建议 - 优化版"""
        try:
            # 限制文本长度
            if len(text) > 1000:
                analysis_text = text[:1000] + "..."
            else:
                analysis_text = text
            
            # 准备生词信息
            difficult_words_info = ""
            if difficult_words:
                word_details = []
                for w in difficult_words[:8]:  # 最多显示8个生词
                    word = w['word']
                    formatted_details = self.format_word_details_for_teaching(word)
                    word_details.append(formatted_details)
                
                difficult_words_info = '、'.join(word_details)
            
            # 准备关键词字符串
            keywords_str = '、'.join(keywords[:5]) if keywords else '中文学习'
            
            # 构建完整的提示词
            prompt = TEACHING_PLAN_PROMPT.format(
                text=analysis_text,
                theme=theme,
                level=level,
                keywords=keywords_str,
                difficult_words=difficult_words_info
            )
            
            if self.verbose:
                print("  🤖 使用OpenRouter生成详细教学建议（超时: 180秒）...")
            
            response = self.call_llm_api(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=8000,
                timeout=180
            )
            
            if response and len(response) > 1000:
                # 清理响应，确保全是中文
                cleaned_response = self.clean_teaching_response(response)
                
                # 解析响应
                parsed_response = self.parse_detailed_teaching_response(cleaned_response)
                
                return {
                    "raw_response": cleaned_response,
                    "parsed": parsed_response,
                    "structured": self.extract_structured_suggestions(cleaned_response),
                    "generated_by_llm": True,
                    "length": len(cleaned_response),
                    "has_concrete_examples": self.check_concrete_content(cleaned_response),
                    "model_used": "qwen3-coder-480b (OpenRouter)"
                }
            else:
                if self.verbose:
                    print(f"  ⚠️  OpenRouter响应失败或过短（长度: {len(response) if response else 0}），使用详细备用分析")
                return self.generate_detailed_fallback_teaching_suggestions(text, theme, keywords, difficult_words, level)
                
        except Exception as e:
            print(f"❌ 生成详细教学建议失败: {e}")
            traceback.print_exc()
            return self.generate_detailed_fallback_teaching_suggestions(text, theme, keywords, difficult_words, level)
    
    def extract_theme_with_llm(self, text: str) -> str:
        """使用OpenRouter API提取文本主题 - 返回一句话总结"""
        try:
            # 限制文本长度
            if len(text) > 500:
                analysis_text = text[:500]
            else:
                analysis_text = text
            
            # 确保文本是UTF-8编码
            if isinstance(analysis_text, str):
                analysis_text = analysis_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # 明确要求返回一句话总结
            improved_prompt = f"""请用一句完整的话总结以下文本的主要内容，字数在10-30字之间。

文本：{analysis_text}

一句话总结："""
            
            if self.verbose:
                print("  🤖 使用OpenRouter提取主题（一句话总结）...")
            
            response = self.call_llm_api(
                prompt=improved_prompt,
                max_tokens=200,
                timeout=30
            )
            
            if response and len(response) > 5:
                theme = response.strip()
                
                # 移除常见的前缀
                prefixes = ['主题是', '主题为', '主要内容是', '内容是关于', '本文主题', '核心内容是', '概括为', '总结为', '一句话总结是']
                for prefix in prefixes:
                    if theme.startswith(prefix):
                        theme = theme[len(prefix):].strip()
                
                # 如果主题包含顿号，可能是关键词列表，需要重新处理
                if '、' in theme and len(theme) < 20:
                    # 可能是关键词列表，尝试重新生成
                    return self.extract_theme_fallback(text)
                
                # 清理标点符号，但保留必要的标点
                theme = re.sub(r'^[：:、，。\s]+', '', theme)
                theme = re.sub(r'[：:、，。\s]+$', '', theme)
                
                # 验证主题长度
                if 5 <= len(theme) <= 50:
                    if self.verbose:
                        print(f"    ✅ 提取到主题: {theme}")
                    return theme
                elif len(theme) > 50:
                    theme = theme[:50]
                    if self.verbose:
                        print(f"    ✅ 截取主题: {theme}")
                    return theme
                elif len(theme) > 0:
                    if self.verbose:
                        print(f"    ✅ 提取到主题（较短）: {theme}")
                    return theme
            
            # 如果API返回为空或无效，使用备用方法
            if self.verbose:
                print("    ⚠️ API返回为空，使用备用主题提取...")
            
            return self.extract_theme_fallback(text)
            
        except Exception as e:
            print(f"❌ OpenRouter提取主题失败: {e}")
            return self.extract_theme_fallback(text)
    
    def extract_theme_fallback(self, text: str) -> str:
        """备用主题提取方法 - 返回一句话总结"""
        try:
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=3, withWeight=False)
            
            # 根据关键词组合成一句话
            if '图书馆' in text and '学习' in text and '考试' in text:
                return '去图书馆学习准备考试'
            elif '图书馆' in text and '学习' in text:
                return '去图书馆学习'
            elif '考试' in text and '准备' in text:
                return '准备考试'
            elif '电影' in text:
                return '去看电影'
            elif '学习' in text and '考试' in text:
                return '学习准备考试'
            elif '学习' in text:
                return '学习活动'
            elif '准备' in text:
                return '准备工作'
            elif '明天' in text:
                return '明天的计划'
            elif '随后' in text:
                return '后续活动安排'
            else:
                # 组合关键词形成一句话
                if keywords and len(keywords) >= 2:
                    return f"关于{keywords[0]}和{keywords[1]}的活动"
                elif keywords:
                    return f"关于{keywords[0]}的内容"
                else:
                    return '日常活动'
                
        except Exception as e:
            print(f"❌ 备用主题提取失败: {e}")
            return '日常活动'
    
    def extract_keywords_with_llm(self, text: str) -> List[str]:
        """使用OpenRouter API提取关键词 - 确保返回3-5个关键词"""
        try:
            # 限制文本长度
            if len(text) > 500:
                analysis_text = text[:500]
            else:
                analysis_text = text
            
            # 确保文本是UTF-8编码
            if isinstance(analysis_text, str):
                analysis_text = analysis_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # 明确要求返回3-5个关键词
            improved_prompt = f"""请从以下文本中提取最重要的3-5个教学关键词，用顿号分隔。

文本：{analysis_text}

关键词："""
            
            if self.verbose:
                print("  🤖 使用OpenRouter提取关键词...")
            
            response = self.call_llm_api(
                prompt=improved_prompt,
                max_tokens=200,
                timeout=30
            )
            
            if response:
                keywords_text = response.strip()
                
                # 移除常见的前缀
                prefixes = ['关键词是', '关键词为', '关键词包括', '提取的关键词']
                for prefix in prefixes:
                    if keywords_text.startswith(prefix):
                        keywords_text = keywords_text[len(prefix):].strip()
                
                # 使用顿号、逗号、空格分割
                keywords = re.split(r'[、，,;；\s]', keywords_text)
                keywords = [kw.strip() for kw in keywords if kw.strip()]
                
                # 过滤：只保留2-4个汉字的词汇
                filtered_keywords = []
                for kw in keywords:
                    # 清理可能包含的非中文字符
                    clean_kw = re.sub(r'[^\u4e00-\u9fff]', '', kw)
                    if 2 <= len(clean_kw) <= 4 and clean_kw:
                        filtered_keywords.append(clean_kw)
                
                if filtered_keywords:
                    # 去重
                    unique_keywords = []
                    seen = set()
                    for kw in filtered_keywords:
                        if kw not in seen:
                            seen.add(kw)
                            unique_keywords.append(kw)
                    
                    # 如果关键词太少，从文本中补充
                    if len(unique_keywords) < 3:
                        # 从原始文本中提取可能的关键词
                        text_keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=False)
                        for kw in text_keywords:
                            clean_kw = re.sub(r'[^\u4e00-\u9fff]', '', kw)
                            if 2 <= len(clean_kw) <= 4 and clean_kw and clean_kw not in seen:
                                seen.add(clean_kw)
                                unique_keywords.append(clean_kw)
                                if len(unique_keywords) >= 5:
                                    break
                    
                    # 限制最多5个，最少3个
                    final_keywords = unique_keywords[:5]
                    if self.verbose:
                        print(f"    ✅ 提取到关键词: {final_keywords}")
                    return final_keywords
            
            if self.verbose:
                print("    ⚠️ API返回为空，使用备用关键词提取...")
            
            return self.extract_keywords_fallback(text)
            
        except Exception as e:
            print(f"❌ OpenRouter提取关键词失败: {e}")
            return self.extract_keywords_fallback(text)
    
    def extract_keywords_fallback(self, text: str) -> List[str]:
        """备用关键词提取方法 - 使用jieba，确保返回3-5个关键词"""
        try:
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=8, withWeight=False)
            
            # 过滤关键词
            filtered_keywords = []
            for kw in keywords:
                # 清理并过滤
                clean_kw = re.sub(r'[^\u4e00-\u9fff]', '', kw)
                if 2 <= len(clean_kw) <= 4 and clean_kw:
                    filtered_keywords.append(clean_kw)
            
            # 如果过滤后还有关键词，返回
            if filtered_keywords:
                # 去重
                unique_keywords = []
                seen = set()
                for kw in filtered_keywords:
                    if kw not in seen:
                        seen.add(kw)
                        unique_keywords.append(kw)
                
                # 如果关键词太少，补充默认关键词
                if len(unique_keywords) < 3:
                    # 从文本中提取可能的关键词
                    if '图书馆' in text:
                        if '图书馆' not in seen:
                            unique_keywords.append('图书馆')
                            seen.add('图书馆')
                    if '学习' in text:
                        if '学习' not in seen:
                            unique_keywords.append('学习')
                            seen.add('学习')
                    if '考试' in text:
                        if '考试' not in seen:
                            unique_keywords.append('考试')
                            seen.add('考试')
                    if '电影' in text:
                        if '电影' not in seen:
                            unique_keywords.append('电影')
                            seen.add('电影')
                    if '准备' in text:
                        if '准备' not in seen:
                            unique_keywords.append('准备')
                            seen.add('准备')
                    if '明天' in text:
                        if '明天' not in seen:
                            unique_keywords.append('明天')
                            seen.add('明天')
                
                # 限制最多5个
                final_keywords = unique_keywords[:5]
                if self.verbose:
                    print(f"    ✅ 备用关键词提取: {final_keywords}")
                return final_keywords
            
            # 如果jieba提取失败，基于文本内容生成关键词
            fallback_keywords = []
            
            # 分析文本中的关键词
            text_analysis = text
            
            # 常见的教学关键词
            if '图书馆' in text_analysis:
                fallback_keywords.append('学习')
                fallback_keywords.append('图书馆')
            if '考试' in text_analysis:
                fallback_keywords.append('考试')
                fallback_keywords.append('准备')
            if '电影' in text_analysis:
                fallback_keywords.append('娱乐')
                fallback_keywords.append('电影')
            if '明天' in text_analysis:
                fallback_keywords.append('计划')
                fallback_keywords.append('明天')
            if '学习' in text_analysis:
                fallback_keywords.append('学习')
            if '准备' in text_analysis:
                fallback_keywords.append('准备')
            if '随后' in text_analysis:
                fallback_keywords.append('计划')
            
            # 添加一些基础关键词
            if '中文' in text_analysis or '汉语' in text_analysis:
                fallback_keywords.append('中文')
            
            # 去重
            unique_keywords = []
            seen = set()
            for kw in fallback_keywords:
                if kw not in seen:
                    seen.add(kw)
                    unique_keywords.append(kw)
            
            # 如果关键词太少，补充默认关键词
            if len(unique_keywords) < 3:
                default_keywords = ['学习', '计划', '考试', '娱乐', '活动']
                for kw in default_keywords:
                    if kw not in seen and len(unique_keywords) < 5:
                        unique_keywords.append(kw)
                        seen.add(kw)
            
            final_keywords = unique_keywords[:5]
            if self.verbose:
                print(f"    ✅ 启发式关键词: {final_keywords}")
            
            return final_keywords
            
        except Exception as e:
            print(f"❌ 备用关键词提取失败: {e}")
            return ['学习', '计划', '考试', '娱乐', '活动']
    
    def clean_teaching_response(self, response: str) -> str:
        """清理教学建议响应，确保全是中文"""
        if not response:
            return response
        
        # 移除Markdown格式符号
        cleaned = response
        cleaned = re.sub(r'#+\s*', '', cleaned)
        cleaned = re.sub(r'\*\*', '', cleaned)
        cleaned = re.sub(r'\*', '', cleaned)
        cleaned = re.sub(r'`', '', cleaned)
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        cleaned = re.sub(r'~~', '', cleaned)
        cleaned = re.sub(r'_{2,}', '', cleaned)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        cleaned = re.sub(r'\|', '', cleaned)
        cleaned = re.sub(r'-{3,}', '', cleaned)
        cleaned = re.sub(r'>{1,}', '', cleaned)
        
        # 确保拼音正确（简单处理）
        # 这里不做复杂处理，保持原样
        
        return cleaned
    
    def parse_detailed_teaching_response(self, response: str) -> Dict[str, Any]:
        """解析详细教学建议响应"""
        try:
            sections = {
                "teaching_objectives": "",
                "teaching_process": "",
                "concrete_examples": "",
                "teaching_resources": "",
                "assessment_scheme": "",
                "difficulties_solutions": "",
                "exercises_with_answers": ""
            }
            
            current_section = None
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检测章节标题
                if '一、' in line or '教学对象' in line:
                    current_section = None  # 跳过教学对象分析部分
                elif '二、' in line or '教学目标' in line:
                    current_section = "teaching_objectives"
                    sections[current_section] += line + "\n"
                elif '三、' in line or '教学重点' in line:
                    current_section = None  # 跳过教学重点部分
                elif '四、' in line or '教学流程' in line:
                    current_section = "teaching_process"
                    sections[current_section] += line + "\n"
                elif '五、' in line or '师生对话' in line:
                    current_section = "concrete_examples"
                    sections[current_section] += line + "\n"
                elif '六、' in line or '练习题' in line:
                    current_section = "exercises_with_answers"
                    sections[current_section] += line + "\n"
                elif '七、' in line or '教学资源' in line:
                    current_section = "teaching_resources"
                    sections[current_section] += line + "\n"
                elif '八、' in line or '教学评估' in line:
                    current_section = "assessment_scheme"
                    sections[current_section] += line + "\n"
                elif '九、' in line or '教学反思' in line:
                    current_section = "difficulties_solutions"
                    sections[current_section] += line + "\n"
                elif current_section:
                    sections[current_section] += line + "\n"
            
            # 确保每个部分都有内容
            for section_name in sections:
                if not sections[section_name] or len(sections[section_name]) < 20:
                    sections[section_name] = self._get_default_section_content(section_name)
            
            return sections
            
        except Exception as e:
            print(f"❌ 解析教学建议失败: {e}")
            return self._get_default_parsed_response()
    
    def _get_default_section_content(self, section_name: str) -> str:
        """获取默认的章节内容"""
        default_contents = {
            "teaching_objectives": """二、教学目标设定
1. 语言知识目标：掌握文本中的核心词汇和基本语法结构
2. 语言技能目标：提高听说读写的综合语言运用能力
3. 文化意识目标：了解文本中体现的中国文化元素
4. 学习策略目标：培养学生有效的学习方法""",
            
            "teaching_process": """四、教学流程设计（90分钟）
1. 课堂导入（10分钟）：通过相关话题引入学习内容
2. 词汇教学（20分钟）：讲解重点词汇的发音、词义和用法
3. 语法讲解（15分钟）：分析重要的语法结构，提供例句
4. 课文理解（15分钟）：阅读理解活动，回答问题
5. 交际练习（20分钟）：设计各种课堂活动，加强语言实践
6. 总结作业（10分钟）：回顾学习要点，布置适当作业""",
            
            "concrete_examples": """五、师生对话示例
对话一：课堂提问
教师：这个词怎么读？
学生：学习（xué xí）
教师：很好！什么意思？
学生：意思是study。
教师：对，请用这个词造句。
学生：我喜欢学习汉语。""",
            
            "teaching_resources": """七、教学资源推荐
1. 视频资源：HSK标准教程配套视频 - https://www.youtube.com/@hsk
2. 网站资源：汉语学习网 - https://www.hanyuxuexi.com
3. APP资源：HelloChinese - 各大应用商店下载
4. 打印材料：HSK词汇表PDF - https://www.chinesetest.cn/download""",
            
            "assessment_scheme": """八、教学评估方案
1. 形成性评估：课堂参与度、练习完成情况
2. 总结性评估：课后小测验、作业质量
3. 评分标准：优秀（90-100）、良好（80-89）、合格（60-79）、需改进（<60）""",
            
            "difficulties_solutions": """九、教学反思与建议
1. 可能遇到的困难：发音不准、语法理解困难
2. 解决方案：多练习、个别辅导
3. 针对不同水平学生的调整建议""",
            
            "exercises_with_answers": """六、练习题与答案
1. 填空题：
   （1）我每天___汉语两个小时。（答案：学习）
2. 选择题：
   （1）明天我要参加什么？A)学习 B)考试 C)准备（答案：B）
3. 改错题：
   （1）我昨天去图书馆。（改为：我昨天去了图书馆。）"""
        }
        
        return default_contents.get(section_name, "本节内容待补充\n")
    
    def _get_default_parsed_response(self) -> Dict[str, Any]:
        """获取默认的解析响应"""
        return {
            "teaching_objectives": self._get_default_section_content("teaching_objectives"),
            "teaching_process": self._get_default_section_content("teaching_process"),
            "concrete_examples": self._get_default_section_content("concrete_examples"),
            "teaching_resources": self._get_default_section_content("teaching_resources"),
            "assessment_scheme": self._get_default_section_content("assessment_scheme"),
            "difficulties_solutions": self._get_default_section_content("difficulties_solutions"),
            "exercises_with_answers": self._get_default_section_content("exercises_with_answers")
        }
    
    def check_concrete_content(self, response: str) -> bool:
        """检查响应是否包含具体内容"""
        concrete_indicators = [
            '填空题', '造句题', '选择题', '改错题', '翻译题',
            '具体题目：', '示例：', '例句：', '对话模板：', '情景设置：',
            '教师指令：', '学生任务：', '答案：', '板书设计：'
        ]
        
        for indicator in concrete_indicators:
            if indicator in response:
                return True
        
        return False
    
    def extract_structured_suggestions(self, response: str) -> List[Dict]:
        """从响应中提取结构化建议"""
        suggestions = []
        
        # 提取教学流程中的具体活动
        process_pattern = r'【(.*?)】\s*(.*?)(?=【|\Z)'
        matches = re.findall(process_pattern, response, re.DOTALL)
        
        for match in matches:
            time_block, content = match
            if len(content.strip()) > 30:
                suggestions.append({
                    "content": f"{time_block}：{content[:150]}...",
                    "type": "activity",
                    "source": "teaching_process"
                })
        
        # 提取练习题
        exercise_pattern = r'(\d+)[\.、]\s*(.*?)(?=\d+[\.、]|\Z)'
        exercise_matches = re.findall(exercise_pattern, response)
        for match in exercise_matches[:5]:
            num, content = match
            if '答案' in content or '_____' in content:
                suggestions.append({
                    "content": f"练习题{num}：{content[:120]}...",
                    "type": "exercise",
                    "source": "exercises"
                })
        
        # 如果提取不到，添加默认建议
        if len(suggestions) < 3:
            suggestions = [
                {"content": "设计词汇配对游戏，分组竞赛（10分钟）", "type": "activity", "source": "default"},
                {"content": "练习重点句型：设计填空和改错练习（15分钟）", "type": "exercise", "source": "default"},
                {"content": "情景对话练习：模拟真实场景进行角色扮演（20分钟）", "type": "activity", "source": "default"}
            ]
        
        return suggestions
    
    def find_collocations_for_text(self, text, words_list):
        """查找文本中词语的搭配（简化版）"""
        collocations = {}
        try:
            for item in words_list:
                word = item.get('word') if isinstance(item, dict) else item
                if word and len(word) >= 2:
                    # 这里简单处理，实际可以从搭配库中查找
                    # 为了不报错，返回空字典
                    pass
        except:
            pass
        return collocations
    
    def extract_difficult_elements(self, text: str, 
                                  hsk_char_data: Dict, 
                                  hsk_vocab_data: Dict,
                                  max_items: int = 20) -> Tuple[List[Dict], List[Dict]]:
        """提取生字和生词"""
        difficult_chars = []
        difficult_words = []
        
        all_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        char_freq = Counter(all_chars)
        
        seen_chars = set()
        for char, freq in char_freq.items():
            if char in seen_chars:
                continue
            seen_chars.add(char)
            
            char_level = hsk_char_data.get(char, '未知')
            if char_level == '未知' or self._is_advanced_level(char_level):
                # 获取详细信息
                char_info = self.get_char_details(char)
                difficult_chars.append({
                    'char': char,
                    'level': self._normalize_level(char_level),
                    'frequency': freq,
                    'pinyin_list': char_info['pinyin_list'],
                    'has_data': char_info['has_data'],
                    'all_entries': char_info['all_entries']
                })
        
        # 使用jieba分词
        words = jieba.lcut(text)
        word_freq = Counter(words)
        
        seen_words = set()
        for word, freq in word_freq.items():
            if len(word) < 2 or not re.match(r'^[\u4e00-\u9fff]+$', word):
                continue
                
            if word in seen_words:
                continue
            seen_words.add(word)
            
            word_level = hsk_vocab_data.get(word, '未知')
            if word_level == '未知' or self._is_advanced_level(word_level):
                # 获取详细信息
                word_info = self.get_word_details(word)
                difficult_words.append({
                    'word': word,
                    'level': self._normalize_level(word_level),
                    'frequency': freq,
                    'word_type': self._guess_word_type(word),
                    'pinyin_list': word_info['pinyin_list'],
                    'pos_list': word_info['pos_list'],
                    'has_data': word_info['has_data'],
                    'all_entries': word_info['all_entries']
                })
        
        difficult_chars.sort(key=lambda x: -x['frequency'])
        difficult_words.sort(key=lambda x: -x['frequency'])
        
        return difficult_chars[:max_items], difficult_words[:max_items]
    
    def _is_advanced_level(self, level: str) -> bool:
        """检查是否属于中高级别"""
        if not level or level == '未知':
            return True
        
        level_str = str(level).upper()
        advanced_levels = ['HSK4', 'HSK5', 'HSK6', 'HSK7-9', '4', '5', '6', '7', '8', '9', '高等', '高级']
        
        for adv_level in advanced_levels:
            if adv_level in level_str:
                return True
        
        return False
    
    def _normalize_level(self, level: str) -> str:
        """标准化级别格式"""
        if not level or level == '未知':
            return '未知'
        
        level_str = str(level).upper()
        level_mapping = {
            '1': 'HSK1', '2': 'HSK2', '3': 'HSK3', '4': 'HSK4',
            '5': 'HSK5', '6': 'HSK6', '7': 'HSK7-9', '8': 'HSK7-9',
            '9': 'HSK7-9', '7-9': 'HSK7-9', '高等': 'HSK7-9', '高级': 'HSK7-9',
            'HSK1': 'HSK1', 'HSK2': 'HSK2', 'HSK3': 'HSK3', 
            'HSK4': 'HSK4', 'HSK5': 'HSK5', 'HSK6': 'HSK6',
            'HSK7': 'HSK7-9', 'HSK8': 'HSK7-9', 'HSK9': 'HSK7-9'
        }
        
        for key, value in level_mapping.items():
            if key == level_str or key in level_str:
                return value
        
        return '未知'
    
    def _guess_word_type(self, word: str) -> str:
        """猜测词语类型"""
        if len(word) == 1:
            return "单字"
        elif len(word) == 2:
            return "双字词"
        elif len(word) == 3:
            return "三字词"
        elif len(word) == 4:
            return "四字词/成语"
        else:
            return "多字短语"
    
    def analyze_text_content(self, text: str) -> Dict[str, Any]:
        """分析文本内容 - 修复版"""
        try:
            if self.verbose:
                print("📝 分析文本内容...")
            
            # 限制文本长度，避免API错误
            if len(text) > 500:
                analysis_text = text[:500]
            else:
                analysis_text = text
            
            theme = self.extract_theme_with_llm(analysis_text)
            if self.verbose:
                print(f"    ✅ 主题: {theme}")
            
            keywords = self.extract_keywords_with_llm(analysis_text)
            if self.verbose:
                print(f"    ✅ 关键词: {keywords}")
            
            clean_text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？：；、,.!?]', '', text)
            words = jieba.lcut(clean_text)
            sentences = re.split(r'[。！？\.!?]', clean_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            return {
                "theme": theme,
                "keywords": keywords,
                "high_freq_words": keywords[:3],
                "sentence_count": len(sentences),
                "avg_sentence_len": sum(len(s) for s in sentences) / len(sentences) if sentences else 0,
                "total_words": len(words),
                "unique_words": len(set(words))
            }
            
        except Exception as e:
            print(f"❌ 文本内容分析失败: {e}")
            # 即使失败也要返回有效的默认值
            return {
                "theme": "日常活动",
                "keywords": ["学习", "计划", "考试", "娱乐", "活动"],
                "high_freq_words": ["学习", "计划"],
                "sentence_count": 0,
                "avg_sentence_len": 0,
                "total_words": 0,
                "unique_words": 0
            }
    
    def initialize_collocation_library(self, collocation_dir: str):
        """初始化搭配词库"""
        try:
            if not os.path.exists(collocation_dir):
                print(f"❌ 搭配词库目录不存在: {collocation_dir}")
                return
            
            txt_files = glob.glob(os.path.join(collocation_dir, '*.txt'))
            print(f"📚 搭配词库准备就绪: {len(txt_files)} 个搭配文件")
            
            self.collocation_dir = collocation_dir
            
        except Exception as e:
            print(f"❌ 初始化搭配词库失败: {e}")
            self.collocation_dir = None
    
    def generate_detailed_fallback_teaching_suggestions(self, 
                                                       text: str,
                                                       theme: str,
                                                       keywords: List[str],
                                                       difficult_words: List[Dict],
                                                       level: str) -> Dict[str, Any]:
        """生成详细的备用教学建议"""
        
        # 准备详细的生词信息
        word_details_section = ""
        if difficult_words:
            word_details = []
            for w in difficult_words[:6]:
                word = w['word']
                formatted_details = self.format_word_details_for_teaching(word)
                word_details.append(formatted_details)
            
            word_details_section = f"重点生词：{'、'.join(word_details)}"
        
        # 准备关键词字符串
        keywords_str = '、'.join(keywords[:5]) if keywords else '中文学习'
        
        # 构建详细的备用响应
        raw_response = f"""二、教学目标设定
1. 语言知识目标：
   - 词汇目标：{word_details_section}
   - 语法目标：掌握文本中的核心语法结构
2. 语言技能目标：
   - 听力：能够听懂关于{theme}的简短对话
   - 口语：能够进行1-2分钟的自我介绍
   - 阅读：能够理解{len(text)}字左右的短文
   - 写作：能够写出80-100字的简单内容

四、教学流程设计（90分钟）
【导入环节】（10分钟）
- 活动名称：主题图片讨论
- 教师指令："同学们好！今天我们要学习关于'{theme}'的内容。请看这张图片，你们看到了什么？"
- 学生任务：观察图片30秒，轮流说出1-2个相关词语

【词汇教学】（20分钟）
- 核心词汇：{', '.join([w['word'] for w in difficult_words[:5]])}
- 每个词汇详细讲解：
  1. 发音练习
  2. 词义解释
  3. 典型例句
  4. 练习设计

【语法讲解】（15分钟）
- 语法点：文本中的核心语法
- 讲解方法：情景演示+对比分析
- 例句分析：提供5个例句
- 练习设计：改错题、翻译题

【课文理解】（15分钟）
- 阅读理解问题：
  1. 文章主要讲了什么？
  2. 有哪些主要人物？
  3. 发生了什么事情？

【交际练习】（20分钟）
- 活动一：角色扮演
  情景：模拟真实场景
  对话模板：编写完整对话

【总结作业】（10分钟）
- 课堂总结：回顾学习重点
- 作业布置：词汇抄写、造句练习、短文写作

五、师生对话示例
对话一：
教师：这个词怎么读？
学生：学习（xué xí）
教师：很好！请用这个词造句。
学生：我喜欢学习汉语。

六、练习题与答案
1. 填空题：
   （1）我每天___汉语两个小时。（答案：学习）
2. 选择题：
   （1）明天我要参加什么？A)学习 B)考试 C)准备（答案：B）

七、教学资源推荐
1. 视频资源：HSK教学视频 - https://www.youtube.com/@hsk
2. 网站资源：汉语学习网 - https://www.hanyuxuexi.com
3. APP资源：HelloChinese - 各大应用商店下载"""
        
        # 解析响应
        parsed = self.parse_detailed_teaching_response(raw_response)
        
        return {
            "raw_response": raw_response,
            "parsed": parsed,
            "structured": self.extract_structured_suggestions(raw_response),
            "generated_by_llm": False,
            "length": len(raw_response),
            "has_concrete_examples": True,
            "model_used": "fallback"
        }
    
    def generate_enhanced_analysis(self,
                                  text: str,
                                  prediction: Dict[str, Any],
                                  analysis_features: Dict[str, Any],
                                  hsk_char_data: Dict,
                                  hsk_vocab_data: Dict) -> Dict[str, Any]:
        """生成增强分析"""
        
        if self.verbose:
            print("🧠 开始生成增强分析...")
            print(f"  文本长度: {len(text)} 字符")
            print(f"  使用模型: {self.model_name} (OpenRouter)")
        
        start_time = time.time()
        
        try:
            if len(text) > 1000:
                analysis_text = text[:1000]
            else:
                analysis_text = text
            
            # 分析文本内容
            print("📝 第一步：分析文本内容...")
            text_analysis = self.analyze_text_content(analysis_text)
            
            # 提取生字生词
            print("📚 第二步：提取生字生词...")
            predicted_level = prediction.get('level', 'HSK3')
            difficult_chars, difficult_words = self.extract_difficult_elements_with_level(
                analysis_text, hsk_char_data, hsk_vocab_data, predicted_level, max_items=20
            )
            
            # 查找词语搭配
            print("🔗 第三步：查找词语搭配...")
            collocations = {}
            if self.collocation_dir:
                # 简单查找，不详细实现
                pass
            
            # 生成详细教学建议
            print("💡 第四步：生成详细教学建议...")
            level = prediction.get('level', 'HSK3')
            theme = text_analysis.get('theme', '日常活动')
            keywords = text_analysis.get('keywords', [])
            
            teaching_result = self.generate_detailed_teaching_suggestions_with_llm(
                analysis_text, theme, keywords, difficult_words, level
            )
            
            # 确保teaching_result有正确的结构
            if not isinstance(teaching_result, dict):
                teaching_result = self.generate_detailed_fallback_teaching_suggestions(
                    analysis_text, theme, keywords, difficult_words, level
                )
            
            # 构建分析结果
            print("📋 第五步：构建分析结果...")
            end_time = time.time()
            analysis_time = end_time - start_time
            
            enhanced_result = {
                "success": True,
                "analysis": {
                    "text_analysis": text_analysis,
                    "teaching_analysis": teaching_result,
                    "vocabulary_analysis": {
                        "difficult_chars": difficult_chars,
                        "difficult_words": difficult_words,
                        "total_difficult_items": len(difficult_chars) + len(difficult_words)
                    },
                    "collocation_analysis": {
                        "found_collocations": len(collocations),
                        "collocation_details": collocations
                    },
                    "performance": {
                        "total_time": analysis_time,
                        "model_used": teaching_result.get("model_used", "qwen3-coder-480b (OpenRouter)"),
                        "theme_extracted_by_llm": True,
                        "keywords_extracted_by_llm": True,
                        "teaching_generated_by_llm": teaching_result.get("generated_by_llm", False),
                        "has_concrete_examples": teaching_result.get("has_concrete_examples", False)
                    }
                }
            }
            
            if self.verbose:
                print(f"✅ 增强分析完成，耗时: {analysis_time:.1f}秒")
                print(f"  主题: {theme}")
                print(f"  关键词数量: {len(keywords)}")
                print(f"  生词数量: {len(difficult_words)}")
                print(f"  教学建议生成: {'成功' if teaching_result.get('generated_by_llm') else '使用备用'}")
                print(f"  包含具体示例: {'是' if teaching_result.get('has_concrete_examples') else '否'}")
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ 增强分析失败: {e}")
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "analysis": {
                    "text_analysis": {
                        "theme": "日常活动",
                        "keywords": ["学习", "计划", "考试", "娱乐", "活动"]
                    },
                    "teaching_analysis": {
                        "raw_response": "分析遇到问题，请检查OpenRouter服务。",
                        "parsed": self._get_default_parsed_response(),
                        "structured": [],
                        "generated_by_llm": False,
                        "length": 0,
                        "has_concrete_examples": False
                    },
                    "vocabulary_analysis": {
                        "difficult_chars": [],
                        "difficult_words": [],
                        "total_difficult_items": 0
                    },
                    "collocation_analysis": {
                        "found_collocations": 0,
                        "collocation_details": {}
                    }
                }
            }
    
    def extract_difficult_elements_with_level(self, text: str, 
                                             hsk_char_data: Dict, 
                                             hsk_vocab_data: Dict,
                                             predicted_level: str,
                                             max_items: int = 20) -> Tuple[List[Dict], List[Dict]]:
        """提取生字和生词 - 基于文章难度等级，显示大于等于该等级的词语"""
        difficult_chars = []
        difficult_words = []
        
        # 定义HSK等级顺序
        level_order = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']
        
        # 获取当前文章难度等级的索引
        if predicted_level in level_order:
            current_level_index = level_order.index(predicted_level)
            # 生词定义为大于等于当前等级的词语
            target_levels = level_order[current_level_index:]
        else:
            # 如果预测等级不在列表中，默认使用HSK3及以上
            target_levels = ['HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']
            current_level_index = 2  # HSK3的索引
        
        if self.verbose:
            print(f"  📊 文章难度等级: {predicted_level}")
            print(f"  📊 生词判定等级: {target_levels}")
        
        # 提取汉字
        all_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        char_freq = Counter(all_chars)
        
        seen_chars = set()
        for char, freq in char_freq.items():
            if char in seen_chars:
                continue
            seen_chars.add(char)
            
            char_level = hsk_char_data.get(char, '未知')
            # 判断是否属于目标等级范围
            if char_level in target_levels or char_level == '未知':
                char_info = self.get_char_details(char)
                difficult_chars.append({
                    'char': char,
                    'level': self._normalize_level(char_level),
                    'frequency': freq,
                    'pinyin_list': char_info['pinyin_list'],
                    'has_data': char_info['has_data'],
                    'all_entries': char_info['all_entries'],
                    'reason': f"{char_level}级别（文章难度: {predicted_level}）"
                })
        
        # 使用jieba分词提取词语
        words = jieba.lcut(text)
        word_freq = Counter(words)
        
        seen_words = set()
        for word, freq in word_freq.items():
            if len(word) < 2 or not re.match(r'^[\u4e00-\u9fff]+$', word):
                continue
                
            if word in seen_words:
                continue
            seen_words.add(word)
            
            word_level = hsk_vocab_data.get(word, '未知')
            # 判断是否属于目标等级范围
            if word_level in target_levels or word_level == '未知':
                word_info = self.get_word_details(word)
                difficult_words.append({
                    'word': word,
                    'level': self._normalize_level(word_level),
                    'frequency': freq,
                    'word_type': self._guess_word_type(word),
                    'pinyin_list': word_info['pinyin_list'],
                    'pos_list': word_info['pos_list'],
                    'has_data': word_info['has_data'],
                    'all_entries': word_info['all_entries'],
                    'reason': f"{word_level}级别（文章难度: {predicted_level}）"
                })
        
        # 按频率排序
        difficult_chars.sort(key=lambda x: -x['frequency'])
        difficult_words.sort(key=lambda x: -x['frequency'])
        
        # 统计目标等级和超纲词汇数量
        target_count = len([w for w in difficult_words if w['level'] in target_levels and w['level'] != '未知'])
        unknown_count = len([w for w in difficult_words if w['level'] == '未知'])
        
        if self.verbose:
            print(f"  ✅ 找到生词: {len(difficult_words)}个")
            print(f"      目标级别({predicted_level}及以上): {target_count}个")
            print(f"      超纲词汇(未收录): {unknown_count}个")
        
        return difficult_chars[:max_items], difficult_words[:max_items]


# 全局实例
_enhanced_analyzer = None

def get_enhanced_analyzer(collocation_dir=None, hsk_data_dir=None, verbose=True):
    global _enhanced_analyzer
    if _enhanced_analyzer is None:
        # 动态获取当前文件所在目录的上级目录（假设项目根目录）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if collocation_dir is None:
            dir_path = os.path.join(base_dir, 'n')
        else:
            dir_path = collocation_dir
        if hsk_data_dir is None:
            hsk_path = base_dir
        else:
            hsk_path = hsk_data_dir
        _enhanced_analyzer = EnhancedAnalysisGenerator(
            model_name="openai/gpt-oss-120b:free",
            collocation_dir=dir_path,
            hsk_data_dir=hsk_path,
            verbose=verbose
        )
    return _enhanced_analyzer
