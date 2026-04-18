# vocab_db.py - 完整修复版本
import pandas as pd
import os
import re
import chardet
from collections import defaultdict

class HSKVocabularyDB:
    """HSK汉字和词语数据库"""
    
    def __init__(self, data_dir=None, verbose=True):
        self.verbose = verbose
        self.char_data = None
        self.word_data = None
        self.char_level_map = {}
        self.word_level_map = {}
        self.level_chars = defaultdict(set)
        self.level_words = defaultdict(set)
        
        if data_dir is None:
            data_dir = self._get_package_data_path()
        
        self.load_data(data_dir)
    
    def _get_package_data_path(self):
        """获取包内的数据文件路径"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'data'),
            os.path.join(os.path.dirname(__file__), '../data'),
            os.path.join(os.path.dirname(__file__), '../../data'),
            'C:/Users/DF-Lenovo/Desktop/hsk_predictor/data',
            './data',
            '../data',
            os.path.join(os.path.expanduser('~'), 'Desktop/hsk_predictor/data')
        ]
        
        for path in possible_paths:
            char_path = os.path.join(path, '汉字.csv')
            if os.path.exists(char_path):
                if self.verbose:
                    print(f"✅ 找到数据目录: {path}")
                return path
        
        if self.verbose:
            print("❌ 无法找到数据目录，尝试的路径:")
            for path in possible_paths:
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"   {exists} {path}")
        
        raise FileNotFoundError("无法找到数据目录，请确保data文件夹存在")
    
    def _is_chinese_char(self, char):
        """判断字符是否为中文字符"""
        if not char:
            return False
        
        try:
            cp = ord(char)
        except:
            return False
        
        if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0x2CEB0 <= cp <= 0x2EBEF)):
            return True
        
        chinese_punctuation = '，。！？；："「」『』（）【】《》〈〉、·～'
        if char in chinese_punctuation:
            return True
        
        return False
    
    def detect_file_encoding(self, filepath):
        """检测文件编码"""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                if encoding is None:
                    encoding = 'gbk'
                elif encoding.lower() in ['ascii', 'windows-1252']:
                    encoding = 'gbk'
                
                if self.verbose:
                    print(f"🔍 检测到编码: {encoding} (置信度: {confidence:.2%})")
                
                return encoding
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 编码检测失败: {e}")
            return 'gbk'
    
    def read_csv_with_encoding(self, filepath):
        """尝试用多种编码读取CSV文件"""
        encodings_to_try = [
            'gbk', 'gb2312', 'gb18030',
            'utf-8', 'utf-8-sig',
            'latin1', 'cp1252',
            'big5',
        ]
        
        for encoding in encodings_to_try:
            try:
                if self.verbose:
                    print(f"🔄 尝试用 {encoding} 编码读取...")
                
                df = pd.read_csv(filepath, encoding=encoding)
                
                if not df.empty and len(df.columns) > 0:
                    if self.verbose:
                        print(f"✅ 成功用 {encoding} 编码读取")
                        print(f"   文件: {os.path.basename(filepath)}")
                        print(f"   形状: {df.shape}")
                        print(f"   列名: {list(df.columns)}")
                    
                    return df, encoding
            except Exception as e:
                if self.verbose and encoding in ['gbk', 'utf-8']:
                    print(f"❌ {encoding} 编码失败: {str(e)[:50]}")
                continue
        
        try:
            if self.verbose:
                print("🔄 尝试使用错误忽略模式读取...")
            df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
            if not df.empty:
                if self.verbose:
                    print(f"⚠️ 使用错误忽略模式读取成功")
                return df, 'utf-8-ignore'
        except Exception as e:
            if self.verbose:
                print(f"❌ 所有读取方法都失败: {e}")
        
        raise ValueError(f"无法读取文件: {filepath}")
    
    def standardize_level(self, level_str):
        """标准化等级字符串"""
        if not isinstance(level_str, str):
            return str(level_str)
        
        level_str = level_str.strip()
        
        # 映射等级到标准格式
        level_mapping = {
            # 汉字等级
            '一级': '1', '二级': '2', '三级': '3', '四级': '4', 
            '五级': '5', '六级': '6', '高等': '7-9', '高等[附表]': '7-9',
            
            # 词语等级
            '初等': '1-2', '中等': '3-4', '高等': '5-6', '高等[附表]': '7-9',
            '一级': '1', '二级': '2', '三级': '3', '四级': '4',
            '五级': '5', '六级': '6', '七级': '7', '八级': '8', '九级': '9',
        }
        
        if level_str in level_mapping:
            return level_mapping[level_str]
        
        # 如果是数字形式，直接返回
        if level_str.isdigit() and 1 <= int(level_str) <= 9:
            return level_str
        
        # 如果是范围，如"1-2"
        if '-' in level_str:
            return level_str
        
        return 'unknown'
    
    def load_data(self, data_dir):
        """加载汉字和词语数据"""
        try:
            if self.verbose:
                print("\n" + "=" * 60)
                print("📚 开始加载HSK数据")
                print("=" * 60)
            
            # 1. 加载汉字数据
            char_path = os.path.join(data_dir, '汉字.csv')
            if os.path.exists(char_path):
                if self.verbose:
                    print(f"\n1️⃣ 加载汉字表: {char_path}")
                
                # 汉字文件是GBK编码
                self.char_data = pd.read_csv(char_path, encoding='gbk')
                
                if self.verbose:
                    print(f"✅ 汉字文件读取成功")
                    print(f"   形状: {self.char_data.shape}")
                    print(f"   列名: {list(self.char_data.columns)}")
                    print(f"   示例数据:")
                    print(self.char_data.head(3))
                
                # 检查并标准化列名
                if '汉字' not in self.char_data.columns:
                    print(f"⚠️ 汉字文件缺少'汉字'列，现有列: {list(self.char_data.columns)}")
                    # 尝试找到正确的列
                    for col in self.char_data.columns:
                        if '汉字' in str(col) or '字' in str(col):
                            self.char_data = self.char_data.rename(columns={col: '汉字'})
                            break
                
                if '级别' not in self.char_data.columns:
                    print(f"⚠️ 汉字文件缺少'级别'列，现有列: {list(self.char_data.columns)}")
                    for col in self.char_data.columns:
                        if '级别' in str(col) or '等级' in str(col):
                            self.char_data = self.char_data.rename(columns={col: '级别'})
                            break
                
                # 清理数据
                self.char_data = self.char_data.dropna(subset=['汉字'])
                self.char_data['汉字'] = self.char_data['汉字'].astype(str).str.strip()
                self.char_data['级别'] = self.char_data['级别'].astype(str).str.strip()
                
                # 标准化等级
                self.char_data['标准化级别'] = self.char_data['级别'].apply(self.standardize_level)
                
                # 构建汉字映射
                char_count = 0
                for _, row in self.char_data.iterrows():
                    chars = str(row['汉字'])
                    level = str(row['标准化级别'])
                    
                    for char in chars:
                        if char and len(char.strip()) > 0:
                            clean_char = char.strip()
                            if self._is_chinese_char(clean_char):
                                self.char_level_map[clean_char] = level
                                self.level_chars[level].add(clean_char)
                                char_count += 1
                
                if self.verbose:
                    print(f"📊 加载了 {char_count} 个汉字")
                    print(f"   唯一汉字: {len(self.char_level_map)}")
                    print(f"   等级分布:")
                    for level in sorted(self.level_chars.keys()):
                        print(f"     {level}: {len(self.level_chars[level])}个")
            else:
                if self.verbose:
                    print(f"⚠️ 汉字文件不存在: {char_path}")
            
            # 2. 加载词语数据
            word_path = os.path.join(data_dir, '词汇.csv')
            if os.path.exists(word_path):
                if self.verbose:
                    print(f"\n2️⃣ 加载词语表: {word_path}")
                
                # 词语文件是UTF-8编码
                self.word_data = pd.read_csv(word_path, encoding='utf-8')
                
                if self.verbose:
                    print(f"✅ 词语文件读取成功")
                    print(f"   形状: {self.word_data.shape}")
                    print(f"   列名: {list(self.word_data.columns)}")
                    print(f"   示例数据:")
                    print(self.word_data.head(3))
                
                # 检查并标准化列名
                if '词语' not in self.word_data.columns:
                    print(f"⚠️ 词语文件缺少'词语'列，现有列: {list(self.word_data.columns)}")
                    for col in self.word_data.columns:
                        if '词语' in str(col) or '词' in str(col):
                            self.word_data = self.word_data.rename(columns={col: '词语'})
                            break
                
                if '级别' not in self.word_data.columns:
                    print(f"⚠️ 词语文件缺少'级别'列，现有列: {list(self.word_data.columns)}")
                    for col in self.word_data.columns:
                        if '级别' in str(col) or '等级' in str(col):
                            self.word_data = self.word_data.rename(columns={col: '级别'})
                            break
                
                # 清理数据
                self.word_data = self.word_data.dropna(subset=['词语'])
                self.word_data['词语'] = self.word_data['词语'].astype(str).str.strip()
                self.word_data['级别'] = self.word_data['级别'].astype(str).str.strip()
                
                # 标准化等级
                self.word_data['标准化级别'] = self.word_data['级别'].apply(self.standardize_level)
                
                # 构建词语映射
                word_count = 0
                for _, row in self.word_data.iterrows():
                    word = str(row['词语']).strip()
                    level = str(row['标准化级别'])
                    
                    if word and len(word) >= 1:
                        if any(self._is_chinese_char(c) for c in word):
                            self.word_level_map[word] = level
                            self.level_words[level].add(word)
                            word_count += 1
                
                if self.verbose:
                    print(f"📊 加载了 {word_count} 个词语")
                    print(f"   唯一词语: {len(self.word_level_map)}")
                    print(f"   等级分布:")
                    for level in sorted(self.level_words.keys()):
                        print(f"     {level}: {len(self.level_words[level])}个")
            else:
                if self.verbose:
                    print(f"⚠️ 词语文件不存在: {word_path}")
            
            if self.verbose:
                print("\n" + "=" * 60)
                print("✅ 数据加载完成!")
                print("=" * 60)
                
                total_chars = len(self.char_level_map)
                total_words = len(self.word_level_map)
                print(f"📊 总计: {total_chars} 汉字, {total_words} 词语")
                
                if total_chars == 0 and total_words == 0:
                    print("⚠️ 警告: 没有加载到任何数据，请检查CSV文件格式")
                else:
                    print("\n🔤 汉字等级示例:")
                    test_chars = ["你", "的", "中", "文", "很", "好"]
                    for char in test_chars:
                        level = self.get_char_level(char)
                        print(f"   '{char}': {level if level else '未知'}")
                    
                    print("\n📚 词语等级示例:")
                    test_words = ["你好", "中文", "很好", "你的"]
                    for word in test_words:
                        level = self.get_word_level(word)
                        print(f"   '{word}': {level if level else '未知'}")
        
        except Exception as e:
            if self.verbose:
                print(f"\n❌ 加载数据失败: {e}")
                import traceback
                traceback.print_exc()
            
            self.char_data = pd.DataFrame()
            self.word_data = pd.DataFrame()
            
            if self.verbose:
                print("⚠️ 使用空数据继续运行")
    
    def segment_text(self, text):
        """中文分词（优先匹配词语，然后汉字）"""
        if not text:
            return []
        
        if not self.word_level_map:
            return [char for char in text if self._is_chinese_char(char)]
        
        sorted_words = sorted(self.word_level_map.keys(), key=len, reverse=True)
        
        words = []
        i = 0
        text_length = len(text)
        
        while i < text_length:
            char = text[i]
            
            if not self._is_chinese_char(char):
                if char in '，。！？；："「」『』（）【】《》〈〉、·～':
                    words.append(char)
                i += 1
                continue
            
            matched = False
            
            for word in sorted_words:
                word_len = len(word)
                if i + word_len <= text_length and text[i:i+word_len] == word:
                    if all(self._is_chinese_char(c) for c in word):
                        words.append(word)
                        i += word_len
                        matched = True
                        break
            
            if not matched:
                words.append(char)
                i += 1
        
        return words
    
    def analyze_text(self, text):
        """全面分析文本"""
        if not text or not self.char_level_map:
            return {
                'segmented': [],
                'char_analysis': {},
                'word_analysis': {},
                'unknown_tokens': [],
                'punctuation': [],
                'char_summary': {},
                'word_summary': {},
                'total_chars': 0,
                'total_words': 0
            }
        
        words = self.segment_text(text)
        
        char_analysis = defaultdict(list)
        word_analysis = defaultdict(list)
        unknown_tokens = []
        punctuation = []
        
        for token in words:
            if token in '，。！？；："「」『』（）【】《》〈〉、·～':
                punctuation.append(token)
                continue
                
            if len(token) == 1:
                level = self.get_char_level(token)
                if level:
                    # 标准化等级到1-9或7-9
                    if level in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '7-9']:
                        char_analysis[level].append(token)
                    elif level == '1-2':
                        char_analysis['1'].append(token)
                    elif level == '3-4':
                        char_analysis['3'].append(token)
                    elif level == '5-6':
                        char_analysis['5'].append(token)
                    else:
                        unknown_tokens.append(token)
                else:
                    unknown_tokens.append(token)
            else:
                level = self.get_word_level(token)
                if level:
                    # 标准化等级
                    if level in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '7-9']:
                        word_analysis[level].append(token)
                    elif level == '1-2':
                        word_analysis['1'].append(token)
                        word_analysis['2'].append(token)
                    elif level == '3-4':
                        word_analysis['3'].append(token)
                        word_analysis['4'].append(token)
                    elif level == '5-6':
                        word_analysis['5'].append(token)
                        word_analysis['6'].append(token)
                    else:
                        unknown_tokens.append(token)
                else:
                    known_chars = [self.get_char_level(c) for c in token]
                    if all(known_chars):
                        word_analysis['复合'].append(token)
                    else:
                        unknown_tokens.append(token)
        
        # 去重并排序
        for level in char_analysis:
            char_analysis[level] = sorted(set(char_analysis[level]))
        for level in word_analysis:
            word_analysis[level] = sorted(set(word_analysis[level]))
        
        # 计算总数（只计算标准等级）
        standard_levels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '7-9']
        total_chars = sum(len(chars) for level, chars in char_analysis.items() if level in standard_levels)
        total_words = sum(len(words) for level, words in word_analysis.items() if level in standard_levels)
        
        return {
            'segmented': words,
            'char_analysis': dict(char_analysis),
            'word_analysis': dict(word_analysis),
            'unknown_tokens': unknown_tokens,
            'punctuation': punctuation,
            'char_summary': {level: len(chars) for level, chars in char_analysis.items()},
            'word_summary': {level: len(words) for level, words in word_analysis.items()},
            'total_chars': total_chars,
            'total_words': total_words
        }
    
    def get_char_level(self, char):
        """获取汉字等级"""
        return self.char_level_map.get(char, None)
    
    def get_word_level(self, word):
        """获取词语等级"""
        return self.word_level_map.get(word, None)
    
    def get_level_chars(self, level):
        """获取指定等级的所有汉字"""
        return self.level_chars.get(str(level), set())
    
    def get_level_words(self, level):
        """获取指定等级的所有词语"""
        return self.level_words.get(str(level), set())
    
    def get_colored_text(self, text, color_scheme=None):
        """获取带颜色标注的文本（HTML格式）"""
        if color_scheme is None:
            color_scheme = {
                '1': '#FFCCCC',  # HSK1
                '2': '#FF9999',  # HSK2
                '3': '#FF6666',  # HSK3
                '4': '#FF9933',  # HSK4
                '5': '#FF6600',  # HSK5
                '6': '#CC3300',  # HSK6
                '7': '#990000',  # HSK7
                '8': '#770000',  # HSK8
                '9': '#550000',  # HSK9
                '7-9': '#990000',  # HSK7-9
                '复合': '#CC99FF',  # 复合词语
                'unknown': '#CCCCCC'  # 未知
            }
        
        words = self.segment_text(text)
        colored_parts = []
        
        for token in words:
            if token in '，。！？；："「」『』（）【】《》〈〉、·～':
                colored_parts.append(f'<span style="background-color: #CCCCCC; padding: 2px; margin: 1px; border-radius: 3px;">{token}</span>')
                continue
            
            if len(token) == 1:
                level = self.get_char_level(token)
                if level:
                    # 标准化等级
                    if level == '1-2':
                        level = '1'
                    elif level == '3-4':
                        level = '3'
                    elif level == '5-6':
                        level = '5'
            else:
                level = self.get_word_level(token)
                if not level and all(self.get_char_level(c) for c in token):
                    level = '复合'
            
            color = color_scheme.get(level, color_scheme['unknown'])
            colored_parts.append(f'<span style="background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px;">{token}</span>')
        
        return ''.join(colored_parts)
    
    def get_stats(self):
        """获取数据库统计信息"""
        stats = {
            'total_chars': len(self.char_level_map),
            'total_words': len(self.word_level_map),
            'char_by_level': {},
            'word_by_level': {}
        }
        
        for level in sorted(self.level_chars.keys()):
            stats['char_by_level'][f'HSK{level}'] = len(self.level_chars[level])
        
        for level in sorted(self.level_words.keys()):
            stats['word_by_level'][f'HSK{level}'] = len(self.level_words[level])
        
        return stats

# 全局实例
_vocab_db_instance = None

def get_vocab_db(verbose=True):
    """获取全局词语数据库实例"""
    global _vocab_db_instance
    if _vocab_db_instance is None:
        _vocab_db_instance = HSKVocabularyDB(verbose=verbose)
    return _vocab_db_instance