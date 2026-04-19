"""
直接调用修复版predictor的API - OpenRouter版
支持真实HSK词汇表分析和词汇高亮显示
新增：增强分析功能（使用OpenRouter Qwen3 Coder 480B模型，180秒超时）
修复：多音字词等级处理、教学建议生成
"""

import sys
import os
import re
import pandas as pd
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import glob
from collections import Counter
import pickle
import zipfile
import shutil

# ========== 配置：请填写您 GitHub Releases 中各个压缩包的下载地址 ==========
# 格式：'本地目录或关键文件': '下载地址'
# 程序会检查关键文件是否存在，如果不存在则下载对应的压缩包并解压
RELEASE_FILES = {
    # 模型文件（必须）
    'models/best_optimized_model.pth': 'https://github.com/li0768/hsk-data/releases/download/v1.0/models.zip',
    # 词汇表数据（必须）
    'data/词汇.csv': 'https://github.com/li0768/hsk-data/releases/download/v1.0/data.zip',
    # 搭配词库（可选，但建议）
    'n/学习.txt': 'https://github.com/li0768/hsk-data/releases/download/v1.0/n.zip',
    'left/图书馆.txt': 'https://github.com/li0768/hsk-data/releases/download/v1.0/left.zip',
    'right/图书馆.txt': 'https://github.com/li0768/hsk-data/releases/download/v1.0/right.zip',
    # 图片库（可选）
    '常用词语释义图片库/': 'https://github.com/li0768/hsk-data/releases/download/v1.0/images.zip',
}
# ====================================================================

# 全局变量
custom_tokenizer = None
predictor = None
hsk_vocabulary_cache = {}
hsk_chars_cache = {}
enhanced_analyzer = None
collocation_data_loaded = False

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 启动HSK文本分析API - OpenRouter增强优化版")

# ========== 新增：从 GitHub Releases 下载所有必要文件 ==========

def download_and_extract(url, target_dir=None, check_file=None):
    """下载 zip 文件并解压到指定目录"""
    if target_dir is None:
        target_dir = current_dir
    # 如果提供了检查文件且已存在，则跳过
    if check_file and os.path.exists(check_file):
        print(f"✅ {check_file} 已存在，跳过下载 {os.path.basename(url)}")
        return True
    print(f"📦 开始下载: {url}")
    zip_path = os.path.join(current_dir, "temp_download.zip")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    print(f"   下载进度: {downloaded/total_size*100:.1f}%", end='\r')
        print(f"\n   ✅ 下载完成，正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"   ✅ 解压完成")
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"   ❌ 下载或解压失败: {e}")
        return False

def download_all_assets():
    """检查并下载所有缺失的资产文件"""
    print("\n🔍 检查本地文件完整性...")
    all_success = True
    for check_path, url in RELEASE_FILES.items():
        # 如果 check_path 以 '/' 结尾，表示这是一个目录，检查目录是否存在且非空
        if check_path.endswith('/'):
            dir_path = os.path.join(current_dir, check_path)
            if os.path.isdir(dir_path) and len(os.listdir(dir_path)) > 0:
                print(f"✅ 目录 {check_path} 已存在且非空，跳过")
                continue
            else:
                print(f"⚠️ 目录 {check_path} 缺失或为空，开始下载...")
                ok = download_and_extract(url, check_file=dir_path)
                if not ok:
                    all_success = False
        else:
            file_path = os.path.join(current_dir, check_path)
            if os.path.exists(file_path):
                print(f"✅ 文件 {check_path} 已存在，跳过")
                continue
            else:
                print(f"⚠️ 文件 {check_path} 缺失，开始下载...")
                ok = download_and_extract(url, check_file=file_path)
                if not ok:
                    all_success = False
    if all_success:
        print("✅ 所有资产文件准备就绪")
    else:
        print("⚠️ 部分资产下载失败，但应用可能仍可运行（某些功能可能受限）")
    return all_success

# ========== 基础函数 ==========

def guess_word_type(word):
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


def generate_fallback_enhanced_analysis(text, prediction_result, analysis_features):
    """生成备用的增强分析结果"""
    return {
        "theme_keywords": {
            "theme": "中文学习文本",
            "keywords": ["学习", "中文", "教学"],
            "keywords_extracted": False
        },
        "teaching_analysis": {
            "summary": "教学分析生成中，请稍候...",
            "raw": "教学分析生成中，请稍候...",
            "predicted_level": prediction_result.get('level', 'HSK3'),
            "confidence": prediction_result.get('confidence', 0.7),
            "estimated_level": analysis_features.get('estimated_hsk_level', 'HSK3'),
            "n_plus_1_level": "HSK4",
            "n_plus_1_principle": "HSK3 → HSK4"
        },
        "vocabulary_analysis": {
            "difficult_words": [],
            "total_difficult_words": 0,
            "target_level_words": 0,
            "other_difficult_words": 0,
            "note": "基于N+1原则，重点教学HSK4级别词汇"
        },
        "collocation_analysis": {
            "found_collocations": 0,
            "collocation_details": {},
            "collocation_words": []
        },
        "practical_suggestions": [
            "设计词汇卡片配对游戏，分组竞赛（10分钟）",
            "练习重点搭配：设计填空练习（8分钟）",
            "设计阶梯式阅读任务，逐步增加文本复杂度（15分钟）"
        ],
        "structured_suggestions": [
            {"content": "设计词汇卡片配对游戏，分组竞赛（10分钟）", "type": "vocabulary", "source": "system"},
            {"content": "练习重点搭配：设计填空练习（8分钟）", "type": "collocation", "source": "system"},
            {"content": "设计阶梯式阅读任务，逐步增加文本复杂度（15分钟）", "type": "activity", "source": "system"}
        ],
        "text_statistics": {
            "length": len(text),
            "sentence_count": analysis_features.get('sentence_count', 1),
            "chinese_char_count": analysis_features.get('chinese_char_count', len(text)),
            "estimated_hsk_level": analysis_features.get('estimated_hsk_level', 'HSK3')
        }
    }

def get_hsk_level_color(level):
    """获取HSK等级对应的颜色"""
    color_map = {
        "HSK1": "#93c5fd",
        "HSK2": "#60a5fa",
        "HSK3": "#3b82f6",
        "HSK4": "#fbbf24",
        "HSK5": "#f59e0b",
        "HSK6": "#d97706",
        "HSK7-9": "#dc2626",
        "未知": "#9ca3af"
    }
    return color_map.get(level, "#9ca3af")

def load_custom_tokenizer():
    """加载自定义分词器"""
    tokenizer_path = os.path.join(current_dir, 'models', 'best_optimized_model_tokenizer.pkl')
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ 分词器文件不存在: {tokenizer_path}")
        return None
    
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✅ 自定义分词器加载成功")
        return tokenizer
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        traceback.print_exc()
        return None

def tokenize_with_custom_tokenizer(tokenizer, text):
    """使用自定义分词器分词"""
    if tokenizer is None:
        words = re.findall(r'[\u4e00-\u9fff]{1,4}', text)
        return words
    
    try:
        clean_text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？：；、,.!?]', '', text)
        
        if hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(clean_text)
        elif hasattr(tokenizer, 'cut'):
            tokens = list(tokenizer.cut(clean_text))
        elif callable(tokenizer):
            tokens = tokenizer(clean_text)
        else:
            tokens = list(tokenizer(clean_text))
        
        tokens = [t for t in tokens if re.match(r'^[\u4e00-\u9fff]+$', t) and len(t) >= 1]
        return tokens
    except Exception as e:
        print(f"❌ 分词失败: {e}")
        return re.findall(r'[\u4e00-\u9fff]{1,4}', text)

# ========== 模块导入 ==========

try:
    from predictor_fixed import HSKTextPredictor
    print("✅ 导入predictor_fixed成功")
except ImportError as e:
    print(f"❌ 导入predictor_fixed失败: {e}")
    HSKTextPredictor = None

ENHANCED_ANALYZER_AVAILABLE = False
try:
    from enhanced_analysis import get_enhanced_analyzer
    ENHANCED_ANALYZER_AVAILABLE = True
    print("✅ 增强分析模块导入成功（OpenRouter版）")
except ImportError as e:
    ENHANCED_ANALYZER_AVAILABLE = False
    print(f"⚠️ 增强分析模块导入失败: {e}")
    print("   请确保已创建 openrouter_client.py 和 teaching_prompts.py 文件")

LOCAL_LLM_AVAILABLE = False
try:
    from llm_local.local_llm import get_local_llm
    LOCAL_LLM_AVAILABLE = True
    print("✅ 本地大模型模块可用")
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    print(f"⚠️ 本地大模型模块导入失败: {e}")

# ========== Flask应用 ==========

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ========== HSK数据处理函数 ==========

def map_hsk_level(level_str):
    """映射HSK级别字符串到标准格式"""
    if not level_str:
        return "未知"
    
    level_str = str(level_str).strip()
    
    level_mapping = {
        '一级': 'HSK1', '二级': 'HSK2', '三级': 'HSK3', 
        '四级': 'HSK4', '五级': 'HSK5', '六级': 'HSK6',
        '高等': 'HSK7-9', '高级': 'HSK7-9',
        '1级': 'HSK1', '2级': 'HSK2', '3级': 'HSK3',
        '4级': 'HSK4', '5级': 'HSK5', '6级': 'HSK6',
        '1': 'HSK1', '2': 'HSK2', '3': 'HSK3',
        '4': 'HSK4', '5': 'HSK5', '6': 'HSK6',
        '7': 'HSK7-9', '8': 'HSK7-9', '9': 'HSK7-9',
        'hsk1': 'HSK1', 'hsk2': 'HSK2', 'hsk3': 'HSK3',
        'hsk4': 'HSK4', 'hsk5': 'HSK5', 'hsk6': 'HSK6',
        'hsk7': 'HSK7-9', 'hsk8': 'HSK7-9', 'hsk9': 'HSK7-9',
        'HSK1': 'HSK1', 'HSK2': 'HSK2', 'HSK3': 'HSK3',
        'HSK4': 'HSK4', 'HSK5': 'HSK5', 'HSK6': 'HSK6',
        'HSK7': 'HSK7-9', 'HSK8': 'HSK7-9', 'HSK9': 'HSK7-9',
        'HSK7-9': 'HSK7-9'
    }
    
    if level_str in level_mapping:
        return level_mapping[level_str]
    
    match = re.search(r'(\d+)', level_str)
    if match:
        level_num = int(match.group(1))
        if level_num <= 6:
            return f"HSK{level_num}"
        else:
            return "HSK7-9"
    
    return "HSK7-9"

def load_hsk_vocabulary():
    """加载HSK词汇表和汉字表，处理多音字词等级归属"""
    global hsk_vocabulary_cache, hsk_chars_cache
    
    if hsk_vocabulary_cache and hsk_chars_cache:
        return hsk_vocabulary_cache, hsk_chars_cache
    
    print("📚 开始加载HSK词汇表（含多音字处理）...")
    
    base_dir = current_dir
    vocab_path = os.path.join(base_dir, 'data', '词汇.csv')
    chars_path = os.path.join(base_dir, 'data', '汉字.csv')
    
    print(f"  词汇表路径: {vocab_path}")
    print(f"  汉字表路径: {chars_path}")
    
    hsk_vocabulary_cache = {}
    hsk_chars_cache = {}
    
    try:
        # ========== 1. 加载词汇表 ==========
        print("  1. 加载词汇表（处理多音字词）...")
        if os.path.exists(vocab_path):
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig']
            vocab_df = None
            
            for encoding in encodings:
                try:
                    vocab_df = pd.read_csv(vocab_path, encoding=encoding)
                    print(f"    ✅ 使用 {encoding} 编码成功加载")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"    ❌ {encoding} 编码错误: {e}")
                    continue
            
            if vocab_df is not None:
                print(f"    📊 词汇表记录数: {len(vocab_df)}")
                
                col_mapping = {
                    '词语': ['词语', '词', '单词', 'word', 'Word', '词汇'],
                    '级别': ['级别', '等级', 'level', 'Level', 'HSK级别', 'HSK等级']
                }
                
                for target_col, possible_cols in col_mapping.items():
                    for col in possible_cols:
                        if col in vocab_df.columns:
                            if col != target_col:
                                vocab_df = vocab_df.rename(columns={col: target_col})
                            break
                
                if '词语' in vocab_df.columns and '级别' in vocab_df.columns:
                    word_level_dict = {}
                    
                    for _, row in vocab_df.iterrows():
                        word = str(row['词语']).strip()
                        level_str = str(row['级别']).strip()
                        
                        if not word:
                            continue
                        
                        level = map_hsk_level(level_str)
                        
                        if word not in word_level_dict:
                            word_level_dict[word] = set()
                        word_level_dict[word].add(level)
                    
                    # 处理多音字词：取最低级别
                    for word, levels in word_level_dict.items():
                        level_order = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9', '未知']
                        
                        min_level = None
                        min_index = float('inf')
                        
                        for level in levels:
                            if level in level_order:
                                idx = level_order.index(level)
                                if idx < min_index:
                                    min_index = idx
                                    min_level = level
                            else:
                                if min_level is None:
                                    min_level = "HSK7-9"
                        
                        if min_level is None:
                            min_level = list(levels)[0] if levels else "未知"
                        
                        if min_level not in hsk_vocabulary_cache:
                            hsk_vocabulary_cache[min_level] = set()
                        hsk_vocabulary_cache[min_level].add(word)
                        
                        if len(word) == 1:
                            if min_level not in hsk_chars_cache:
                                hsk_chars_cache[min_level] = set()
                            hsk_chars_cache[min_level].add(word)
                    
                    print(f"    ✅ 词汇缓存构建完成（已处理多音字词）: {sum(len(v) for v in hsk_vocabulary_cache.values())} 个词语")
                    
                    for level in sorted(hsk_vocabulary_cache.keys()):
                        count = len(hsk_vocabulary_cache[level])
                        sample = list(hsk_vocabulary_cache[level])[:3]
                        print(f"      {level}: {count} 词, 示例: {sample}")
                else:
                    print("    ❌ 词汇表缺少必要列")
            else:
                print("    ❌ 无法加载词汇表文件")
        else:
            print("    ❌ 词汇表文件不存在")
        
        # ========== 2. 加载汉字表 ==========
        print("\n  2. 加载汉字表（处理多音字）...")
        if os.path.exists(chars_path):
            encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-8-sig']
            chars_df = None
            
            for encoding in encodings:
                try:
                    chars_df = pd.read_csv(chars_path, encoding=encoding)
                    print(f"    ✅ 使用 {encoding} 编码成功加载")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"    ❌ {encoding} 编码错误: {e}")
                    continue
            
            if chars_df is not None:
                print(f"    📊 汉字表记录数: {len(chars_df)}")
                
                col_mapping = {
                    '汉字': ['汉字', '字', 'char', 'Char', '字符'],
                    '级别': ['级别', '等级', 'level', 'Level', 'HSK级别', 'HSK等级']
                }
                
                for target_col, possible_cols in col_mapping.items():
                    for col in possible_cols:
                        if col in chars_df.columns:
                            if col != target_col:
                                chars_df = chars_df.rename(columns={col: target_col})
                            break
                
                if '汉字' in chars_df.columns and '级别' in chars_df.columns:
                    char_level_dict = {}
                    
                    for _, row in chars_df.iterrows():
                        char = str(row['汉字']).strip()
                        level_str = str(row['级别']).strip()
                        
                        if not char or len(char) != 1:
                            continue
                        
                        level = map_hsk_level(level_str)
                        
                        if char not in char_level_dict:
                            char_level_dict[char] = set()
                        char_level_dict[char].add(level)
                    
                    # 处理多音字：取最低级别
                    for char, levels in char_level_dict.items():
                        level_order = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9', '未知']
                        
                        min_level = None
                        min_index = float('inf')
                        
                        for level in levels:
                            if level in level_order:
                                idx = level_order.index(level)
                                if idx < min_index:
                                    min_index = idx
                                    min_level = level
                            else:
                                if min_level is None:
                                    min_level = "HSK7-9"
                        
                        if min_level is None:
                            min_level = list(levels)[0] if levels else "未知"
                        
                        if min_level not in hsk_chars_cache:
                            hsk_chars_cache[min_level] = set()
                        hsk_chars_cache[min_level].add(char)
                    
                    print(f"    ✅ 汉字缓存构建完成（已处理多音字）: {sum(len(v) for v in hsk_chars_cache.values())} 个汉字")
                    
                    for level in sorted(hsk_chars_cache.keys()):
                        count = len(hsk_chars_cache[level])
                        sample = ''.join(list(hsk_chars_cache[level])[:10])
                        print(f"      {level}: {count} 字, 示例: {sample}")
                else:
                    print("    ❌ 汉字表缺少必要列")
            else:
                print("    ❌ 无法加载汉字表文件")
        else:
            print("    ❌ 汉字表文件不存在")
        
        print("\n📚 HSK词汇表加载完成!")
        return hsk_vocabulary_cache, hsk_chars_cache
        
    except Exception as e:
        print(f"❌ 加载HSK词汇表失败: {e}")
        traceback.print_exc()
        return {}, {}

def initialize_predictor():
    """初始化预测器"""
    global predictor
    
    if predictor is not None:
        return predictor
    
    print("🧠 初始化预测器...")
    
    load_hsk_vocabulary()
    
    if HSKTextPredictor is None:
        print("❌ 无法初始化预测器：predictor_fixed 模块未找到")
        return None
    
    model_path = os.path.join(current_dir, 'models', 'best_optimized_model.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        model_path = None
    
    try:
        predictor = HSKTextPredictor(
            model_path=model_path,
            enable_vocab_analysis=False,
            verbose=True
        )
        print("✅ 预测器初始化成功!")
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")
        predictor = None
    
    return predictor

def initialize_enhanced_analyzer():
    """初始化增强分析器 - OpenRouter版"""
    global enhanced_analyzer, collocation_data_loaded
    
    if enhanced_analyzer is not None and collocation_data_loaded:
        return enhanced_analyzer
    
    if not ENHANCED_ANALYZER_AVAILABLE:
        print("❌ 增强分析模块不可用")
        return None
    
    try:
        base_dir = current_dir
        collocation_dir = os.path.join(base_dir, 'n')
        
        if not os.path.exists(collocation_dir):
            print(f"❌ 搭配词库目录不存在: {collocation_dir}")
            return None
        
        print(f"📂 搭配词库目录: {collocation_dir}")
        
        enhanced_analyzer = get_enhanced_analyzer(
            collocation_dir=collocation_dir,
            verbose=True
        )
        
        if enhanced_analyzer:
            print(f"✅ 使用模型: {enhanced_analyzer.model_name}")
            print(f"✅ 超时设置: {enhanced_analyzer.extended_timeout}秒")
            print(f"✅ OpenRouter API已配置")
            print(f"✅ 加载模式: 懒加载（按需加载搭配数据）")
        
        collocation_data_loaded = True
        print("✅ 增强分析器初始化成功")
        
        return enhanced_analyzer
        
    except Exception as e:
        print(f"❌ 增强分析器初始化失败: {e}")
        traceback.print_exc()
        return None

def analyze_text_features(text):
    """分析文本特征 - 基于真实HSK词汇表，返回完整的词汇列表"""
    hsk_vocabulary, hsk_chars = load_hsk_vocabulary()
    
    all_chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_char_count = len(all_chinese_chars)
    punctuation_count = len(re.findall(r'[。，；：！？、""''()（）【】《》]', text))
    sentence_count = len(re.split(r'[。！？.!?]', text))
    
    char_level_full_list = {}
    word_level_full_list = {}
    
    # ========== 1. 分析汉字级别 ==========
    for char in all_chinese_chars:
        char_level = "未知"
        for level, chars_set in hsk_chars.items():
            if char in chars_set:
                char_level = level
                break
        
        if char_level not in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']:
            if '-' in char_level:
                char_level = 'HSK' + char_level.split('-')[0]
            elif char_level.isdigit():
                char_level = 'HSK' + char_level
            else:
                char_level = '未知'
        
        if char_level not in char_level_full_list:
            char_level_full_list[char_level] = {
                'count': 0,
                'items': []
            }
        
        if char not in char_level_full_list[char_level]['items']:
            char_level_full_list[char_level]['count'] += 1
            char_level_full_list[char_level]['items'].append(char)
    
    # ========== 2. 词汇识别 ==========
    hsk_vocab_found = {}
    hsk_vocab_full_list = {}
    
    i = 0
    text_length = len(text)
    matched_positions = []
    
    while i < text_length:
        if not re.match(r'[\u4e00-\u9fff]', text[i]):
            i += 1
            continue
        
        matched_word = None
        matched_level = None
        
        for length in range(min(4, text_length - i), 1, -1):
            candidate = text[i:i+length]
            
            for level, words_set in hsk_vocabulary.items():
                if candidate in words_set:
                    matched_word = candidate
                    matched_level = level
                    break
            
            if matched_word:
                break
        
        if matched_word:
            if matched_level not in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']:
                if '-' in matched_level:
                    matched_level = 'HSK' + matched_level.split('-')[0]
                elif matched_level.isdigit():
                    matched_level = 'HSK' + matched_level
                else:
                    matched_level = '未知'
            
            if matched_level not in hsk_vocab_found:
                hsk_vocab_found[matched_level] = 0
                hsk_vocab_full_list[matched_level] = []
            
            hsk_vocab_found[matched_level] += 1
            
            if matched_word not in hsk_vocab_full_list[matched_level]:
                hsk_vocab_full_list[matched_level].append(matched_word)
            
            for pos in range(i, i + len(matched_word)):
                matched_positions.append(pos)
            
            i += len(matched_word)
        else:
            i += 1
    
    # ========== 3. 处理单字 ==========
    for i, char in enumerate(text):
        if not re.match(r'[\u4e00-\u9fff]', char):
            continue
            
        if i in matched_positions:
            continue
        
        char_level = "未知"
        for level, chars_set in hsk_chars.items():
            if char in chars_set:
                char_level = level
                break
        
        if char_level not in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']:
            if '-' in char_level:
                char_level = 'HSK' + char_level.split('-')[0]
            elif char_level.isdigit():
                char_level = 'HSK' + char_level
            else:
                char_level = '未知'
        
        if char_level not in hsk_vocab_found:
            hsk_vocab_found[char_level] = 0
            hsk_vocab_full_list[char_level] = []
        
        hsk_vocab_found[char_level] += 1
        
        if char not in hsk_vocab_full_list[char_level]:
            hsk_vocab_full_list[char_level].append(char)
    
    # ========== 4. 排序和整理结果 ==========
    for level in hsk_vocab_full_list:
        hsk_vocab_full_list[level].sort()
    
    for level in char_level_full_list:
        char_level_full_list[level]['items'].sort()
    
    estimated_level = "未知"
    hsk_levels = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']
    
    for level in reversed(hsk_levels):
        if level in char_level_full_list and char_level_full_list[level]['count'] > 0:
            estimated_level = level
            break
    
    return {
        "text_length": len(text),
        "chinese_char_count": chinese_char_count,
        "punctuation_count": punctuation_count,
        "sentence_count": sentence_count,
        "estimated_hsk_level": estimated_level,
        "char_level_full_list": char_level_full_list,
        "word_level_full_list": hsk_vocab_full_list,
        "hsk_vocabulary_found": hsk_vocab_found,
        "unknown_words": hsk_vocab_full_list.get("未知", [])
    }

def generate_colored_html_with_tooltip(text, analysis_features):
    """生成带颜色高亮和悬停提示的HTML文本"""
    hsk_vocabulary, hsk_chars = load_hsk_vocabulary()
    
    if not hsk_vocabulary or not hsk_chars:
        return f'<div style="font-family: Arial, sans-serif; line-height: 1.6; font-size: 16px;">{text}</div>'
    
    level_colors = {
        "HSK1": "#93c5fd",
        "HSK2": "#60a5fa",
        "HSK3": "#3b82f6",
        "HSK4": "#fbbf24",
        "HSK5": "#f59e0b",
        "HSK6": "#d97706",
        "HSK7-9": "#dc2626",
        "未知": "#9ca3af"
    }
    
    html_parts = []
    i = 0
    text_length = len(text)
    
    word_positions = {}
    
    while i < text_length:
        char = text[i]
        
        if not re.match(r'[\u4e00-\u9fff]', char):
            i += 1
            continue
        
        matched_word = None
        matched_level = None
        
        for length in range(min(4, text_length - i), 1, -1):
            candidate = text[i:i+length]
            
            for level, words_set in hsk_vocabulary.items():
                if candidate in words_set:
                    matched_word = candidate
                    matched_level = level
                    break
            
            if matched_word:
                break
        
        if matched_word:
            if matched_level not in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']:
                if '-' in matched_level:
                    matched_level = 'HSK' + matched_level.split('-')[0]
                elif matched_level.isdigit():
                    matched_level = 'HSK' + matched_level
            
            for pos in range(i, i + len(matched_word)):
                word_positions[pos] = (matched_word if pos == i else None, matched_level)
            
            i += len(matched_word)
        else:
            i += 1
    
    for i, char in enumerate(text):
        if i in word_positions:
            word_info = word_positions[i]
            if word_info and word_info[0]:
                word, level = word_info
                color = level_colors.get(level, "#9ca3af")
                
                style = [
                    f"color: {color}",
                    "font-weight: 600",
                    "padding: 2px 4px",
                    "border-radius: 3px",
                    "transition: all 0.2s",
                    "cursor: help",
                    "position: relative",
                    "z-index: 1"
                ]
                
                bg_opacity = {
                    "HSK1": "0.1", "HSK2": "0.15", "HSK3": "0.2",
                    "HSK4": "0.25", "HSK5": "0.3", "HSK6": "0.35",
                    "HSK7-9": "0.4", "未知": "0.1"
                }.get(level, "0.1")
                
                bg_color = color + hex(int(float(bg_opacity) * 255))[2:].zfill(2)
                style.append(f"background-color: {bg_color}")
                
                html_parts.append(
                    f'<span style="{"; ".join(style)}" data-level="{level}" data-word="{word}" title="{word} - {level}">'
                    f'{word}</span>'
                )
            continue
        
        if not re.match(r'[\u4e00-\u9fff]', char):
            if char in '，。！？；："「」『』（）【】《》〈〉、·～':
                html_parts.append(f'<span style="color: #6b7280;">{char}</span>')
            else:
                html_parts.append(char)
            continue
        
        char_level = "未知"
        for level, chars_set in hsk_chars.items():
            if char in chars_set:
                char_level = level
                break
        
        if char_level not in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']:
            if '-' in char_level:
                char_level = 'HSK' + char_level.split('-')[0]
            elif char_level.isdigit():
                char_level = 'HSK' + char_level
        
        color = level_colors.get(char_level, "#9ca3af")
        
        style = [
            f"color: {color}",
            "font-weight: 500",
            "padding: 1px 2px",
            "border-radius: 2px",
            "cursor: help",
            "transition: all 0.2s"
        ]
        
        bg_opacity = {
            "HSK1": "0.05", "HSK2": "0.08", "HSK3": "0.12",
            "HSK4": "0.15", "HSK5": "0.18", "HSK6": "0.22",
            "HSK7-9": "0.25", "未知": "0.05"
        }.get(char_level, "0.05")
        
        bg_color = color + hex(int(float(bg_opacity) * 255))[2:].zfill(2)
        style.append(f"background-color: {bg_color}")
        
        html_parts.append(
            f'<span style="{"; ".join(style)}" data-level="{char_level}" title="{char} - {char_level}">{char}</span>'
        )
    
    html_content = "".join(html_parts)
    
    css_styles = """
    <style>
        .hsk-highlight {
            transition: all 0.2s;
            cursor: help;
        }
        
        .hsk-highlight:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 10;
        }
        
        [data-level="HSK1"] { border-bottom: 1px dotted #93c5fd; }
        [data-level="HSK2"] { border-bottom: 1px dotted #60a5fa; }
        [data-level="HSK3"] { border-bottom: 1px dashed #3b82f6; }
        [data-level="HSK4"] { border-bottom: 1px solid #fbbf24; }
        [data-level="HSK5"] { border-bottom: 1px solid #f59e0b; }
        [data-level="HSK6"] { border-bottom: 2px solid #d97706; }
        [data-level="HSK7-9"] { border-bottom: 2px solid #dc2626; }
        [data-level="未知"] { border-bottom: 1px dotted #9ca3af; }
        
        .hsk-highlight:hover {
            transform: scale(1.05);
            z-index: 100;
        }
    </style>
    """
    
    full_html = f'''
    <div class="hsk-text-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.8; font-size: 18px; position: relative;">
        {css_styles}
        <div style="padding: 25px; background: white; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.05); min-height: 80px;">
            {html_content}
        </div>
    </div>
    '''
    
    return full_html

def generate_colored_text_only(text):
    """生成纯带颜色高亮的文本"""
    hsk_vocabulary, hsk_chars = load_hsk_vocabulary()
    
    if not hsk_vocabulary or not hsk_chars:
        return text
    
    level_colors = {
        "HSK1": "#93c5fd",
        "HSK2": "#60a5fa",
        "HSK3": "#3b82f6",
        "HSK4": "#fbbf24",
        "HSK5": "#f59e0b",
        "HSK6": "#d97706",
        "HSK7-9": "#dc2626",
        "未知": "#9ca3af"
    }
    
    html_parts = []
    i = 0
    text_length = len(text)
    
    while i < text_length:
        char = text[i]
        
        if not re.match(r'[\u4e00-\u9fff]', char):
            html_parts.append(char)
            i += 1
            continue
        
        matched_word = None
        matched_level = None
        
        for length in range(min(4, text_length - i), 0, -1):
            candidate = text[i:i+length]
            
            for level, words_set in hsk_vocabulary.items():
                if candidate in words_set:
                    matched_word = candidate
                    matched_level = level
                    break
            
            if matched_word:
                break
        
        if matched_word:
            color = level_colors.get(matched_level, "#9ca3af")
            style = f"color: {color}; font-weight: 600;"
            
            html_parts.append(f'<span style="{style}">{matched_word}</span>')
            i += len(matched_word)
        else:
            char_level = "未知"
            for level, chars_set in hsk_chars.items():
                if char in chars_set:
                    char_level = level
                    break
            
            color = level_colors.get(char_level, "#9ca3af")
            style = f"color: {color}; font-weight: 500;"
            
            html_parts.append(f'<span style="{style}">{char}</span>')
            i += 1
    
    return "".join(html_parts)

def generate_display_text(text, prediction, analysis_features, simple=False):
    """生成显示文本"""
    if simple:
        return [
            f"文本: {text[:100]}{'...' if len(text) > 100 else ''}",
            f"预测等级: {prediction['level']}",
            f"置信度: {prediction['confidence']:.1%}",
            f"文本长度: {len(text)} 字符",
            f"估算级别: {analysis_features['estimated_hsk_level']}"
        ]
    
    def get_color_block_html(level):
        color = get_hsk_level_color(level)
        return f'<span style="display: inline-block; width: 12px; height: 12px; background-color: {color}; border-radius: 2px; margin-right: 5px; vertical-align: middle;" title="{level}"></span>'
    
    def get_probability_bar_html(level, probability):
        color = get_hsk_level_color(level)
        bar_width = int(probability * 20)
        bar_html = f'<span style="display: inline-block; width: {bar_width}px; height: 12px; background-color: {color}; border-radius: 2px; margin-left: 8px; vertical-align: middle;" title="{level}: {probability:.1%}"></span>'
        return bar_html
    
    display_text = [
        "=" * 60,
        "📊 中文文本完整分析报告",
        "=" * 60,
        "",
        "一、基础信息",
        f"  文本长度: {len(text)} 字符",
        f"  中文字符: {analysis_features['chinese_char_count']}",
        f"  标点符号: {analysis_features['punctuation_count']}",
        f"  句子数量: {analysis_features['sentence_count']}",
        "",
        "二、预测结果", 
        f"  AI预测等级: {prediction['level']}",
        f"  预测置信度: {prediction['confidence']:.1%}",
        f"  估算HSK级别: {analysis_features['estimated_hsk_level']}",
        ""
    ]
    
    display_text.extend([
        "三、汉字级别完整列表",
        "-" * 40
    ])
    
    char_level_full_list = analysis_features.get('char_level_full_list', {})
    if char_level_full_list:
        for level in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9', '未知']:
            if level in char_level_full_list and char_level_full_list[level]['count'] > 0:
                char_data = char_level_full_list[level]
                color_block = get_color_block_html(level)
                display_text.append(f"  {color_block}{level}: {char_data['count']} 字")
                chars_list = char_data['items']
                for i in range(0, len(chars_list), 10):
                    display_text.append(f"      {''.join(chars_list[i:i+10])}")
    else:
        display_text.append("  无汉字数据")
    
    display_text.extend([
        "",
        "四、词汇级别完整列表",
        "-" * 40
    ])
    
    word_level_full_list = analysis_features.get('word_level_full_list', {})
    if word_level_full_list:
        for level in ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9', '未知']:
            if level in word_level_full_list and len(word_level_full_list[level]) > 0:
                words_list = word_level_full_list[level]
                color_block = get_color_block_html(level)
                display_text.append(f"  {color_block}{level}: {len(words_list)} 个词汇")
                for i in range(0, len(words_list), 5):
                    display_text.append(f"      {'、'.join(words_list[i:i+5])}")
    else:
        display_text.append("  无词汇数据")
    
    if 'probabilities' in prediction:
        display_text.extend([
            "",
            "五、概率分布",
            "-" * 40
        ])
        for level, prob in prediction['probabilities'].items():
            color_bar = get_probability_bar_html(level, prob)
            display_text.append(f"  {level}: {prob:.1%} {color_bar}")
    
    unknown_words = analysis_features.get('unknown_words', [])
    if unknown_words:
        display_text.extend([
            "",
            "六、生词识别",
            "-" * 40,
            f"  发现生词: {len(unknown_words)} 个"
        ])
        unknown_color_block = get_color_block_html('未知')
        for i in range(0, len(unknown_words), 10):
            if i == 0:
                display_text.append(f"      {unknown_color_block}{'、'.join(unknown_words[i:i+10])}")
            else:
                display_text.append(f"      {'、'.join(unknown_words[i:i+10])}")
    
    display_text.extend([
        "",
        "=" * 60,
        "✅ 分析完成",
        "=" * 60
    ])
    
    return display_text

# ========== 新增：增强教学建议API ==========

@app.route('/api/enhance_teaching', methods=['POST'])
def enhance_teaching():
    """专门生成详细教学建议的API - 无字符限制版本"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        requirement = data.get('requirement', '')
        
        if not text:
            return jsonify({"success": False, "error": "缺少文本内容"}), 400
        
        print(f"🎯 开始生成详细教学建议，文本长度: {len(text)}")
        
        # 生成详细教学建议提示词
        detailed_prompt = f"""作为一名国际中文教育专家，请为以下文本提供极其详细、具体、可操作的教学建议，能够直接用于课堂教学。

文本内容：
{text}

用户要求：{requirement or '请提供极其详细的教学建议'}

请提供极其详细的教学分析，包含以下内容：

一、详细教学目标（具体到每个知识点）
1. 语言知识目标（具体词汇、语法点）
2. 语言技能目标（听、说、读、写）
3. 文化意识目标
4. 学习策略目标

二、完整教学流程（60分钟课堂，每分钟都有具体安排）
【00:00-10:00】课堂导入
  - 具体活动名称：
  - 教师具体指令：（逐字写出）
  - 学生任务说明：
  - 所需材料清单：
  - 预期学生反应：
  - 教师应对策略：

【10:00-25:00】词汇教学
  - 重点词汇清单：（列出所有重要词汇）
  - 每个词汇的具体讲解：
    （1）词汇：[词语]
      - 发音讲解：（具体发音方法）
      - 词义解释：（用学生能理解的方式）
      - 使用场景：（具体例句，至少3个）
      - 常见错误：（学生可能犯的错误及纠正方法）
  - 词汇练习设计：
    （1）练习一：[名称]
      - 练习目标：
      - 具体步骤：
      - 时间分配：
      - 评估标准：
    （2）练习二：[名称]
      （同样详细）

【25:00-40:00】语法讲解
  - 语法点：[具体语法]
  - 讲解方法：（详细说明如何讲解）
  - 板书设计：（写出完整的板书）
  - 例句分析：（至少5个例句）
  - 练习设计：（写出完整的练习题）

【40:00-55:00】互动练习
  - 活动一：情景对话
    * 情景设置：（详细描述）
    * 角色分配：
    * 对话模板：（写出完整的对话）
    * 教师指导语：
    * 学生任务卡：（写出具体内容）
  - 活动二：小组讨论
    * 讨论话题：
    * 分组方式：
    * 讨论引导问题：（至少5个）
    * 汇报要求：

【55:00-60:00】总结与作业
  - 课堂总结：（写出具体的总结语）
  - 作业布置：（具体作业内容）
  - 下节课预告：

三、具体师生对话示例（至少5组完整对话）
对话一：[情景描述]
教师：[具体说的话]
学生A：[可能的回答]
学生B：[可能的回答]
教师：[反馈和引导]
（至少5轮对话）

四、教学资源推荐（必须有真实可用的链接）
1. 视频资源：[标题] - [具体网址] - [使用说明]
2. 网站资源：[网站名] - [网址] - [主要内容]
3. APP资源：[APP名称] - [下载方式] - [主要功能]
4. 打印材料：[材料名称] - [下载链接] - [使用方法]

五、学生评估方案
1. 形成性评估：（课堂中的评估方式）
2. 总结性评估：（课后评估方式）
3. 评估标准：（具体的评分标准）

六、教学反思与建议
1. 可能遇到的困难及解决方案
2. 针对不同水平学生的调整建议
3. 教学设备使用建议

请确保所有内容都是具体的、可操作的，避免空洞的理论阐述。请提供尽可能详细的内容。"""

        try:
            # 尝试使用OpenRouter API
            from openrouter_client import get_openrouter_client
            client = get_openrouter_client()
            
            response = client.generate_text(
                prompt=detailed_prompt,
                system_prompt="你是一位资深国际中文教育专家，请用中文回答，不要使用任何英文单词，所有拼音必须正确。",
                max_tokens=8000
            )
            
            if response and len(response) > 100:
                print(f"✅ 详细教学建议生成成功: {len(response)} 字符")
                
                return jsonify({
                    "success": True,
                    "result": {
                        "enhanced_content": response,
                        "content_length": len(response),
                        "model_used": "qwen3-coder-480b (OpenRouter)"
                    }
                })
            else:
                print(f"❌ 教学建议内容过短: {len(response) if response else 0} 字符")
                
        except Exception as e:
            print(f"❌ 调用OpenRouter失败: {e}")
        
        # 备用方案
        return jsonify({
            "success": True,
            "result": {
                "enhanced_content": generate_fallback_teaching_content(text),
                "content_length": 1000,
                "model_used": "fallback"
            }
        })
        
    except Exception as e:
        print(f"❌ 生成教学建议失败: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def generate_fallback_teaching_content(text):
    """生成备用教学建议"""
    return f"""# 详细教学建议（备用版本）

## 一、教学目标
1. **语言知识目标**：掌握文本中的核心词汇和基本句型
2. **语言技能目标**：能够理解文本内容并进行简单交流
3. **文化意识目标**：了解文本中体现的中国文化元素

## 二、教学流程（60分钟）

### 1. 课堂导入（10分钟）
**活动**：图片/视频导入
**教师指令**："同学们好！今天我们要学习一篇关于...的文章。先看这张图片，你们看到了什么？"
**学生任务**：观察图片并用学过的词语描述
**材料**：相关图片或短视频

### 2. 词汇教学（15分钟）
**重点词汇**：从文本中选取5-8个核心词汇
**教学步骤**：
1. 领读词汇（教师带读2遍，学生跟读3遍）
2. 词义解释（用简单中文和图片解释）
3. 例句展示（每个词汇提供2-3个例句）
4. 词汇练习（填空、配对、造句）

**具体示例**：
- 词汇："学习"
  - 发音：xué xí
  - 词义：study, learn
  - 例句：我喜欢学习汉语。/ 他在北京学习。
  - 练习：用"学习"造一个句子：____________

### 3. 阅读理解（15分钟）
**活动**：分段阅读
**步骤**：
1. 第一遍：快速阅读，找出主要人物和事件
2. 第二遍：仔细阅读，回答问题
3. 第三遍：跟读练习，注意语音语调

**问题设计**：
1. 文章主要讲了什么？
2. 文章中有几个主要人物？
3. 发生了什么事情？
4. 结果怎么样？

### 4. 对话练习（15分钟）
**情景对话示例**：
教师：假设你在学校遇到新同学，你会怎么打招呼？
学生A：你好！我叫[名字]，你叫什么名字？
学生B：你好！我叫[名字]，你学习汉语多久了？
学生A：我学习汉语一年了。
学生B：你喜欢学习汉语吗？
学生A：很喜欢，汉语很有意思。

**分组练习**：
- 两人一组，练习对话
- 每组准备1分钟
- 请2-3组上台展示

### 5. 总结与作业（5分钟）
**课堂总结**：今天我们学习了...，重点词汇有...，主要句型是...
**作业布置**：
1. 抄写今天学的词汇（每个3遍）
2. 用今天学的词汇造3个句子
3. 准备一个简单的自我介绍（下节课展示）

## 三、教学资源推荐
1. **HSK在线练习**：www.chinesetest.cn（官方练习）
2. **汉语学习视频**：CCTV汉语教学频道
3. **词汇记忆APP**：Anki记忆卡片

## 四、注意事项
1. 根据学生水平调整语速和难度
2. 多给予积极反馈和鼓励
3. 准备不同难度的练习，满足不同学生需求
4. 课堂中多用实物、图片等直观教具

此教学建议总字数：约1000字，可根据实际课堂情况调整。"""

# ========== 原始API路由（保持不变） ==========

@app.route('/')
def index():
    return jsonify({
        "service": "HSK文本分析API - OpenRouter增强优化版",
        "version": "5.0.0",
        "status": "running",
        "features": [
            "基于真实HSK词汇表的深度分析",
            "汉字级别分布统计",
            "词汇识别与生词检测",
            "原文词汇难度可视化",
            "完整的文本分析报告",
            "优化增强分析（精准主题和关键词提取）",
            "词语搭配库支持",
            "使用OpenRouter Qwen3 Coder 480B模型",
            "180秒超时支持",
            "无需本地Ollama服务"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        hsk_vocabulary, hsk_chars = load_hsk_vocabulary()
        
        pred = initialize_predictor()
        model_loaded = pred is not None
        
        enhanced = initialize_enhanced_analyzer()
        enhanced_loaded = enhanced is not None
        
        vocab_count = sum(len(v) for v in hsk_vocabulary.values())
        char_count = sum(len(v) for v in hsk_chars.values())
        
        llm_local_available = LOCAL_LLM_AVAILABLE
        
        # 检查OpenRouter可用性
        openrouter_available = False
        try:
            from openrouter_client import get_openrouter_client
            client = get_openrouter_client(verbose=False)
            quota = client.check_quota()
            openrouter_available = quota.get('success', False)
        except:
            openrouter_available = False
        
        return jsonify({
            "status": "healthy",
            "model_loaded": model_loaded,
            "enhanced_analyzer_loaded": enhanced_loaded,
            "llm_local_available": llm_local_available,
            "openrouter_available": openrouter_available,
            "api_type": "OpenRouter (qwen3-coder-480b)",
            "vocab_analysis_enabled": vocab_count > 0,
            "vocabulary_count": vocab_count,
            "character_count": char_count,
            "timeout_setting": "180秒",
            "message": "API服务正常运行（使用OpenRouter云端API）" if model_loaded else "API服务运行中（预测器未加载）",
            "timestamp": pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)[:200],
            "message": "API服务异常"
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """快速预测API - 与完整分析使用完全相同的预测器实例和文本预处理"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "文本不能为空"}), 400
        
        print(f"🚀 快速分析请求: {text[:50]}...")
        
        # ✅ 关键修复：使用与完整分析完全相同的预测器实例
        predictor_instance = initialize_predictor()
        
        if predictor_instance:
            # ✅ 使用与完整分析完全相同的预测方法
            result = predictor_instance.predict(text)
            print(f"✅ 预测完成: {result['level']} ({result['confidence']:.1%})")
        else:
            # 备用规则（预测器未加载时）
            print("⚠️ 预测器未加载，使用备用规则")
            result = fallback_predict(text)
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        print(f"❌ 预测错误: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)[:200]}), 500


def fallback_predict(text):
    """备用预测规则（当预测器不可用时）"""
    text_length = len(text)
    
    if text_length < 10:
        level = "HSK1"
        confidence = 0.8
    elif text_length < 20:
        level = "HSK2"
        confidence = 0.75
    elif text_length < 40:
        level = "HSK3"
        confidence = 0.7
    elif text_length < 60:
        level = "HSK4"
        confidence = 0.65
    elif text_length < 80:
        level = "HSK5"
        confidence = 0.6
    elif text_length < 120:
        level = "HSK6"
        confidence = 0.55
    else:
        level = "HSK7-9"
        confidence = 0.5
    
    return {
        "level": level,
        "level_key": level.replace("HSK", "").replace("-9", ""),
        "confidence": confidence,
        "probabilities": {
            "HSK1": 0.1, "HSK2": 0.15, "HSK3": 0.2,
            "HSK4": 0.15, "HSK5": 0.1, "HSK6": 0.1, "HSK7-9": 0.2
        },
        "text": text
    }

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """完整分析API"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "缺少text参数"}), 400
        
        text = data['text']
        simple = data.get('simple', False)
        
        print(f"📊 开始完整分析文本: {text[:50]}...")
        
        predictor = initialize_predictor()
        if predictor is None:
            prediction_result = {
                "level": "HSK3",
                "level_key": "3",
                "confidence": 0.7,
                "probabilities": {
                    "HSK1": 0.1, "HSK2": 0.2, "HSK3": 0.3,
                    "HSK4": 0.15, "HSK5": 0.1, "HSK6": 0.1, "HSK7-9": 0.05
                },
                "text": text
            }
            print("⚠️ 使用模拟预测结果")
        else:
            prediction_result = predictor.predict(text)
            print(f"✅ 预测完成: {prediction_result['level']} ({prediction_result['confidence']:.1%})")
        
        prediction_result['probabilities'] = {k: float(v) for k, v in prediction_result['probabilities'].items()}
        prediction_result['confidence'] = float(prediction_result['confidence'])
        
        analysis_features = analyze_text_features(text)
        print(f"✅ 文本特征分析完成")
        
        display_text = generate_display_text(text, prediction_result, analysis_features, simple)
        
        colored_html = generate_colored_html_with_tooltip(text, analysis_features)
        
        analysis_result = {
            "prediction": prediction_result,
            "analysis": {
                "features": analysis_features,
                "char_level_full_list": analysis_features['char_level_full_list'],
                "word_level_full_list": analysis_features['word_level_full_list'],
                "vocabulary_loaded": bool(hsk_vocabulary_cache),
                "characters_loaded": bool(hsk_chars_cache),
                "timestamp": pd.Timestamp.now().isoformat(),
                "analysis_type": "simple" if simple else "full"
            },
            "colored_text": colored_html,
            "display_text": display_text,
            "unknown_tokens": analysis_features['unknown_words']
        }
        
        print(f"✅ 完整分析完成: {prediction_result['level']}, 置信度: {prediction_result['confidence']:.1%}")
        
        return jsonify({
            "success": True,
            "result": analysis_result
        })
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"分析失败: {str(e)[:200]}"
        }), 500

@app.route('/api/enhanced_analyze', methods=['POST'])
def enhanced_analyze():
    """增强分析 - OpenRouter版，基于N+1原则提取生词"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "缺少text参数"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"success": False, "error": "文本不能为空"}), 400
        
        print(f"🚀 开始增强分析（OpenRouter版），文本长度: {len(text)}")
        
        start_time = time.time()
        
        print("📊 第一步：获取完整分析结果...")
        
        predictor_instance = initialize_predictor()
        if predictor_instance is None:
            prediction_result = {
                "level": "HSK3",
                "level_key": "3",
                "confidence": 0.7,
                "probabilities": {
                    "HSK1": 0.1, "HSK2": 0.2, "HSK3": 0.3,
                    "HSK4": 0.15, "HSK5": 0.1, "HSK6": 0.1, "HSK7-9": 0.05
                },
                "text": text
            }
            print("⚠️ 使用默认预测结果")
        else:
            prediction_result = predictor_instance.predict(text)
            print(f"✅ 预测完成: {prediction_result['level']} ({prediction_result['confidence']:.1%})")
        
        predicted_level = prediction_result['level']
        
        analysis_features = analyze_text_features(text)
        print(f"✅ 文本特征分析完成")
        print(f"  估算HSK级别: {analysis_features['estimated_hsk_level']}")
        
        print("📚 第二步：准备HSK数据...")
        hsk_char_data = {}
        hsk_vocab_data = {}
        
        if hsk_chars_cache:
            for level, chars_set in hsk_chars_cache.items():
                for char in chars_set:
                    hsk_char_data[char] = level
        
        if hsk_vocabulary_cache:
            for level, words_set in hsk_vocabulary_cache.items():
                for word in words_set:
                    hsk_vocab_data[word] = level
        
        print(f"  汉字数据: {len(hsk_char_data)} 条")
        print(f"  词汇数据: {len(hsk_vocab_data)} 条")
        
        print("🎯 第三步：初始化增强分析器...")
        analyzer = initialize_enhanced_analyzer()
        
        if analyzer is None:
            print("⚠️ 增强分析器初始化失败，使用备用分析")
            # 使用备用分析
            enhanced_result = generate_fallback_enhanced_analysis(text, prediction_result, analysis_features)
        else:
            # 使用增强分析器分析文本内容（主题和关键词）
            print("📝 第三步：分析文本内容（使用OpenRouter）...")
            text_analysis = analyzer.analyze_text_content(text)
            theme = text_analysis.get('theme', '中文教学文本')
            keywords = text_analysis.get('keywords', [])
            print(f"  ✅ 主题: {theme}")
            print(f"  ✅ 关键词: {keywords}")
            
            # N+1生词提取
            print("📖 第四步：基于N+1原则准确提取生词...")

            level_order = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']
            if predicted_level in level_order:
                current_index = level_order.index(predicted_level)
                target_index = min(current_index + 1, len(level_order) - 1)
                target_level = level_order[target_index]
                print(f"  📊 预测等级: {predicted_level}, N+1目标等级: {target_level}")
            else:
                target_level = 'HSK4'
                predicted_level = 'HSK3'

            full_analysis = analyze_text_features(text)
            word_level_full_list = full_analysis.get('word_level_full_list', {})

            difficult_words = []

            for level, words in word_level_full_list.items():
                if not words:
                    continue
        
                is_target_or_higher = False
                reason = ""
                if level == target_level:
                    is_target_or_higher = True
                    reason = f"N+1目标级别: {target_level}"
                elif level in level_order and target_level in level_order:
                    level_index = level_order.index(level)
                    target_index = level_order.index(target_level)
                    if level_index > target_index:
                        is_target_or_higher = True
                        reason = f"超纲词汇: {level}（目标: {target_level}）"
                elif level == '未知':
                    is_target_or_higher = True
                    reason = "未在HSK词汇表中"
        
                if is_target_or_higher:
                    for word in words:
                        freq = text.count(word)
                        if freq > 0:
                            difficult_words.append({
                                'word': word,
                                'level': level,
                                'frequency': freq,
                                'word_type': guess_word_type(word),
                                'reason': reason,
                                'is_target_level': level == target_level
                            })

            target_level_words = [w for w in difficult_words if w['is_target_level']]
            if len(target_level_words) < 2 and target_level in word_level_full_list:
                potential_words = word_level_full_list.get(target_level, [])
                for word in potential_words[:3]:
                    if word in text:
                        if not any(w['word'] == word for w in difficult_words):
                            freq = text.count(word)
                            difficult_words.append({
                                'word': word,
                                'level': target_level,
                                'frequency': freq,
                                'word_type': guess_word_type(word),
                                'reason': f"N+1目标级别: {target_level}（补充）",
                                'is_target_level': True
                            })

            difficult_words.sort(key=lambda x: (
                not x['is_target_level'],
                -x['frequency']
            ))

            difficult_words = difficult_words[:10]

            print(f"  ✅ 找到生词: {len(difficult_words)}个")
            if difficult_words:
                for word_info in difficult_words[:min(5, len(difficult_words))]:
                    print(f"      {word_info['word']} ({word_info['level']}) - {word_info['reason']}")
        
                target_count = len([w for w in difficult_words if w['is_target_level']])
                other_count = len([w for w in difficult_words if not w['is_target_level']])
                print(f"      目标级别({target_level}): {target_count}个, 超纲词汇: {other_count}个")
            
            # 查找搭配词
            print("🔧 第五步：查找搭配词...")
            collocations = {}
            if analyzer and hasattr(analyzer, 'find_collocations_for_text'):
                target_words_for_collocation = [w['word'] for w in difficult_words if len(w['word']) >= 2]
                # 这个方法可能不存在，用try包装
                try:
                    collocations = analyzer.find_collocations_for_text(text, 
                        [{'word': w} for w in target_words_for_collocation])
                    print(f"  ✅ 找到搭配: {len(collocations)}个词语")
                except:
                    print(f"  ⚠️ 查找搭配失败，跳过")
                    collocations = {}
            
            # 生成详细教学建议
            print("📝 第六步：生成详细的教学分析（使用OpenRouter）...")
            teaching_result = analyzer.generate_detailed_teaching_suggestions_with_llm(
                text, theme, keywords, difficult_words, predicted_level
            )
            
            teaching_analysis = teaching_result.get('raw_response', '')
            structured_suggestions = teaching_result.get('structured', [])
            
            # 构建增强分析结果
            enhanced_result = {
                "theme_keywords": {
                    "theme": theme,
                    "keywords": keywords,
                    "keywords_extracted": len(keywords) > 0
                },
                "teaching_analysis": {
                    "summary": teaching_analysis[:500] + "..." if len(teaching_analysis) > 500 else teaching_analysis,
                    "raw": teaching_analysis,
                    "predicted_level": predicted_level,
                    "confidence": prediction_result['confidence'],
                    "estimated_level": analysis_features['estimated_hsk_level'],
                    "n_plus_1_level": target_level,
                    "n_plus_1_principle": f"{predicted_level} → {target_level}"
                },
                "vocabulary_analysis": {
                    "difficult_words": difficult_words,
                    "total_difficult_words": len(difficult_words),
                    "target_level_words": target_count,
                    "other_difficult_words": other_count,
                    "note": f"基于N+1原则，重点教学{target_level}级别词汇"
                },
                "collocation_analysis": {
                    "found_collocations": len(collocations),
                    "collocation_details": collocations,
                    "collocation_words": list(collocations.keys())[:10] if collocations else []
                },
                "practical_suggestions": [s.get('content', '') if isinstance(s, dict) else s for s in structured_suggestions[:10]],
                "structured_suggestions": structured_suggestions[:10],
                "text_statistics": {
                    "length": len(text),
                    "sentence_count": analysis_features['sentence_count'],
                    "chinese_char_count": analysis_features['chinese_char_count'],
                    "estimated_hsk_level": analysis_features['estimated_hsk_level']
                }
            }
        
        # 生成可视化结果
        print("🎨 第七步：生成可视化结果...")
        colored_html = generate_colored_html_with_tooltip(text, analysis_features)
        
        # 构建最终结果
        print("📋 第八步：构建最终结果...")
        end_time = time.time()
        total_time = end_time - start_time
        
        final_result = {
            "prediction": prediction_result,
            "analysis": analysis_features,
            "colored_text": colored_html,
            "enhanced_analysis": enhanced_result if 'enhanced_result' in locals() else generate_fallback_enhanced_analysis(text, prediction_result, analysis_features),
            "performance": {
                "total_time": total_time,
                "theme_extraction": True,
                "keywords_extraction": True,
                "teaching_analysis_generated": True,
                "difficult_words_found": len(difficult_words) if 'difficult_words' in locals() else 0,
                "collocations_found": len(collocations) if 'collocations' in locals() else 0,
                "model_used": "qwen3-coder-480b (OpenRouter)",
                "timeout_setting": 180,
                "llm_calls": 3,
                "response_length": len(teaching_analysis) if 'teaching_analysis' in locals() else 0
            }
        }
        
        response_data = {
            "success": True,
            "result": final_result
        }
        
        print(f"✅ 增强分析完成，总耗时: {total_time:.1f}秒")
        if 'theme' in locals():
            print(f"  最终主题: {theme}")
        if 'keywords' in locals():
            print(f"  关键词: {keywords}")
        print(f"  N+1目标: {predicted_level}→{target_level if 'target_level' in locals() else 'HSK3'}")
        print(f"  生词数量: {len(difficult_words) if 'difficult_words' in locals() else 0}个")
        print(f"  使用模型: qwen3-coder-480b (OpenRouter)")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ 增强分析异常: {str(e)}")
        traceback.print_exc()
        
        try:
            basic_result = analyze_text_features(text)
            colored_html = generate_colored_html_with_tooltip(text, basic_result)
            
            return jsonify({
                "success": False,
                "error": f"增强分析失败: {str(e)[:100]}",
                "fallback": {
                    "basic_analysis": basic_result,
                    "colored_text": colored_html
                }
            }), 500
        except:
            return jsonify({
                "success": False,
                "error": f"增强分析失败: {str(e)[:100]}"
            }), 500

# ========== 其他辅助API路由 ==========

@app.route('/api/get_collocation', methods=['POST'])
def get_collocation():
    """获取单个词语的搭配数据"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({
                "success": False,
                "error": "请输入要检索的词语"
            })
        
        if not re.match(r'^[\u4e00-\u9fff]{2,4}$', word):
            return jsonify({
                "success": False,
                "error": "请输入2-4个中文字符的词语"
            })
        
        print(f"\n🔍 ======= 开始解析 {word} =======")
        
        base_dir = current_dir
        collocation_dir = os.path.join(base_dir, 'n')
        file_path = os.path.join(collocation_dir, f"{word}.txt")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return jsonify({
                "success": True,
                "result": get_backup_collocations_result(word)
            })
        
        raw_content = ""
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'cp936', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    raw_content = f.read()
                    if word in raw_content:
                        print(f"✅ 使用编码 {encoding} 成功读取文件")
                        break
            except:
                continue
        
        if not raw_content:
            print(f"❌ 无法读取文件: {file_path}")
            return jsonify({
                "success": True,
                "result": get_backup_collocations_result(word)
            })
        
        print(f"📄 文件长度: {len(raw_content)} 字符")
        
        lines = raw_content.split('\n')
        print(f"📝 共 {len(lines)} 行")
        
        collocations = []
        total_examples = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            print(f"\n处理第{line_num+1}行: {line[:80] if len(line) > 80 else line}")
            
            pattern = rf'(\S+?)\s*_{word}\s*'
            matches = list(re.finditer(pattern, line))
            
            if not matches:
                pattern2 = rf'([^{word}_]+)_{word}'
                matches = list(re.finditer(pattern2, line))
            
            if not matches:
                if word in line:
                    word_pos = line.find(word)
                    if word_pos > 0:
                        left_start = max(0, word_pos - 10)
                        left_part = line[left_start:word_pos]
                        left_words = re.findall(r'([\u4e00-\u9fff]+)\s*$', left_part)
                        if left_words:
                            left_word = left_words[-1]
                            matches = [type('Match', (), {
                                'group': lambda self, x=0: left_word if x == 1 else f"{left_word}_{word}",
                                'start': lambda: word_pos - len(left_word),
                                'end': lambda: word_pos + len(word)
                            })()]
                            print(f"  手动提取左词: {left_word}")
            
            if not matches:
                print(f"  未找到搭配模式")
                continue
            
            print(f"  找到{len(matches)}个搭配模式")
            
            for i, match in enumerate(matches):
                try:
                    left_word = match.group(1)
                    pattern_str = f"{left_word}_{word}"
                    
                    start_pos = match.end()
                    end_pos = matches[i+1].start() if i+1 < len(matches) else len(line)
                    
                    content_for_this_pattern = line[start_pos:end_pos].strip()
                    
                    print(f"  [{i+1}] 搭配模式: {pattern_str}")
                    print(f"     对应内容长度: {len(content_for_this_pattern)}")
                    
                    br_positions = []
                    search_start = 0
                    while True:
                        pos = content_for_this_pattern.find('<br', search_start)
                        if pos == -1:
                            break
                        br_positions.append(pos)
                        search_start = pos + 1
                    
                    examples = []
                    
                    if br_positions:
                        for j, br_pos in enumerate(br_positions):
                            example_start = 0 if j == 0 else br_positions[j-1] + 1
                            example_end = br_pos
                            example_text = content_for_this_pattern[example_start:example_end].strip()
                            
                            example_text = re.sub(r'<[^>]+>', '', example_text)
                            example_text = re.sub(r'\s+', ' ', example_text).strip()
                            
                            if example_text and len(example_text) > 3 and word in example_text:
                                highlighted = example_text.replace(word, f"<font color='red'><b>{word}</b></font>")
                                examples.append(highlighted)
                                print(f"      提取例句{j+1}: {example_text[:40]}...")
                    else:
                        word_positions = []
                        search_start = 0
                        while True:
                            pos = content_for_this_pattern.find(word, search_start)
                            if pos == -1:
                                break
                            word_positions.append(pos)
                            search_start = pos + len(word)
                        
                        if word_positions:
                            for j, word_pos in enumerate(word_positions):
                                example_start = max(0, word_pos - 20)
                                example_end = word_positions[j+1] if j+1 < len(word_positions) else len(content_for_this_pattern)
                                example_end = min(len(content_for_this_pattern), example_end + 50)
                                
                                example_text = content_for_this_pattern[example_start:example_end].strip()
                                example_text = re.sub(r'<[^>]+>', '', example_text)
                                example_text = re.sub(r'\s+', ' ', example_text).strip()
                                
                                if example_text and len(example_text) > 3 and word in example_text:
                                    highlighted = example_text.replace(word, f"<font color='red'><b>{word}</b></font>")
                                    examples.append(highlighted)
                                    print(f"      按目标词分割例句{j+1}: {example_text[:40]}...")
                    
                    unique_examples = []
                    seen = set()
                    for ex in examples:
                        simple_ex = re.sub(r'<[^>]+>', '', ex).strip()
                        if simple_ex and simple_ex not in seen:
                            seen.add(simple_ex)
                            unique_examples.append(ex)
                    
                    examples = unique_examples
                    
                    if examples:
                        numbered_examples = []
                        for idx, example in enumerate(examples, 1):
                            numbered_examples.append(f"例句{idx}: {example}")
                        
                        collocations.append({
                            "left": left_word,
                            "right": "",
                            "pattern": pattern_str,
                            "example": numbered_examples[0] if numbered_examples else "",
                            "all_examples": numbered_examples
                        })
                        
                        total_examples += len(examples)
                        print(f"      提取到{len(examples)}个例句")
                    else:
                        print(f"      未提取到例句")
                        
                except Exception as e:
                    print(f"      处理搭配模式时出错: {e}")
                    continue
        
        print(f"\n✅ 解析完成: {len(collocations)}个搭配模式，{total_examples}个例句")
        
        if not collocations:
            print("⚠️ 未找到任何搭配模式，使用备用数据")
            return jsonify({
                "success": True,
                "result": get_backup_collocations_result(word)
            })
        
        all_examples = []
        for colloc in collocations:
            if 'all_examples' in colloc:
                all_examples.extend(colloc['all_examples'])
        
        return jsonify({
            "success": True,
            "result": {
                "word": word,
                "collocations": collocations,
                "examples": all_examples,
                "total_collocations": len(collocations),
                "total_examples": len(all_examples),
                "note": "精准解析版"
            }
        })
            
    except Exception as e:
        print(f"❌ 获取搭配失败: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def get_backup_collocations_result(word):
    """获取备用搭配数据结果"""
    backup_collocations = get_backup_collocations(word)
    all_examples = []
    for colloc in backup_collocations:
        if 'all_examples' in colloc:
            all_examples.extend(colloc['all_examples'])
    
    return {
        "word": word,
        "collocations": backup_collocations,
        "examples": all_examples,
        "total_collocations": len(backup_collocations),
        "total_examples": len(all_examples),
        "note": "使用备用数据"
    }

def get_backup_collocations(word):
    """获取备用搭配数据"""
    backup_data = {
        "学习": [
            {
                "left": "认真",
                "right": "",
                "pattern": "认真_学习",
                "example": "例句1: 他每天都很认真<font color='red'><b>学习</b></font>。",
                "all_examples": [
                    "例句1: 他每天都很认真<font color='red'><b>学习</b></font>。",
                    "例句2: 为了考试，他努力<font color='red'><b>学习</b></font>。",
                    "例句3: 我喜欢<font color='red'><b>学习</b></font>新的知识。"
                ]
            },
            {
                "left": "",
                "right": "汉语",
                "pattern": "_学习汉语",
                "example": "例句1: 我在北京<font color='red'><b>学习</b></font>汉语。",
                "all_examples": [
                    "例句1: 我在北京<font color='red'><b>学习</b></font>汉语。",
                    "例句2: <font color='red'><b>学习</b></font>汉语需要多练习。"
                ]
            }
        ],
        "工作": [
            {
                "left": "努力",
                "right": "",
                "pattern": "努力_工作",
                "example": "例句1: 他<font color='red'><b>工作</b></font>很努力。",
                "all_examples": [
                    "例句1: 他<font color='red'><b>工作</b></font>很努力。",
                    "例句2: 为了项目，大家都很努力<font color='red'><b>工作</b></font>。"
                ]
            }
        ],
        "阿公": [
            {
                "left": "猫头鹰",
                "right": "",
                "pattern": "猫头鹰_阿公",
                "example": "例句1: 猫头鹰<font color='red'><b>阿公</b></font>，真对不起可你干嘛大白天睡觉呀。",
                "all_examples": [
                    "例句1: 猫头鹰<font color='red'><b>阿公</b></font>，真对不起可你干嘛大白天睡觉呀。",
                    "例句2: 在猫头鹰<font color='red'><b>阿公</b></font>头顶上方的树枝条儿上，倒挂着一只蝙蝠。"
                ]
            },
            {
                "left": "老",
                "right": "",
                "pattern": "老_阿公",
                "example": "例句1: 老<font color='red'><b>阿公</b></font>动员不少老阿公、老阿嬷。",
                "all_examples": [
                    "例句1: 老<font color='red'><b>阿公</b></font>动员不少老阿公、老阿嬷。",
                    "例句2: 嘿嘿，老<font color='red'><b>阿公</b></font>，进我房里喝茶去。"
                ]
            }
        ]
    }
    
    if word in backup_data:
        return backup_data[word]
    
    return [
        {
            "left": "使用",
            "right": "",
            "pattern": f"使用_{word}",
            "example": f"例句1: 这个词语'{word}'在日常交流中经常<font color='red'><b>使用</b></font>。",
            "all_examples": [
                f"例句1: 这个词语'{word}'在日常交流中经常<font color='red'><b>使用</b></font>。",
                f"例句2: 学会正确<font color='red'><b>使用</b></font>'{word}'这个词语。"
            ]
        },
        {
            "left": "学习",
            "right": "",
            "pattern": f"学习_{word}",
            "example": f"例句1: <font color='red'><b>学习</b></font>'{word}'这个词语对提高汉语水平很有帮助。",
            "all_examples": [
                f"例句1: <font color='red'><b>学习</b></font>'{word}'这个词语对提高汉语水平很有帮助。",
                f"例句2: 我正在<font color='red'><b>学习</b></font>如何使用'{word}'。"
            ]
        }
    ]

@app.route('/api/test_encoding', methods=['POST'])
def test_encoding():
    """测试文件编码"""
    try:
        data = request.get_json()
        word = data.get('word', '学习')
        
        base_dir = current_dir
        collocation_dir = os.path.join(base_dir, 'n')
        file_path = os.path.join(collocation_dir, f"{word}.txt")
        
        result = {
            "file_exists": os.path.exists(file_path),
            "file_path": file_path,
            "encodings_tested": []
        }
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-8-sig', 'big5', 'latin-1']
            
            for encoding in encodings:
                try:
                    decoded = raw_data.decode(encoding, errors='strict')
                    result["encodings_tested"].append({
                        "encoding": encoding,
                        "success": True,
                        "preview": decoded[:100] if len(decoded) > 100 else decoded
                    })
                except:
                    result["encodings_tested"].append({
                        "encoding": encoding,
                        "success": False
                    })
            
            try:
                import chardet
                detection = chardet.detect(raw_data)
                result["chardet_result"] = {
                    "encoding": detection.get('encoding'),
                    "confidence": detection.get('confidence'),
                    "language": detection.get('language')
                }
            except:
                result["chardet_result"] = "chardet未安装"
        
        return jsonify({"success": True, "result": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/color_text', methods=['POST'])
def color_text():
    """仅对文本进行颜色高亮处理"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "缺少text参数"}), 400
        
        text = data['text']
        
        colored_text = generate_colored_text_only(text)
        
        return jsonify({
            "success": True,
            "result": {
                "original_text": text,
                "colored_text": colored_text,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"❌ 颜色高亮失败: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"颜色高亮失败: {str(e)[:200]}"
        }), 500

@app.route('/api/simple_predict', methods=['POST'])
def simple_predict():
    """简化预测"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "缺少text参数"}), 400
        
        text = data['text']
        
        if len(text) < 10:
            level = "HSK1"
            confidence = 0.8
        elif len(text) < 20:
            level = "HSK2"
            confidence = 0.75
        elif len(text) < 40:
            level = "HSK3"
            confidence = 0.7
        elif len(text) < 60:
            level = "HSK4"
            confidence = 0.65
        elif len(text) < 80:
            level = "HSK5"
            confidence = 0.6
        elif len(text) < 120:
            level = "HSK6"
            confidence = 0.55
        else:
            level = "HSK7-9"
            confidence = 0.5
        
        result = {
            "level": level,
            "level_key": level.replace("HSK", "").replace("-9", ""),
            "confidence": confidence,
            "probabilities": {
                "HSK1": 0.1, "HSK2": 0.15, "HSK3": 0.2, 
                "HSK4": 0.15, "HSK5": 0.1, "HSK6": 0.1, "HSK7-9": 0.2
            },
            "text": text
        }
        
        return jsonify({
            "success": True,
            "result": result,
            "note": "使用简单规则预测"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/test_enhanced', methods=['GET'])
def test_enhanced():
    """测试增强分析功能 - OpenRouter版"""
    try:
        analyzer = initialize_enhanced_analyzer()
        
        # 测试OpenRouter连接
        openrouter_available = False
        try:
            from openrouter_client import get_openrouter_client
            client = get_openrouter_client(verbose=False)
            quota = client.check_quota()
            openrouter_available = quota.get('success', False)
        except:
            openrouter_available = False
        
        return jsonify({
            "success": True,
            "enhanced_analyzer_available": analyzer is not None,
            "openrouter_available": openrouter_available,
            "model_in_use": analyzer.model_name if analyzer else "qwen/qwen3-coder-480b-a35b:free",
            "collocation_data_loaded": collocation_data_loaded,
            "collocation_count": len(analyzer.collocation_data) if analyzer and hasattr(analyzer, 'collocation_data') else 0,
            "timeout_setting": analyzer.extended_timeout if analyzer else 180,
            "api_type": "OpenRouter (云端API，无需本地Ollama)",
            "message": "增强分析功能测试完成，使用OpenRouter API"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)[:200],
            "message": "测试失败"
        }), 500

def initialize_custom_tokenizer():
    """初始化自定义分词器"""
    global custom_tokenizer
    if custom_tokenizer is None:
        custom_tokenizer = load_custom_tokenizer()
    return custom_tokenizer

@app.route('/api/find_collocation_file', methods=['POST', 'OPTIONS'])
def find_collocation_file():
    """遍历所有子文件夹查找匹配的搭配文件 - 每个搭配模式独立，例句按<br/>分割"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.json
        word = data.get('word')
        collocation_type = data.get('type')  # 'left' 或 'right'
        base_path = data.get('base_path')
        
        if not all([word, collocation_type, base_path]):
            return jsonify({'success': False, 'error': '缺少参数'})
        
        # 后台打印日志（仅控制台可见）
        print(f"开始查找: {base_path} 下的所有子文件夹，目标文件: {word}.txt (类型: {collocation_type})")
        
        found_file = None
        found_content = None
        
        if not os.path.exists(base_path):
            return jsonify({'success': False, 'error': f'路径不存在: {base_path}'})
        
        try:
            subfolders = [f for f in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, f))]
        except Exception as e:
            return jsonify({'success': False, 'error': f'无法读取子文件夹: {str(e)}'})
        
        encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-8-sig', 'latin-1']
        
        for subfolder in sorted(subfolders):
            folder_path = os.path.join(base_path, subfolder)
            file_path = os.path.join(folder_path, f"{word}.txt")
            
            if os.path.exists(file_path):
                print(f"✅ 找到文件: {file_path}")
                
                content = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        print(f"   使用 {encoding} 编码读取成功")
                        break
                    except:
                        continue
                
                if content is not None:
                    found_file = file_path
                    found_content = content
                    break
        
        if found_content:
            # ========== 修复的解析逻辑：按<br/>分割例句 ==========
            collocations = []
            lines = found_content.split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                i += 1
                
                if not line:
                    continue
                
                # 检查是否是搭配模式行
                pattern = None
                
                if collocation_type == 'left' and '_' + word in line:
                    # 左搭配模式：xxx_图书馆
                    match = re.search(r'([\u4e00-\u9fff]+)_' + re.escape(word), line)
                    if match:
                        left_word = match.group(1)
                        pattern = f"{left_word}_{word}"
                        
                        # 收集该模式的所有例句
                        examples = []
                        
                        # 处理当前行中可能包含的第一个例句
                        line_parts = line.split('_' + word, 1)
                        if len(line_parts) > 1 and line_parts[1].strip():
                            first_example = line_parts[1].strip()
                            # 按<br/>分割这一行中的例句
                            if '<br/>' in first_example:
                                parts = first_example.split('<br/>')
                                for part in parts:
                                    part = part.strip()
                                    if part and len(part) > 5 and word in part:
                                        # 清理HTML标签，但保留搜索词高亮
                                        clean_part = re.sub(r'<[^>]+>', '', part)
                                        highlighted = clean_part.replace(word, f"<span class='highlight-red'>{word}</span>")
                                        examples.append(highlighted)
                            else:
                                if first_example and len(first_example) > 5 and word in first_example:
                                    clean_first = re.sub(r'<[^>]+>', '', first_example)
                                    highlighted = clean_first.replace(word, f"<span class='highlight-red'>{word}</span>")
                                    examples.append(highlighted)
                        
                        # 继续读取后续行，直到遇到下一个模式
                        while i < len(lines):
                            next_line = lines[i].strip()
                            if not next_line:
                                i += 1
                                continue
                            
                            # 如果下一行是新的模式，跳出循环
                            if (collocation_type == 'left' and '_' + word in next_line) or \
                               (collocation_type == 'right' and word + '_' in next_line):
                                break
                            
                            # 否则，这一行是当前模式的例句行
                            if next_line and len(next_line) > 5:
                                # 按<br/>分割这一行中的多个例句
                                if '<br/>' in next_line:
                                    parts = next_line.split('<br/>')
                                    for part in parts:
                                        part = part.strip()
                                        if part and len(part) > 5 and word in part:
                                            clean_part = re.sub(r'<[^>]+>', '', part)
                                            highlighted = clean_part.replace(word, f"<span class='highlight-red'>{word}</span>")
                                            examples.append(highlighted)
                                else:
                                    if word in next_line:
                                        clean_next = re.sub(r'<[^>]+>', '', next_line)
                                        highlighted = clean_next.replace(word, f"<span class='highlight-red'>{word}</span>")
                                        examples.append(highlighted)
                            
                            i += 1
                        
                        # 去重
                        unique_examples = []
                        seen = set()
                        for ex in examples:
                            plain_ex = re.sub(r'<[^>]+>', '', ex)
                            if plain_ex not in seen:
                                seen.add(plain_ex)
                                unique_examples.append(ex)
                        
                        if unique_examples:
                            collocations.append({
                                'pattern': pattern,
                                'all_examples': unique_examples
                            })
                            print(f"    添加模式: {pattern} ({len(unique_examples)}个例句)")
                            for idx, ex in enumerate(unique_examples[:3]):
                                print(f"      例句{idx+1}: {ex[:50]}...")
                
                elif collocation_type == 'right' and word + '_' in line:
                    # 右搭配模式：图书馆_xxx
                    match = re.search(re.escape(word) + r'_([\u4e00-\u9fff]+)', line)
                    if match:
                        right_word = match.group(1)
                        pattern = f"{word}_{right_word}"
                        
                        # 收集该模式的所有例句
                        examples = []
                        
                        # 处理当前行中可能包含的第一个例句
                        line_parts = line.split(word + '_', 1)
                        if len(line_parts) > 1:
                            after_right = line_parts[1]
                            right_match = re.search(r'^([\u4e00-\u9fff]+)\s*(.*)', after_right)
                            if right_match:
                                first_example = right_match.group(2).strip()
                                # 按<br/>分割这一行中的例句
                                if '<br/>' in first_example:
                                    parts = first_example.split('<br/>')
                                    for part in parts:
                                        part = part.strip()
                                        if part and len(part) > 5 and word in part:
                                            clean_part = re.sub(r'<[^>]+>', '', part)
                                            highlighted = clean_part.replace(word, f"<span class='highlight-red'>{word}</span>")
                                            examples.append(highlighted)
                                else:
                                    if first_example and len(first_example) > 5 and word in first_example:
                                        clean_first = re.sub(r'<[^>]+>', '', first_example)
                                        highlighted = clean_first.replace(word, f"<span class='highlight-red'>{word}</span>")
                                        examples.append(highlighted)
                        
                        # 继续读取后续行，直到遇到下一个模式
                        while i < len(lines):
                            next_line = lines[i].strip()
                            if not next_line:
                                i += 1
                                continue
                            
                            # 如果下一行是新的模式，跳出循环
                            if (collocation_type == 'left' and '_' + word in next_line) or \
                               (collocation_type == 'right' and word + '_' in next_line):
                                break
                            
                            # 否则，这一行是当前模式的例句行
                            if next_line and len(next_line) > 5:
                                # 按<br/>分割这一行中的多个例句
                                if '<br/>' in next_line:
                                    parts = next_line.split('<br/>')
                                    for part in parts:
                                        part = part.strip()
                                        if part and len(part) > 5 and word in part:
                                            clean_part = re.sub(r'<[^>]+>', '', part)
                                            highlighted = clean_part.replace(word, f"<span class='highlight-red'>{word}</span>")
                                            examples.append(highlighted)
                                else:
                                    if word in next_line:
                                        clean_next = re.sub(r'<[^>]+>', '', next_line)
                                        highlighted = clean_next.replace(word, f"<span class='highlight-red'>{word}</span>")
                                        examples.append(highlighted)
                            
                            i += 1
                        
                        # 去重
                        unique_examples = []
                        seen = set()
                        for ex in examples:
                            plain_ex = re.sub(r'<[^>]+>', '', ex)
                            if plain_ex not in seen:
                                seen.add(plain_ex)
                                unique_examples.append(ex)
                        
                        if unique_examples:
                            collocations.append({
                                'pattern': pattern,
                                'all_examples': unique_examples
                            })
                            print(f"    添加模式: {pattern} ({len(unique_examples)}个例句)")
                            for idx, ex in enumerate(unique_examples[:3]):
                                print(f"      例句{idx+1}: {ex[:50]}...")
            
            # 统计总例句数
            total_examples = sum(len(c.get('all_examples', [])) for c in collocations)
            
            print(f"解析完成: {len(collocations)} 个搭配模式, {total_examples} 个例句")
            
            if collocations:
                try:
                    parent_dir = os.path.dirname(os.path.dirname(base_path))
                    rel_path = os.path.relpath(found_file, parent_dir)
                except:
                    rel_path = found_file
                
                return jsonify({
                    'success': True,
                    'result': {
                        'word': word,
                        'collocations': collocations,
                        'total_collocations': len(collocations),
                        'total_examples': total_examples,
                        'file_path': rel_path,
                        'note': f'找到 {len(collocations)} 个搭配模式'
                    }
                })
            else:
                return jsonify({
                    'success': True,
                    'result': {
                        'word': word,
                        'collocations': [],
                        'total_collocations': 0,
                        'total_examples': 0,
                        'note': f'未找到"{word}"的{("左" if collocation_type == "left" else "右")}搭配数据'
                    }
                })
        else:
            return jsonify({
                'success': True,
                'result': {
                    'word': word,
                    'collocations': [],
                    'total_collocations': 0,
                    'total_examples': 0,
                    'note': f'未找到"{word}"的{("左" if collocation_type == "left" else "右")}搭配数据'
                }
            })
        
    except Exception as e:
        print(f"查找搭配文件失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/word_info', methods=['POST', 'OPTIONS'])
def word_info():
    """获取词语的详细信息：拼音、词性、英语、例句、图片"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.json
        word = data.get('word')
        base_path = data.get('base_path')
        
        if not word or not base_path:
            return jsonify({'success': False, 'error': '缺少参数'})
        
        print(f"📖 查询词语信息: {word}")
        
        # ========== 1. 从词语例句拼音词性英语.txt 文件中读取信息 ==========
        info_file = os.path.join(base_path, '词语例句拼音词性英语.txt')
        word_info = {
            'word': word,
            'pinyin': '',
            'pos': '',
            'english': '',
            'examples': []
        }
        
        found_in_file = False
        
        if os.path.exists(info_file):
            print(f"  找到信息文件: {info_file}")
            
            # 尝试不同编码读取
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig']
            content = None
            
            for encoding in encodings:
                try:
                    with open(info_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"  使用 {encoding} 编码读取成功")
                    break
                except:
                    continue
            
            if content:
                lines = content.split('\n')
                
                # 第一行是标题行，跳过
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 按制表符分割
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        file_word = parts[0].strip()
                        
                        # 匹配词语
                        if file_word == word:
                            word_info['pinyin'] = parts[1].strip() if len(parts) > 1 else ''
                            word_info['pos'] = parts[2].strip() if len(parts) > 2 else ''
                            word_info['english'] = parts[3].strip() if len(parts) > 3 else ''
                            
                            # 例句可能有多句，用空格分隔
                            example_text = parts[4].strip() if len(parts) > 4 else ''
                            if example_text:
                                # 按句号分割多个例句
                                sentences = re.split(r'[。！？.!?]', example_text)
                                for sent in sentences:
                                    sent = sent.strip()
                                    if sent and len(sent) > 3:
                                        word_info['examples'].append(sent)
                            
                            found_in_file = True
                            print(f"  ✅ 找到词语信息: {word}")
                            break
                
                # 如果没找到，尝试模糊匹配（忽略空格和标点）
                if not found_in_file:
                    clean_word = re.sub(r'[^\u4e00-\u9fff]', '', word)
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split('\t')
                        if len(parts) >= 5:
                            file_word = parts[0].strip()
                            clean_file_word = re.sub(r'[^\u4e00-\u9fff]', '', file_word)
                            
                            if clean_file_word == clean_word:
                                word_info['pinyin'] = parts[1].strip() if len(parts) > 1 else ''
                                word_info['pos'] = parts[2].strip() if len(parts) > 2 else ''
                                word_info['english'] = parts[3].strip() if len(parts) > 3 else ''
                                
                                example_text = parts[4].strip() if len(parts) > 4 else ''
                                if example_text:
                                    sentences = re.split(r'[。！？.!?]', example_text)
                                    for sent in sentences:
                                        sent = sent.strip()
                                        if sent and len(sent) > 3:
                                            word_info['examples'].append(sent)
                                
                                found_in_file = True
                                print(f"  ✅ 模糊匹配找到词语信息: {file_word} -> {word}")
                                break
        
        if not found_in_file:
            print(f"  ⚠️ 未在信息文件中找到词语: {word}")
            word_info['pinyin'] = '未找到'
            word_info['pos'] = '未找到'
            word_info['english'] = '未找到'
        
        # ========== 2. 查找图片资源 ==========
        image_dir = os.path.join(base_path, '常用词语释义图片库')
        image_info = {
            'found': False,
            'filename': '',
            'data_url': ''
        }
        
        if os.path.exists(image_dir):
            print(f"  查找图片: {image_dir}")
            
            # 获取词语的纯汉字形式（忽略大小写、数字、特殊符号）
            clean_word_for_search = re.sub(r'[^a-zA-Z\u4e00-\u9fff]', '', word)
            
            # 遍历所有子文件夹
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        # 提取文件名中的汉字部分
                        file_name_without_ext = os.path.splitext(file)[0]
                        clean_file_name = re.sub(r'[^a-zA-Z\u4e00-\u9fff]', '', file_name_without_ext)
                        
                        # 比较是否匹配
                        if clean_file_name == clean_word_for_search or clean_word_for_search in clean_file_name:
                            image_path = os.path.join(root, file)
                            print(f"  ✅ 找到图片: {image_path}")
                            
                            try:
                                with open(image_path, 'rb') as f:
                                    img_data = f.read()
                                
                                import base64
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                
                                # 确定MIME类型
                                if file.lower().endswith('.png'):
                                    mime = 'image/png'
                                elif file.lower().endswith('.gif'):
                                    mime = 'image/gif'
                                elif file.lower().endswith('.bmp'):
                                    mime = 'image/bmp'
                                else:
                                    mime = 'image/jpeg'
                                
                                image_info['found'] = True
                                image_info['filename'] = file
                                image_info['data_url'] = f'data:{mime};base64,{img_base64}'
                                
                                break
                            except Exception as e:
                                print(f"  ❌ 读取图片失败: {e}")
        
        # 构建返回结果
        result = {
            'word': word,
            'pinyin': word_info['pinyin'],
            'pos': word_info['pos'],
            'english': word_info['english'],
            'examples': word_info['examples'],
            'image_info': image_info,
            'found_in_file': found_in_file
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        print(f"❌ 查询词语信息失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

# ========== 主程序入口 ==========
print("\n" + "=" * 60)
print("🚀 HSK文本分析API启动 - OpenRouter增强优化版")
print("=" * 60)

# 下载所有缺失的资产文件
download_all_assets()

print("📚 预加载HSK词汇表...")
hsk_vocabulary, hsk_chars = load_hsk_vocabulary()

print("\n🧠 预加载预测器...")
initialize_predictor()

print("\n🔧 预加载增强分析器...")
initialize_enhanced_analyzer()

print("\n📡 服务信息:")
print("   地址: http://localhost:5000")
print("   健康检查: http://localhost:5000/api/health")
print("   完整分析: POST http://localhost:5000/api/analyze")
print("   增强分析: POST http://localhost:5000/api/enhanced_analyze")
print("   详细教学建议: POST http://localhost:5000/api/enhance_teaching")
print("   颜色高亮: POST http://localhost:5000/api/color_text")

print("\n🎯 功能特色:")
print("   - 基于真实HSK词汇表（已修复多音字处理）")
print("   - 汉字级别分布分析")
print("   - 词汇难度可视化")
print("   - 生词自动识别")
print("   - 增强分析（详细教学建议生成）")
print("   - 词语搭配库支持")
print("   - 使用OpenRouter Qwen3 Coder 480B模型")
print("   - 无字符限制教学建议生成")

print("\n🔍 增强分析特色:")
print("   - 详细教学建议（包含完整的九大教学环节）")
print("   - 精准提取文本主题")
print("   - 提取教学关键词")
print("   - 基于N+1原则提取生字生词")
print("   - 查找词语搭配和例句")
print("   - 提供详细教学要点分析")
print("   - 包含具体练习题和答案")
print("   - 推荐真实教学资源链接")

print("\n⚙️ 系统要求:")
print("   - OpenRouter API密钥（已配置）")
print("   - 使用模型: qwen/qwen3-coder-480b-a35b:free")
print("   - 搭配词库目录: hsk_predictor - 1/n/")
print("   - 无需本地Ollama服务")

print("\n💡 按 Ctrl+C 停止")
print("=" * 60)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
