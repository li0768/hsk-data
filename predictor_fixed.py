"""
修复导入问题的predictor版本
"""
import os
import torch
import torch.nn.functional as F
import pickle
import sys
import json
from tabulate import tabulate

# 直接导入其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入core模块
import core
from core import OptimizedHSKClassifier, CustomTokenizer

# 导入vocab_db模块
import vocab_db
from vocab_db import get_vocab_db

# 复制原来的HSKTextPredictor类到这里
class HSKTextPredictor:
    def __init__(self, model_path=None, enable_vocab_analysis=True, verbose=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.level_names = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', 'HSK7-9']
        self.level_keys = ['1', '2', '3', '4', '5', '6', '7-9']
        self.enable_vocab_analysis = enable_vocab_analysis
        self.verbose = verbose
        
        if self.enable_vocab_analysis:
            try:
                self.vocab_db = get_vocab_db(verbose=verbose)
                if verbose:
                    print("✅ 词汇分析功能已启用")
            except Exception as e:
                if verbose:
                    print(f"⚠️ 词汇分析功能初始化失败: {e}")
                self.enable_vocab_analysis = False
                self.vocab_db = None
        else:
            self.vocab_db = None
        
        if model_path is None:
            model_path = self._find_model_path()
        
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    def _find_model_path(self):
        """自动查找模型文件路径"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '../models/best_optimized_model.pth'),
            os.path.join(os.path.dirname(__file__), '../../models/best_optimized_model.pth'),
            'best_optimized_model.pth',
            './models/best_optimized_model.pth',
            'C:/Users/DF-Lenovo/Desktop/hsk_predictor/models/best_optimized_model.pth',
            os.path.join(os.path.expanduser('~'), 'Desktop/hsk_predictor/models/best_optimized_model.pth')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if self.verbose:
                    print(f"🔍 找到模型文件: {path}")
                return path
        
        raise FileNotFoundError("无法自动找到模型文件，请手动指定model_path参数")
    
    def load_model(self, model_path):
        """加载模型和分词器"""
        try:
            if self.verbose:
                print(f"🔄 加载模型: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model_config = checkpoint['config']
            
            self.model = OptimizedHSKClassifier(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            tokenizer_path = model_path.replace('.pth', '_tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                if self.verbose:
                    print(f"🔍 找到分词器文件: {tokenizer_path}")
                
                sys.modules['__main__'].CustomTokenizer = CustomTokenizer
                
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                
                if self.verbose:
                    print(f"✅ 模型和分词器加载成功!")
            else:
                if self.verbose:
                    print(f"⚠️ 未找到分词器文件: {tokenizer_path}")
                self.tokenizer = CustomTokenizer()
                if self.verbose:
                    print("✅ 使用新分词器")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self._fallback_load(model_path)
    
    def _fallback_load(self, model_path):
        """备选加载方案"""
        try:
            if self.verbose:
                print("🔄 尝试备选加载方案...")
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model_config = checkpoint['config']
            
            self.model = OptimizedHSKClassifier(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = CustomTokenizer()
            
            if self.verbose:
                print("✅ 备选方案加载成功（使用新分词器）")
            
        except Exception as e:
            raise RuntimeError(f"所有加载方案都失败: {e}")
    
    # 在 predictor_fixed.py 的 HSKTextPredictor 类中

    def predict(self, text):
        """预测单条文本的HSK级别"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型或分词器未正确加载")
    
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
    
        try:
            # ✅ 添加文本预处理（与完整分析保持一致）
            text = text.strip()
        
            # 可选：清理文本中的特殊字符
            # text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？：；、,.!?]', '', text)
        
            encoding = self.tokenizer.encode(text)
            input_ids = encoding['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = encoding['attention_mask'].unsqueeze(0).to(self.device)
        
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1)
                predicted_level_idx = torch.argmax(outputs, 1).item()
                confidence = torch.max(probabilities).item()
        
            result = {
                'level': self.level_names[predicted_level_idx],
                'level_key': self.level_keys[predicted_level_idx],
                'confidence': confidence,
                'probabilities': {},
                'text': text
            }
        
            for i, level in enumerate(self.level_names):
                result['probabilities'][level] = probabilities[0][i].item()
        
            return result
        
        except Exception as e:
            raise RuntimeError(f"预测过程中出错: {e}")
    
    def get_stats(self):
        """获取系统统计信息"""
        stats = {
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'vocab_analysis_enabled': self.enable_vocab_analysis,
            'device': str(self.device),
            'levels_supported': self.level_names
        }
        
        if self.vocab_db:
            try:
                db_stats = self.vocab_db.get_stats()
                stats.update(db_stats)
            except:
                pass
        
        return stats

# 简化的其他方法
def simplified_analyze_text_comprehensive(self, text, simple_mode=False):
    """简化版的分析方法"""
    prediction = self.predict(text)
    
    return {
        'prediction': prediction,
        'analysis': None,
        'colored_text': None,
        'display_text': ["简化版分析，完整分析需要vocab_db功能"]
    }

# 添加到类中
HSKTextPredictor.analyze_text_comprehensive = simplified_analyze_text_comprehensive