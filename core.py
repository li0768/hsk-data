import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

# 模型配置
OPTIMIZED_CONFIG = {
    'vocab_size': 60000,
    'embed_dim': 300,
    'hidden_dim': 384,
    'num_layers': 3,
    'dropout': 0.4,
    'num_classes': 7,
    'conv_channels': 128,
    'attention_heads': 8,
}

class OptimizedHSKClassifier(nn.Module):
    """优化的HSK分类器模型"""
    
    def __init__(self, config=OPTIMIZED_CONFIG):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'], padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, config['embed_dim']))
        
        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config['embed_dim'], config['conv_channels'], 2, padding=1),
                nn.BatchNorm1d(config['conv_channels']),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(config['embed_dim'], config['conv_channels'], 3, padding=1),
                nn.BatchNorm1d(config['conv_channels']),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(config['embed_dim'], config['conv_channels'], 4, padding=2),
                nn.BatchNorm1d(config['conv_channels']),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        ])
        
        # BiLSTM
        self.lstm = nn.LSTM(
            config['embed_dim'],
            config['hidden_dim'] // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=config['num_layers'],
            dropout=0.2
        )
        self.lstm_norm = nn.LayerNorm(config['hidden_dim'])
        
        # 多头注意力
        self.self_attention = nn.MultiheadAttention(
            config['hidden_dim'],
            num_heads=config['attention_heads'],
            dropout=0.1,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(config['hidden_dim'])
        
        # 特征融合维度计算
        conv_features_dim = config['conv_channels'] * len(self.conv_layers) * 2
        lstm_features_dim = config['hidden_dim'] * 3
        total_features = conv_features_dim + lstm_features_dim
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 主分类器
        self.main_classifier = nn.Sequential(
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, config['num_classes'])
        )
        
        # 辅助分类器
        self.aux_classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, config['num_classes'])
        )
        
    def forward(self, input_ids, attention_mask=None, use_aux=False):
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入 + 位置编码
        x = self.embedding(input_ids)
        if seq_len <= 200:
            x = x + self.pos_encoding[:, :seq_len, :]
        x = self.embed_dropout(x)
        
        # 多尺度卷积特征
        x_transposed = x.transpose(1, 2)
        conv_features = []
        
        for conv in self.conv_layers:
            conv_out = conv(x_transposed)
            max_pool = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            avg_pool = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
            conv_features.extend([max_pool, avg_pool])
        
        conv_combined = torch.cat(conv_features, dim=1)
        
        # LSTM序列编码
        lstm_out, (hidden, _) = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # 多头注意力
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
            
        attn_out, attn_weights = self.self_attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )
        attn_out = self.attention_norm(lstm_out + attn_out)
        
        # 多尺度LSTM特征池化
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        
        # LSTM最终状态
        if hidden.dim() == 3:
            forward_final = hidden[-2, :, :]
            backward_final = hidden[-1, :, :]
            lstm_final = torch.cat([forward_final, backward_final], dim=1)
        else:
            lstm_final = avg_pool
        
        # 特征融合
        lstm_features = torch.cat([max_pool, avg_pool, lstm_final], dim=1)
        all_features = torch.cat([conv_combined, lstm_features], dim=1)
        
        fused_features = self.feature_fusion(all_features)
        
        # 主分类
        main_logits = self.main_classifier(fused_features)
        
        if use_aux:
            aux_logits = self.aux_classifier(fused_features)
            return main_logits, aux_logits
        
        return main_logits

class CustomTokenizer:
    """自定义分词器 - 修复版"""
    
    def __init__(self, vocab_size=60000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_index = self.word2idx  # ✅ 添加 word_index 属性
        self.index_word = self.idx2word  # ✅ 添加 index_word 属性
        self.num_words = vocab_size  # ✅ 添加 num_words 属性
        
    def encode(self, text, max_length=200):
        """编码文本为模型输入"""
        words = list(text.strip())
        input_ids = [self.word2idx.get(word, 1) for word in words]
        
        # 填充或截断
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        else:
            input_ids = input_ids + [0] * (max_length - len(input_ids))
        
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, input_ids):
        """解码模型输出为文本"""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        words = []
        for idx in input_ids:
            if idx == 0:  # PAD token
                continue
            if idx == 1:  # UNK token
                words.append('<UNK>')
            elif idx in self.idx2word:
                words.append(self.idx2word[idx])
            else:
                words.append('<UNK>')
        
        return ''.join(words)
    
    # ✅ 添加 Keras 风格的兼容方法
    def texts_to_sequences(self, texts):
        """将文本转换为序列（Keras风格）"""
        if isinstance(texts, str):
            texts = [texts]
        
        sequences = []
        for text in texts:
            words = list(text.strip())
            seq = [self.word2idx.get(word, 1) for word in words[:200]]  # 限制长度
            sequences.append(seq)
        
        return sequences
    
    def tokenize(self, text):
        """分词方法"""
        return list(text.strip())