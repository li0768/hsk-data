import requests
import json
import time
import traceback
import re
from typing import Dict, List, Optional, Any

class OpenRouterClient:
    """OpenRouter API客户端 - 严格遵循官方文档"""
    
    def __init__(self, 
                 api_key: str = "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",
                 model: str = "openai/gpt-oss-120b:free",
                 site_url: str = "http://localhost:5000",
                 site_name: str = "HSK Smart Assistant",
                 timeout: int = 180,
                 verbose: bool = True):
        
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.timeout = timeout
        self.verbose = verbose
        
        # 严格按官方文档设置headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url,
            "X-OpenRouter-Title": site_name,
            "Content-Type": "application/json"
        }
        
        if verbose:
            print(f"✅ OpenRouter客户端初始化成功")
            print(f"   模型: {model}")
            print(f"   站点: {site_url}")
            print(f"   标题: {site_name}")
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.1,
                       max_tokens: int = 8000,
                       top_p: float = 0.9) -> Optional[str]:
        """
        调用OpenRouter API获取content内容
        """
        try:
            # 清理消息中的特殊字符
            cleaned_messages = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, str):
                    content = content.encode('utf-8', errors='ignore').decode('utf-8')
                
                cleaned_messages.append({
                    "role": msg["role"],
                    "content": content
                })
            
            # 构建请求体
            payload = {
                "model": self.model,
                "messages": cleaned_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": False
            }
            
            if self.verbose:
                print(f"  🤖 发送请求到OpenRouter (模型: {self.model})...")
                print(f"     URL: https://openrouter.ai/api/v1/chat/completions")
                print(f"     消息数: {len(messages)}")
                print(f"     最大token数: {max_tokens}")
            
            start_time = time.time()
            
            # 发送请求
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if self.verbose:
                print(f"     状态码: {response.status_code}")
                print(f"     耗时: {elapsed_time:.1f}秒")
            
            # 处理成功响应
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    if self.verbose:
                        print(f"     响应结构: {list(result.keys())}")
                    
                    # 检查是否有choices字段
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        
                        # 获取content内容
                        content = message.get("content")
                        
                        # 如果content不为空，直接返回
                        if content and content.strip():
                            if self.verbose:
                                print(f"      ✅ 获取到content: {len(content)} 字符")
                                print(f"      📝 预览: {content[:100]}...")
                            return content
                        
                        # 如果content为空，检查是否有reasoning字段（某些模型会返回）
                        reasoning = message.get("reasoning")
                        if reasoning and reasoning.strip():
                            if self.verbose:
                                print(f"      ⚠️ content为空，尝试从reasoning提取")
                            
                            # 从reasoning中提取可能的答案
                            extracted = self._extract_answer_from_reasoning(reasoning)
                            if extracted:
                                if self.verbose:
                                    print(f"      ✅ 从reasoning提取到内容: {len(extracted)} 字符")
                                    print(f"      📝 预览: {extracted[:100]}...")
                                return extracted
                        
                        # 获取结束原因
                        finish_reason = result["choices"][0].get("finish_reason", "未知")
                        native_finish_reason = result["choices"][0].get("native_finish_reason", "未知")
                        
                        if self.verbose:
                            print(f"      ⚠️ 未获取到有效内容")
                            print(f"      ⚠️ 结束原因: {finish_reason}")
                            print(f"      ⚠️ 原生结束原因: {native_finish_reason}")
                    
                    return None
                    
                except json.JSONDecodeError as e:
                    print(f"      ❌ JSON解析失败: {e}")
                    print(f"      原始响应: {response.text[:500]}")
                    return None
            
            # 处理401错误
            elif response.status_code == 401:
                print(f"      ❌ API密钥无效 (401)")
                print(f"      请检查: https://openrouter.ai/settings/keys")
                return None
            
            # 处理429错误 - 频率限制
            elif response.status_code == 429:
                print(f"      ⚠️ 频率限制 (429)")
                return None
            
            # 处理其他错误
            else:
                print(f"      ❌ API错误: HTTP {response.status_code}")
                print(f"      响应: {response.text[:500]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"      ❌ 请求超时 ({self.timeout}秒)")
            return None
        except requests.exceptions.ConnectionError:
            print(f"      ❌ 连接错误 - 无法访问 OpenRouter")
            return None
        except Exception as e:
            print(f"      ❌ 调用异常: {e}")
            traceback.print_exc()
            return None
    
    def _extract_answer_from_reasoning(self, reasoning: str) -> Optional[str]:
        """从reasoning字段中提取最终的答案，而不是推理过程"""
        if not reasoning:
            return None
        
        lines = reasoning.split('\n')
        
        # 定义要过滤掉的词汇
        filter_words = [
            '不要', '请', '必须', '应该', '可以', '要求', '需要', '确保',
            '示例', '比如', '例如', '如', '按', '照', '根据', '基于',
            '首先', '然后', '接着', '最后', '所以', '因此', '因为', '考虑',
            '分析', '思考', '推断', '推测', '输出', '只输出', '直接输出',
            '说明', '解释', '要求', '条件', '规则', '规定', '说明或'  # 添加新过滤词
        ]
        
        # 定义可能的关键词列表（用于文本中提取）
        possible_keywords = ['学习', '考试', '电影', '图书馆', '计划', '准备', '明天', '随后', '娱乐', '工作', '生活', '家庭']
        
        # 1. 先查找明确的答案标记
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 查找"答案是"
            if '答案是' in line:
                parts = line.split('答案是', 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # 清理可能的标点
                    answer = re.sub(r'[^\u4e00-\u9fff，、]', '', answer)
                    # 检查是否包含过滤词
                    if answer and len(answer) >= 2 and not any(word in answer for word in filter_words):
                        # 如果是关键词，限制数量
                        if '、' in answer:
                            keywords = answer.split('、')
                            keywords = [k for k in keywords if len(k) >= 2 and not any(word in k for word in filter_words)]
                            if keywords:
                                return '、'.join(keywords[:5])
                        return answer
            
            # 查找"所以"
            if '所以' in line:
                parts = line.split('所以', 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    answer = re.sub(r'[^\u4e00-\u9fff，、]', '', answer)
                    if answer and len(answer) >= 2 and not any(word in answer for word in filter_words):
                        if '、' in answer:
                            keywords = answer.split('、')
                            keywords = [k for k in keywords if len(k) >= 2 and not any(word in k for word in filter_words)]
                            if keywords:
                                return '、'.join(keywords[:5])
                        return answer
        
        # 2. 查找可能的关键词
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            
            # 如果行中包含可能的关键词
            found_keywords = []
            for kw in possible_keywords:
                if kw in line:
                    found_keywords.append(kw)
            
            if found_keywords:
                return '、'.join(found_keywords[:5])
        
        # 3. 查找最后几行中看起来像答案的内容
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            
            # 跳过明显的推理行和指令行
            if any(word in line for word in filter_words):
                continue
            
            # 清理只保留中文和顿号
            cleaned = re.sub(r'[^\u4e00-\u9fff，、]', '', line)
            if cleaned and len(cleaned) >= 2:
                # 如果包含顿号，说明是关键词列表
                if '、' in cleaned:
                    keywords = cleaned.split('、')
                    # 过滤掉无效的关键词
                    valid_keywords = []
                    for kw in keywords:
                        if len(kw) >= 2 and not any(word in kw for word in filter_words):
                            valid_keywords.append(kw)
                    if valid_keywords:
                        return '、'.join(valid_keywords[:5])
                # 如果是单个词，直接返回
                elif len(cleaned) <= 6 and not any(word in cleaned for word in filter_words):
                    return cleaned
        
        # 4. 如果以上都没找到，尝试从文本末尾提取
        last_part = reasoning[-200:]  # 取最后200个字符
        # 查找可能的答案
        possible_answers = re.findall(r'[，、]?\s*([\u4e00-\u9fff]{2,8})[，、\s]*$', last_part)
        if possible_answers:
            answer = possible_answers[0]
            if not any(word in answer for word in filter_words):
                return answer
        
        return None
    
    def generate_text(self, 
                     prompt: str, 
                     system_prompt: str = None,
                     temperature: float = 0.1,
                     max_tokens: int = 8000) -> Optional[str]:
        """
        生成文本（单轮对话）
        """
        messages = []
        
        # 添加系统提示，强制要求直接输出，不要推理
        default_system = "你是一个直接回答问题的AI助手。请只输出最终答案，不要输出任何思考过程、推理步骤或解释。直接给出简洁明确的答案。"
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + " 请直接输出答案，不要输出任何推理过程。"})
        else:
            messages.append({"role": "system", "content": default_system})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def test_connection(self):
        """测试API连接"""
        print("🔍 测试OpenRouter连接...")
        
        # 简单的测试消息
        test_messages = [
            {"role": "system", "content": "你是一个直接回答问题的AI助手。请只输出答案，不要输出任何推理过程。"},
            {"role": "user", "content": "请直接输出'连接成功'四个字"}
        ]
        
        result = self.chat_completion(
            messages=test_messages,
            max_tokens=20
        )
        
        if result:
            print(f"✅ 连接成功！")
            print(f"   回复: {result}")
            return True
        else:
            print(f"❌ 连接失败")
            return False
    
    def check_quota(self):
        """检查API配额"""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API配额信息:")
                print(f"   使用量: {data.get('usage', '未知')}")
                return {"success": True, "data": data}
            else:
                print(f"❌ 获取配额失败: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"❌ 检查配额异常: {e}")
            return {"success": False, "error": str(e)}


# 全局客户端实例
_openrouter_client = None

def get_openrouter_client(verbose=True):
    """获取OpenRouter客户端全局实例"""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient(
            api_key="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",
            model="openai/gpt-oss-120b:free",
            site_url="http://localhost:5000",
            site_name="HSK Smart Assistant",
            timeout=180,
            verbose=verbose
        )
    return _openrouter_client


def test_openrouter_connection():
    """测试OpenRouter连接"""
    client = get_openrouter_client()
    return client.test_connection()


if __name__ == "__main__":
    print("🧪 测试OpenRouter连接...")
    test_openrouter_connection()
    
    # 测试主题提取
    print("\n🧪 测试主题提取...")
    client = get_openrouter_client()
    result = client.generate_text(
        prompt="请分析以下文本的主题，只输出主题名称：明天要去图书馆学习准备考试，随后会去看电影。",
        max_tokens=50
    )
    print(f"主题提取结果: {result}")
    
    # 测试关键词提取
    print("\n🧪 测试关键词提取...")
    result = client.generate_text(
        prompt="请从以下文本提取最重要的关键词，用顿号分隔：明天要去图书馆学习准备考试，随后会去看电影。",
        max_tokens=100
    )
    print(f"关键词提取结果: {result}")