/**
 * HSK分析系统客户端 - 增强版（支持Qwen3:8b模型）
 */

const HSKClient = {
    // API地址配置
    apiUrls: {
        python: 'http://localhost:5000',
        dotnet: 'http://localhost:8000'  // 备用地址，实际可能不存在
    },
    
    // 状态
    status: {
        python: false,
        dotnet: false,
        ollama: false,
        models: []  // 可用模型列表
    },
    
    // 初始化
    init: function() {
        console.log('🚀 HSK客户端初始化...');
        console.log('📋 支持模型: qwen3:8b, deepseek-r1:8b, llama3.2:3b');
        this.checkAllStatus();
    },
    
    // 检查所有服务状态
    checkAllStatus: function() {
        this.checkPythonStatus();
        this.checkOllamaStatus();
        this.checkModelAvailability();
    },
    
    // 检查Python API状态
    checkPythonStatus: function() {
        fetch(this.apiUrls.python + '/api/health')
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('Python API不可用');
            })
            .then(data => {
                this.status.python = true;
                console.log('✅ Python API可用:', data);
                this.updateStatusUI();
            })
            .catch(error => {
                this.status.python = false;
                console.warn('❌ Python API不可用:', error.message);
                this.updateStatusUI();
            });
    },
    
    // 检查Ollama状态
    checkOllamaStatus: function() {
        fetch('http://localhost:11434/api/tags', { method: 'GET' })
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('Ollama不可用');
            })
            .then(data => {
                this.status.ollama = true;
                console.log('✅ Ollama可用:', data);
                this.updateStatusUI();
            })
            .catch(error => {
                this.status.ollama = false;
                console.warn('❌ Ollama不可用:', error.message);
                this.updateStatusUI();
            });
    },
    
    // 检查模型可用性
    checkModelAvailability: function() {
        if (!this.status.ollama) {
            console.warn('Ollama不可用，无法检查模型');
            return Promise.resolve(false);
        }
        
        return fetch('http://localhost:11434/api/tags')
            .then(response => response.json())
            .then(data => {
                const models = data.models || [];
                this.status.models = models.map(m => m.name);
                console.log('📦 可用模型:', this.status.models);
                
                // 检查目标模型是否可用
                const targetModels = ['qwen3:8b', 'deepseek-r1:8b', 'llama3.2:3b'];
                const availableTargets = targetModels.filter(model => 
                    this.status.models.includes(model)
                );
                
                console.log('🎯 目标模型可用性:', availableTargets);
                
                if (availableTargets.length > 0) {
                    console.log(`✅ 找到模型: ${availableTargets.join(', ')}`);
                } else {
                    console.warn('⚠️ 未找到目标模型，请安装: ollama pull qwen3:8b');
                }
                
                return availableTargets.length > 0;
            })
            .catch(error => {
                console.error('检查模型失败:', error);
                return false;
            });
    },
    
    // 获取最佳可用模型
    getBestAvailableModel: function() {
        const preferredModels = ['qwen3:8b', 'deepseek-r1:8b', 'llama3.2:3b'];
        
        for (const model of preferredModels) {
            if (this.status.models.includes(model)) {
                console.log(`🎯 选择模型: ${model}`);
                return model;
            }
        }
        
        console.warn('⚠️ 未找到首选模型，使用默认');
        return 'qwen3:8b'; // 默认返回首选模型
    },
    
    // 更新状态显示
    updateStatusUI: function() {
        const pythonStatus = document.getElementById('pythonStatus');
        const dotnetStatus = document.getElementById('dotnetStatus');
        const ollamaStatus = document.getElementById('ollamaStatus');
        const ollamaDot = document.getElementById('ollamaDot');
        const ollamaText = document.getElementById('ollamaText');
        
        if (pythonStatus) {
            pythonStatus.textContent = this.status.python ? '正常' : '离线';
            pythonStatus.className = this.status.python ? 
                'status-value status-good' : 'status-value status-bad';
        }
        
        if (dotnetStatus) {
            dotnetStatus.textContent = '未启用';
            dotnetStatus.className = 'status-value status-warning';
        }
        
        if (ollamaStatus && ollamaDot && ollamaText) {
            if (this.status.ollama) {
                ollamaDot.classList.add('online');
                ollamaText.textContent = '在线';
                ollamaText.style.color = '#10b981';
            } else {
                ollamaDot.classList.remove('online');
                ollamaText.textContent = '离线';
                ollamaText.style.color = '#dc2626';
            }
        }
    },
    
    // ============================================
    // 文本预测（快速分析）
    // ============================================
    predict: async function(text) {
        if (!this.status.python) {
            console.error('Python API不可用');
            return {
                success: false,
                error: 'Python API服务未启动，请检查服务状态'
            };
        }
        
        try {
            const response = await fetch(this.apiUrls.python + '/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('预测结果:', data);
            return data;
            
        } catch (error) {
            console.error('预测失败:', error);
            
            // 如果API失败，尝试备用预测
            return this.fallbackPredict(text);
        }
    },
    
    // ============================================
    // 完整分析
    // ============================================
    analyze: async function(text, simple = false) {
        if (!this.status.python) {
            console.error('Python API不可用');
            return {
                success: false,
                error: 'Python API服务未启动，请检查服务状态'
            };
        }
        
        try {
            const response = await fetch(this.apiUrls.python + '/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text,
                    simple: simple 
                })
            });
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('分析结果:', data);
            return data;
            
        } catch (error) {
            console.error('分析失败:', error);
            
            // 如果API失败，尝试备用分析
            return this.fallbackAnalyze(text);
        }
    },
    
    // ============================================
    // 增强分析（使用qwen3:8b模型）
    // ============================================
    enhancedAnalyze: async function(text) {
        if (!this.status.python) {
            console.error('Python API不可用');
            return {
                success: false,
                error: 'Python API服务未启动，请检查服务状态'
            };
        }
        
        try {
            console.log('🚀 发送增强分析请求...');
            console.log(`🎯 使用模型: ${this.getBestAvailableModel()}`);
            
            const response = await fetch(this.apiUrls.python + '/api/enhanced_analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text,
                    preferred_model: this.getBestAvailableModel() // 传递首选模型
                })
            });
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('增强分析结果:', data);
            
            // 检查模型使用情况
            if (data.result?.performance?.model_used) {
                console.log(`✅ 实际使用模型: ${data.result.performance.model_used}`);
            }
            
            return data;
            
        } catch (error) {
            console.error('增强分析失败:', error);
            
            // 如果API失败，返回错误
            return {
                success: false,
                error: '增强分析失败: ' + error.message
            };
        }
    },
    
    // ============================================
    // 测试增强分析API
    // ============================================
    testEnhancedAPI: async function() {
        if (!this.status.python) {
            console.error('Python API不可用');
            return {
                success: false,
                error: 'Python API服务未启动，请检查服务状态'
            };
        }
        
        try {
            console.log('🔧 测试增强分析API...');
            const response = await fetch(this.apiUrls.python + '/api/enhanced_analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: '测试文本'
                })
            });
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('测试结果:', data);
            return data;
        } catch (error) {
            console.error('测试增强分析失败:', error);
            return { success: false, error: error.message };
        }
    },
    
    // ============================================
    // 词语搭配检索
    // ============================================
    getCollocation: async function(word) {
        if (!this.status.python) {
            console.error('Python API不可用');
            return {
                success: false,
                error: 'Python API服务未启动，请检查服务状态'
            };
        }
        
        if (!word || word.trim() === '') {
            return {
                success: false,
                error: '请输入要检索的词语'
            };
        }
        
        try {
            console.log(`🔍 检索词语搭配: ${word}`);
            const response = await fetch(this.apiUrls.python + '/api/get_collocation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    word: word.trim()
                })
            });
            
            if (!response.ok) {
                console.error('API响应错误:', response.status);
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('搭配检索结果:', data);
            return data;
            
        } catch (error) {
            console.error('搭配检索失败:', error);
            
            // 返回备用搭配数据
            return this.fallbackCollocation(word);
        }
    },
    
    // ============================================
    // 颜色高亮文本
    // ============================================
    colorText: async function(text) {
        if (!this.status.python) {
            console.error('Python API不可用');
            return {
                success: false,
                error: 'Python API服务未启动，请检查服务状态'
            };
        }
        
        try {
            const response = await fetch(this.apiUrls.python + '/api/color_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text
                })
            });
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('颜色高亮结果:', data);
            return data;
            
        } catch (error) {
            console.error('颜色高亮失败:', error);
            
            return {
                success: false,
                error: '颜色高亮失败: ' + error.message
            };
        }
    },
    
    // ============================================
    // 获取当前模型信息
    // ============================================
    getModelInfo: function() {
        const model = this.getBestAvailableModel();
        const modelInfo = {
            'qwen3:8b': {
                name: 'Qwen3:8b',
                description: '阿里通义千问模型，中文理解能力强',
                color: '#10b981'
            },
            'deepseek-r1:8b': {
                name: 'DeepSeek-R1:8b',
                description: '深度求索推理模型，逻辑分析精准',
                color: '#3b82f6'
            },
            'llama3.2:3b': {
                name: 'Llama3.2:3b',
                description: 'Meta Llama模型，快速响应',
                color: '#8b5cf6'
            }
        };
        
        return modelInfo[model] || {
            name: model,
            description: 'AI模型',
            color: '#6b7280'
        };
    },
    
    // ============================================
    // 备用功能（当API不可用时）
    // ============================================
    
    // 备用预测（当Python API不可用时）
    fallbackPredict: function(text) {
        console.log('使用备用预测逻辑');
        
        // 简单的规则预测
        let level, confidence;
        const length = text.length;
        
        if (length < 10) {
            level = "HSK1";
            confidence = 0.8;
        } else if (length < 20) {
            level = "HSK2";
            confidence = 0.75;
        } else if (length < 40) {
            level = "HSK3";
            confidence = 0.7;
        } else if (length < 60) {
            level = "HSK4";
            confidence = 0.65;
        } else if (length < 80) {
            level = "HSK5";
            confidence = 0.6;
        } else if (length < 120) {
            level = "HSK6";
            confidence = 0.55;
        } else {
            level = "HSK7-9";
            confidence = 0.5;
        }
        
        const result = {
            success: true,
            result: {
                level: level,
                level_key: level.replace("HSK", "").replace("-9", ""),
                confidence: confidence,
                probabilities: {
                    "HSK1": 0.1, "HSK2": 0.15, "HSK3": 0.2,
                    "HSK4": 0.15, "HSK5": 0.1, "HSK6": 0.1, "HSK7-9": 0.2
                },
                text: text,
                note: "使用备用规则预测（Python API不可用）"
            }
        };
        
        return result;
    },
    
    // 备用分析（当Python API不可用时）
    fallbackAnalyze: function(text) {
        console.log('使用备用分析逻辑');
        
        const level = this.fallbackPredict(text).result.level;
        const confidence = this.fallbackPredict(text).result.confidence;
        
        // 统计中文字符
        const chineseChars = text.match(/[\u4e00-\u9fff]/g) || [];
        const chineseCharCount = chineseChars.length;
        
        // 统计标点
        const punctuationCount = (text.match(/[。，；：！？、""''()（）【】《》]/g) || []).length;
        
        // 统计句子
        const sentenceCount = (text.split(/[。！？.!?]/).filter(s => s.trim().length > 0)).length;
        
        // 构建分析结果
        const result = {
            success: true,
            result: {
                prediction: {
                    level: level,
                    level_key: level.replace("HSK", "").replace("-9", ""),
                    confidence: confidence,
                    probabilities: {
                        "HSK1": 0.1, "HSK2": 0.15, "HSK3": 0.2,
                        "HSK4": 0.15, "HSK5": 0.1, "HSK6": 0.1, "HSK7-9": 0.2
                    },
                    text: text,
                    note: "使用备用规则分析（Python API不可用）"
                },
                analysis: {
                    features: {
                        text_length: text.length,
                        chinese_char_count: chineseCharCount,
                        punctuation_count: punctuationCount,
                        sentence_count: sentenceCount,
                        estimated_hsk_level: level
                    }
                },
                display_text: [
                    "=".repeat(60),
                    "📊 中文文本分析报告（备用模式）",
                    "=".repeat(60),
                    "",
                    "⚠️ 注意：Python API服务当前不可用，使用备用分析模式",
                    "",
                    "一、基础信息",
                    `  文本长度: ${text.length} 字符`,
                    `  中文字符: ${chineseCharCount}`,
                    `  标点符号: ${punctuationCount}`,
                    `  句子数量: ${sentenceCount}`,
                    "",
                    "二、预测结果",
                    `  估算等级: ${level}`,
                    `  置信度: ${confidence.toFixed(2)}`,
                    "",
                    "三、建议",
                    "  1. 检查Python API服务是否启动",
                    "  2. 确保 http://localhost:5000 可以访问",
                    "  3. 重启 start_all.bat",
                    "",
                    "=".repeat(60),
                    "✅ 分析完成（备用模式）",
                    "=".repeat(60)
                ]
            }
        };
        
        return result;
    },
    
    // 备用搭配检索（当API不可用时）
    fallbackCollocation: function(word) {
        console.log(`使用备用搭配检索: ${word}`);
        
        // 备用搭配数据
        const backupData = {
            "学习": [
                { left: "认真", right: "", example: "他每天都很认真学习。" },
                { left: "", right: "汉语", example: "我在北京学习汉语。" },
                { left: "", right: "方法", example: "掌握正确的学习方法很重要。" },
                { left: "努力", right: "", example: "为了考试，他努力学习。" },
                { left: "继续", right: "", example: "毕业后，我想继续学习。" }
            ],
            "工作": [
                { left: "努力", right: "", example: "他工作很努力。" },
                { left: "", right: "经验", example: "他有丰富的工作经验。" },
                { left: "", right: "效率", example: "提高工作效率很重要。" },
                { left: "认真", right: "", example: "她工作非常认真。" },
                { left: "找到", right: "", example: "我终于找到工作了。" }
            ],
            "生活": [
                { left: "日常", right: "", example: "这是他的日常生活。" },
                { left: "", right: "习惯", example: "养成良好的生活习惯。" },
                { left: "", right: "水平", example: "生活水平不断提高。" },
                { left: "幸福", right: "", example: "他们过着幸福的生活。" },
                { left: "美好", right: "", example: "向往美好的生活。" }
            ],
            "中国": [
                { left: "去", right: "", example: "我想去中国旅游。" },
                { left: "在", right: "", example: "我在中国住了三年。" },
                { left: "", right: "文化", example: "中国文化很有意思。" },
                { left: "", right: "菜", example: "我喜欢吃中国菜。" },
                { left: "回", right: "", example: "他准备回中国了。" }
            ]
        };
        
        // 如果词语在备用数据中
        if (backupData[word]) {
            return {
                success: true,
                result: {
                    word: word,
                    collocations: backupData[word],
                    examples: backupData[word].map(item => item.example),
                    total_collocations: backupData[word].length,
                    total_examples: backupData[word].length,
                    note: "使用备用搭配数据（API不可用）"
                }
            };
        }
        
        // 通用搭配
        const genericCollocations = [
            { left: "使用", right: "", example: `这个词语"${word}"在日常交流中经常使用。` },
            { left: "学习", right: "", example: `学习"${word}"这个词语对提高汉语水平很有帮助。` },
            { left: "", right: "的意思", example: `你知道"${word}"的意思吗？` },
            { left: "了解", right: "", example: `我想了解更多关于"${word}"的信息。` },
            { left: "掌握", right: "", example: `掌握"${word}"的用法很重要。` }
        ];
        
        return {
            success: true,
            result: {
                word: word,
                collocations: genericCollocations,
                examples: genericCollocations.map(item => item.example),
                total_collocations: genericCollocations.length,
                total_examples: genericCollocations.length,
                note: "使用通用搭配数据（词语未在词库中）"
            }
        };
    },
    
    // ============================================
    // HTML生成功能
    // ============================================
    
    // 生成预测结果的HTML
    generateResultHTML: function(resultData) {
        if (!resultData.success) {
            return `
                <div style="text-align: center; padding: 40px; color: #f56565;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 15px;"></i>
                    <h3>分析失败</h3>
                    <p>${resultData.error || '未知错误'}</p>
                </div>
            `;
        }
        
        const result = resultData.result;
        const level = result.level;
        const confidence = result.confidence;
        
        // 级别颜色映射
        const levelColors = {
            "HSK1": "#93c5fd", "HSK2": "#60a5fa", "HSK3": "#3b82f6",
            "HSK4": "#fbbf24", "HSK5": "#f59e0b", "HSK6": "#d97706",
            "HSK7-9": "#dc2626", "超纲": "#9ca3af"
        };
        
        const levelColor = levelColors[level] || "#9ca3af";
        
        // 置信度百分比
        const confidencePercent = Math.round(confidence * 100);
        
        // 置信度条颜色
        let confidenceColor;
        if (confidencePercent >= 80) {
            confidenceColor = "#10b981";
        } else if (confidencePercent >= 60) {
            confidenceColor = "#f59e0b";
        } else
            confidenceColor = "#ef4444";
        
        // 生成概率网格
        let probabilitiesHTML = '';
        if (result.probabilities) {
            probabilitiesHTML = '<div class="probabilities-grid">';
            for (const [lvl, prob] of Object.entries(result.probabilities)) {
                const probPercent = Math.round(prob * 100);
                probabilitiesHTML += `
                    <div class="probability-item">
                        <div class="probability-level">${lvl}</div>
                        <div class="probability-value">${probPercent}%</div>
                    </div>
                `;
            }
            probabilitiesHTML += '</div>';
        }
        
        return `
            <div class="fade-in">
                <div style="text-align: center; margin-bottom: 20px;">
                    <div class="level-badge" style="background: ${levelColor}; font-size: 1.4rem;">
                        <i class="fas fa-certificate"></i> ${level}
                    </div>
                    <p style="color: #718096; margin-top: 5px;">预测置信度</p>
                </div>
                
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%; background: ${confidenceColor};"></div>
                </div>
                <div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 20px;">
                    ${confidencePercent}%
                </div>
                
                ${probabilitiesHTML}
                
                <div style="margin-top: 25px; padding: 15px; background: #f7fafc; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #4a5568;">
                        <i class="fas fa-info-circle"></i> 分析说明
                    </h4>
                    <p style="margin-bottom: 0; color: #718096;">
                        ${result.note || '基于AI模型的文本难度分析'}
                    </p>
                </div>
            </div>
        `;
    },
    
    // 生成分析结果的HTML
    generateAnalysisHTML: function(analysisData) {
        if (!analysisData.success) {
            return `
                <div style="text-align: center; padding: 40px; color: #f56565;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 15px;"></i>
                    <h3>完整分析失败</h3>
                    <p>${analysisData.error || '未知错误'}</p>
                </div>
            `;
        }
        
        const result = analysisData.result;
        const prediction = result.prediction;
        const analysis = result.analysis;
        
        const level = prediction.level;
        const confidence = prediction.confidence;
        
        // 级别颜色映射
        const levelColors = {
            "HSK1": "#93c5fd", "HSK2": "#60a5fa", "HSK3": "#3b82f6",
            "HSK4": "#fbbf24", "HSK5": "#f59e0b", "HSK6": "#d97706",
            "HSK7-9": "#dc2626", "超纲": "#9ca3af"
        };
        
        const levelColor = levelColors[level] || "#9ca3af";
        
        // 构建HTML
        let html = `
            <div class="fade-in">
                <div style="text-align: center; margin-bottom: 20px;">
                    <div class="level-badge" style="background: ${levelColor}; font-size: 1.4rem;">
                        <i class="fas fa-certificate"></i> ${level}
                    </div>
                    <p style="color: #718096; margin-top: 5px;">完整分析结果</p>
                </div>
                
                <div class="info-grid" style="margin-bottom: 20px;">
                    <div class="info-card">
                        <div class="value">${analysis.features.text_length || 0}</div>
                        <div class="label">文本长度</div>
                    </div>
                    <div class="info-card">
                        <div class="value">${analysis.features.chinese_char_count || 0}</div>
                        <div class="label">中文字符</div>
                    </div>
                    <div class="info-card">
                        <div class="value">${analysis.features.estimated_hsk_level || '超纲'}</div>
                        <div class="label">估算级别</div>
                    </div>
                    <div class="info-card">
                        <div class="value">${Math.round(confidence * 100)}%</div>
                        <div class="label">置信度</div>
                    </div>
                </div>
        `;
        
        // 显示文本内容
        if (result.display_text && Array.isArray(result.display_text)) {
            html += `
                <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #4a5568;">
                        <i class="fas fa-file-alt"></i> 详细分析报告
                    </h4>
                    <div style="max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.9rem; line-height: 1.5;">
            `;
            
            for (const line of result.display_text) {
                html += `<div style="margin-bottom: 5px;">${line}</div>`;
            }
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // 如果有颜色高亮的文本
        if (result.colored_text) {
            html += `
                <div style="margin-top: 20px;">
                    <h4 style="color: #4a5568;">
                        <i class="fas fa-paint-brush"></i> 文本难度标注
                    </h4>
                    <div style="padding: 15px; background: white; border-radius: 8px; border: 1px solid #e5e7eb; max-height: 200px; overflow-y: auto;">
                        ${result.colored_text}
                    </div>
                </div>
            `;
        }
        
        html += `
                <div style="margin-top: 20px; padding: 15px; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <h4 style="margin-top: 0; color: #1e40af;">
                        <i class="fas fa-lightbulb"></i> 分析说明
                    </h4>
                    <p style="margin-bottom: 0; color: #4b5563;">
                        完整分析包含了文本特征统计、HSK级别预测和词汇分析。
                        ${prediction.note ? `<br><small style="color: #6b7280;">${prediction.note}</small>` : ''}
                    </p>
                </div>
            </div>
        `;
        
        return html;
    },
    
    // 生成搭配检索结果的HTML
    generateCollocationHTML: function(collocationData) {
        if (!collocationData.success) {
            return `
                <div style="text-align: center; padding: 40px; color: #f56565;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 15px;"></i>
                    <h3>搭配检索失败</h3>
                    <p>${collocationData.error || '未知错误'}</p>
                </div>
            `;
        }
        
        const result = collocationData.result;
        const word = result.word;
        const collocations = result.collocations || [];
        const examples = result.examples || [];
        const totalCollocations = result.total_collocations || 0;
        const totalExamples = result.total_examples || 0;
        
        let html = `
            <div class="fade-in">
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="display: inline-block; padding: 10px 25px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border-radius: 20px; font-size: 1.2rem;">
                        <i class="fas fa-link"></i> "${word}" 搭配检索
                    </div>
                </div>
                
                <div class="info-grid" style="margin-bottom: 20px;">
                    <div class="info-card">
                        <div class="value">${totalCollocations}</div>
                        <div class="label">搭配数量</div>
                    </div>
                    <div class="info-card">
                        <div class="value">${totalExamples}</div>
                        <div class="label">例句数量</div>
                    </div>
                    <div class="info-card">
                        <div class="value">
                            ${totalCollocations > 0 ? '<i class="fas fa-check-circle" style="color: #10b981;"></i>' : '<i class="fas fa-times-circle" style="color: #dc2626;"></i>'}
                        </div>
                        <div class="label">数据状态</div>
                    </div>
                    <div class="info-card">
                        <div class="value">
                            ${result.note && result.note.includes('备用') ? '<i class="fas fa-shield-alt" style="color: #f59e0b;"></i>' : '<i class="fas fa-database" style="color: #3b82f6;"></i>'}
                        </div>
                        <div class="label">数据来源</div>
                    </div>
                </div>
        `;
        
        // 显示搭配
        if (collocations.length > 0) {
            html += `
                <div style="margin-bottom: 25px;">
                    <h4 style="color: #4a5568; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-bottom: 15px;">
                        <i class="fas fa-project-diagram"></i> 搭配模式
                    </h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px;">
            `;
            
            for (const colloc of collocations) {
                const leftPart = colloc.left ? `<span style="color: #6b7280;">${colloc.left}</span>` : '';
                const rightPart = colloc.right ? `<span style="color: #6b7280;">${colloc.right}</span>` : '';
                
                html += `
                    <div style="flex: 1 1 calc(50% - 12px); min-width: 200px; padding: 15px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; transition: all 0.3s;">
                        <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 10px; text-align: center;">
                            ${leftPart}<span style="color: #dc2626; font-weight: bold;">${word}</span>${rightPart}
                        </div>
                        ${colloc.example ? `
                            <div style="color: #6b7280; font-size: 0.9rem; padding: 10px; background: #f9fafb; border-radius: 6px; margin-top: 8px; border-left: 3px solid #3b82f6;">
                                <i class="fas fa-quote-left" style="margin-right: 5px; color: #9ca3af;"></i>${colloc.example}
                            </div>
                        ` : ''}
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // 显示例句
        if (examples.length > 0) {
            html += `
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #4a5568; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-bottom: 15px;">
                        <i class="fas fa-comment-alt"></i> 例句示范 (${examples.length}个)
                    </h4>
                    <div style="max-height: 300px; overflow-y: auto; padding: 10px;">
            `;
            
            for (let i = 0; i < examples.length; i++) {
                // 高亮目标词
                const highlightedExample = examples[i].replace(
                    new RegExp(word, 'g'), 
                    `<span style="color: #dc2626; font-weight: bold; background: rgba(220, 38, 38, 0.1); padding: 2px 4px; border-radius: 3px;">${word}</span>`
                );
                
                html += `
                    <div style="padding: 12px; margin-bottom: 10px; background: ${i % 2 === 0 ? '#f8fafc' : 'white'}; border-radius: 6px; border-left: 4px solid #93c5fd;">
                        <div style="display: flex; align-items: flex-start;">
                            <div style="width: 30px; font-weight: bold; color: #4b5563; flex-shrink: 0;">${i + 1}.</div>
                            <div style="flex-grow: 1; line-height: 1.6;">${highlightedExample}</div>
                        </div>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // 如果没有任何数据
        if (collocations.length === 0 && examples.length === 0) {
            html += `
                <div style="text-align: center; padding: 40px; color: #6b7280;">
                    <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 15px;"></i>
                    <h3>未找到搭配数据</h3>
                    <p>数据库中未找到"${word}"的搭配信息</p>
                    <div style="margin-top: 20px; padding: 15px; background: #f9fafb; border-radius: 8px; display: inline-block;">
                        <p style="margin: 0; font-size: 0.9rem;">${result.note || '请尝试其他词语'}</p>
                    </div>
                </div>
            `;
        }
        
        html += `
                <div style="margin-top: 25px; padding: 15px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 8px; border: 1px solid #7dd3fc;">
                    <h4 style="margin-top: 0; color: #0369a1;">
                        <i class="fas fa-info-circle"></i> 使用说明
                    </h4>
                    <div style="color: #1e40af; line-height: 1.6;">
                        <p><strong>💡 教学建议：</strong></p>
                        <ul style="margin-left: 20px;">
                            <li>使用这些搭配设计词汇练习</li>
                            <li>通过例句让学生理解词语用法</li>
                            <li>设计填空练习，巩固搭配记忆</li>
                            <li>组织小组对话，运用目标搭配</li>
                        </ul>
                        ${result.note ? `<p style="color: #6b7280; font-size: 0.9rem; margin-top: 10px;"><i class="fas fa-sticky-note"></i> ${result.note}</p>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        return html;
    },
    
    // 生成增强分析结果的HTML
    generateEnhancedAnalysisHTML: function(enhancedData) {
        if (!enhancedData.success) {
            return `
                <div style="text-align: center; padding: 40px; color: #f56565;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 15px;"></i>
                    <h3>增强分析失败</h3>
                    <p>${enhancedData.error || '未知错误'}</p>
                </div>
            `;
        }
        
        const result = enhancedData.result;
        const performance = result.performance || {};
        const modelInfo = this.getModelInfo();
        
        let html = `
            <div class="fade-in">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="color: #1e40af; margin-bottom: 10px;">
                        <i class="fas fa-robot"></i> AI增强分析报告
                    </h2>
                    <div style="display: inline-flex; align-items: center; gap: 10px; padding: 10px 20px; background: linear-gradient(135deg, ${modelInfo.color} 0%, ${modelInfo.color}99 100%); color: white; border-radius: 25px;">
                        <i class="fas fa-microchip"></i>
                        <span>使用模型: ${modelInfo.name}</span>
                    </div>
                    <p style="margin-top: 10px; color: #6b7280;">${modelInfo.description}</p>
                </div>
        `;
        
        // 性能统计
        if (performance.total_time) {
            html += `
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="display: inline-block; padding: 8px 16px; background: #f0f9ff; border-radius: 15px; color: #0369a1;">
                        <i class="fas fa-stopwatch"></i> 分析耗时: ${performance.total_time.toFixed(1)}秒
                    </div>
                </div>
            `;
        }
        
        // 显示增强分析内容
        if (result.enhanced_analysis) {
            const enhanced = result.enhanced_analysis;
            
            if (enhanced.theme_keywords?.theme) {
                html += `
                    <div style="margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 10px;">
                        <h4 style="color: #0369a1; margin-bottom: 10px;">
                            <i class="fas fa-bullseye"></i> 文本主题
                        </h4>
                        <p style="font-size: 1.2rem; font-weight: 600; color: #1e293b;">${enhanced.theme_keywords.theme}</p>
                    </div>
                `;
            }
            
            // 教学关键词显示 - 点击跳转百度搜索
            if (enhanced.theme_keywords?.keywords && enhanced.theme_keywords.keywords.length > 0) {
                html += `
                    <div style="margin-bottom: 20px; padding: 20px; background: white; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h4 style="color: #4a5568; margin-bottom: 15px;">
                            <i class="fas fa-key"></i> 教学关键词
                        </h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                `;
                
                enhanced.theme_keywords.keywords.forEach((keyword, index) => {
                    if (index < 8) { // 最多显示8个关键词
                        // 创建搜索链接（百度搜索）
                        const searchUrl = `https://www.baidu.com/s?wd=国际中文教学+${keyword}+教学资源`;
                        html += `
                            <a href="${searchUrl}" target="_blank" style="text-decoration: none;">
                                <div style="padding: 8px 15px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 20px; border: 1px solid #7dd3fc; color: #0369a1; font-weight: 600; cursor: pointer; transition: all 0.3s; display: inline-flex; align-items: center; gap: 5px;">
                                    <i class="fas fa-search"></i> ${keyword}
                                </div>
                            </a>
                        `;
                    }
                });
                
                html += `
                        </div>
                        <p style="margin-top: 15px; color: #6b7280; font-size: 0.9rem;">
                            <i class="fas fa-info-circle"></i> 点击关键词可快速搜索相关教学资源
                        </p>
                    </div>
                `;
            }
            
            if (enhanced.teaching_analysis?.summary) {
                html += `
                    <div style="margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 10px; border: 1px solid #fbbf24;">
                        <h4 style="color: #92400e; margin-bottom: 10px;">
                            <i class="fas fa-chalkboard-teacher"></i> Qwen3:8b智能教学分析
                        </h4>
                        <div style="color: #92400e; line-height: 1.6; white-space: pre-line; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;">
                            ${this.formatTeachingAnalysis(enhanced.teaching_analysis.summary)}
                        </div>
                    </div>
                `;
            }
            
            if (enhanced.vocabulary_analysis?.difficult_words && enhanced.vocabulary_analysis.difficult_words.length > 0) {
                html += `
                    <div style="margin-bottom: 20px; padding: 20px; background: white; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h4 style="color: #4a5568; margin-bottom: 15px;">
                            <i class="fas fa-exclamation-circle"></i> N+1生字词识别
                        </h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                `;
                
                enhanced.vocabulary_analysis.difficult_words.forEach((word, index) => {
                    if (index < 8) {
                        html += `
                            <div style="padding: 10px 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid ${this.getHSKColor(word.level)};">
                                <div style="font-size: 1.2rem; font-weight: bold; color: #1e293b;">${word.word}</div>
                                <div style="font-size: 0.85rem; color: #6b7280;">${word.level || '超纲'} | ${word.word_type || '未知类型'}</div>
                            </div>
                        `;
                    }
                });
                
                html += `
                        </div>
                        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #6b7280;">
                                <span>总生词: ${enhanced.vocabulary_analysis.total_difficult_words || 0}个</span>
                                <span>目标级别: ${enhanced.vocabulary_analysis.target_level_words || 0}个</span>
                                <span>超纲词汇: ${enhanced.vocabulary_analysis.other_difficult_words || 0}个</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        html += `
                <div style="margin-top: 25px; padding: 15px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 10px; text-align: center;">
                    <div style="display: inline-flex; align-items: center; gap: 10px; padding: 10px 20px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border-radius: 25px; margin-bottom: 10px;">
                        <i class="fas fa-check-circle"></i>
                        <strong>AI增强分析完成</strong>
                    </div>
                    <div style="margin-top: 10px; color: #0369a1; font-size: 0.9rem;">
                        模型: ${modelInfo.name} | 分析深度: 完整 | 教学建议: AI生成
                    </div>
                </div>
            </div>
        `;
        
        return html;
    },
    
    // 格式化教学分析文本，添加更好的排版
    formatTeachingAnalysis: function(text) {
        if (!text) return '';
        
        // 替换标题格式
        let formatted = text
            .replace(/^# (.+)$/gm, '<h4 style="color: #1e40af; margin-top: 20px; margin-bottom: 10px; font-size: 1.2rem;">$1</h4>')
            .replace(/^## (.+)$/gm, '<h5 style="color: #3b82f6; margin-top: 15px; margin-bottom: 8px; font-size: 1.1rem;">$1</h5>')
            .replace(/^### (.+)$/gm, '<h6 style="color: #6b7280; margin-top: 12px; margin-bottom: 6px; font-weight: 600;">$1</h6>');
        
        // 替换列表项
        formatted = formatted
            .replace(/^\d+\.\s+(.+)$/gm, '<div style="margin-left: 20px; margin-bottom: 8px; padding-left: 10px; border-left: 3px solid #93c5fd;"><strong>$1</strong></div>')
            .replace(/^[-•]\s+(.+)$/gm, '<div style="margin-left: 20px; margin-bottom: 6px; padding-left: 10px;">• $1</div>');
        
        // 添加段落间距
        formatted = formatted
            .replace(/\n\n/g, '</div><div style="margin-bottom: 15px;">')
            .replace(/\n/g, '<br>');
        
        return `<div style="line-height: 1.6;">${formatted}</div>`;
    },
    
    // 获取HSK颜色
    getHSKColor: function(level) {
        const colorMap = {
            'HSK1': '#10b981', 'HSK2': '#3b82f6', 'HSK3': '#8b5cf6',
            'HSK4': '#ec4899', 'HSK5': '#f59e0b', 'HSK6': '#ef4444',
            'HSK7-9': '#dc2626', '超纲': '#9ca3af'
        };
        return colorMap[level] || '#9ca3af';
    }
};

// ============================================
// 高级HTML生成函数
// ============================================

/**
 * 生成高级快速分析HTML报告
 */
function generatePremiumResultHTML(result) {
    const analysis = result.analysis || result;
    const predictedLevel = analysis.predicted_level || 'HSK3';
    const confidence = analysis.confidence || 0.85;
    const levelNum = parseInt(predictedLevel.replace('HSK', '')) || 3;
    const nPlus1Level = levelNum < 6 ? `HSK${levelNum + 1}` : 'HSK6';
    
    return `
    <div class="enhanced-report fade-in">
        <div class="enhanced-header">
            <h2><i class="fas fa-chart-line"></i> 快速分析报告</h2>
            <p>基于HSK标准的高效文本难度评估</p>
        </div>
        
        <div class="hsk-level-card">
            <div class="hsk-level-header">
                <div class="hsk-level-title">
                    <i class="fas fa-bolt"></i>
                    <span>快速评估结果</span>
                </div>
                <div class="hsk-level-badge">${predictedLevel}</div>
            </div>
            
            <div class="probability-bar-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; color: rgba(255,255,255,0.9);">
                    <span>置信度</span>
                    <span><strong>${(confidence * 100).toFixed(1)}%</strong></span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${confidence * 100}%"></div>
                </div>
            </div>
            
            <div class="hsk-level-details">
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">目标等级</div>
                    <div class="hsk-detail-value">${predictedLevel}</div>
                </div>
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">N+1目标</div>
                    <div class="hsk-detail-value">${nPlus1Level}</div>
                </div>
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">适合学生</div>
                    <div class="hsk-detail-value">${predictedLevel}水平</div>
                </div>
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">分析速度</div>
                    <div class="hsk-detail-value">极快</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; text-align: center;">
                <div style="display: flex; justify-content: center; gap: 5px; flex-wrap: wrap;">
                    ${['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6'].map(level => {
                        const isCurrent = level === predictedLevel;
                        return `
                            <span class="hsk-tag hsk-tag-${level.toLowerCase()}" 
                                  style="${isCurrent ? 'transform: scale(1.1); box-shadow: 0 0 20px rgba(245, 158, 11, 0.6);' : ''}">
                                ${level}
                                ${isCurrent ? '<i class="fas fa-check" style="margin-left: 5px;"></i>' : ''}
                            </span>
                        `;
                    }).join('')}
                </div>
            </div>
        </div>
        
        <div class="analysis-section">
            <h4 class="section-title"><i class="fas fa-lightbulb"></i> 快速评估说明</h4>
            <div style="padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 10px; border: 1px solid #7dd3fc;">
                <div style="line-height: 1.6; color: #0369a1;">
                    <p style="margin-bottom: 10px;"><strong>📊 分析特点：</strong></p>
                    <ul style="margin-left: 20px; margin-bottom: 15px;">
                        <li>基于HSK词汇标准快速匹配</li>
                        <li>毫秒级响应速度</li>
                        <li>准确度高达${(confidence * 100).toFixed(0)}%</li>
                        <li>适合教学中的快速评估需求</li>
                    </ul>
                    <p><strong>🎯 教学建议：</strong>此文本适合${predictedLevel}级别的学生，建议作为${predictedLevel}的课堂材料或练习。</p>
                </div>
            </div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="value">${predictedLevel}</div>
                <div class="label">推荐等级</div>
            </div>
            <div class="info-card">
                <div class="value">${nPlus1Level}</div>
                <div class="label">进阶目标</div>
            </div>
            <div class="info-card">
                <div class="value">${(confidence * 100).toFixed(0)}%</div>
                <div class="label">置信水平</div>
            </div>
            <div class="info-card">
                <div class="value"><i class="fas fa-bolt"></i></div>
                <div class="label">快速模式</div>
            </div>
        </div>
    </div>`;
}

/**
 * 生成高级完整分析HTML报告
 */
function generatePremiumAnalysisHTML(result) {
    const analysis = result.analysis || result;
    const predictedLevel = analysis.predicted_level || 'HSK3';
    const confidence = analysis.confidence || 0.92;
    const levelNum = parseInt(predictedLevel.replace('HSK', '')) || 3;
    const nPlus1Level = levelNum < 6 ? `HSK${levelNum + 1}` : 'HSK6';
    
    // 从分析结果中提取更多数据
    const wordCount = analysis.word_count || analysis.text_length || 0;
    const sentenceCount = analysis.sentence_count || 0;
    const hskDistribution = analysis.hsk_distribution || analysis.distribution || {};
    const uniqueWords = analysis.unique_words || analysis.vocabulary_count || 0;
    
    // 生成HSK分布显示
    const hskLevels = ['HSK1', 'HSK2', 'HSK3', 'HSK4', 'HSK5', 'HSK6', '超纲'];
    let hskDistributionHTML = '';
    
    if (Object.keys(hskDistribution).length > 0) {
        hskDistributionHTML = hskLevels.map(level => {
            const count = hskDistribution[level] || 0;
            const percentage = wordCount > 0 ? Math.round((count / wordCount) * 100) : 0;
            const color = HSKClient.getHSKColor(level);
            return `
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="hsk-tag hsk-tag-${level.toLowerCase()}" style="margin: 0 10px 0 0; min-width: 60px; text-align: center; background: ${color}; color: white;">
                        ${level}
                    </span>
                    <div style="flex-grow: 1; background: #e5e7eb; height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="height: 100%; background: ${color}; width: ${Math.min(percentage * 2, 100)}%;"></div>
                    </div>
                    <span style="margin-left: 10px; font-weight: 600; min-width: 40px; text-align: right;">${count} (${percentage}%)</span>
                </div>
            `;
        }).join('');
    } else {
        hskDistributionHTML = '<div style="padding: 15px; background: #f9fafb; border-radius: 8px; color: #6b7280; text-align: center;">正在分析词汇分布...</div>';
    }
    
    return `
    <div class="enhanced-report fade-in">
        <div class="enhanced-header">
            <h2><i class="fas fa-search"></i> 完整分析报告</h2>
            <p>全面深入的文本教学价值评估</p>
        </div>
        
        <div class="theme-keywords-container">
            <div class="theme-box">
                <h4><i class="fas fa-chart-pie"></i> 文本概览</h4>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>总字数:</span>
                        <span style="font-weight: 600;">${wordCount}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>句子数量:</span>
                        <span style="font-weight: 600;">${sentenceCount}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>独特词汇:</span>
                        <span style="font-weight: 600;">${uniqueWords}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>词汇密度:</span>
                        <span style="font-weight: 600;">${wordCount > 0 ? ((wordCount / sentenceCount).toFixed(1)) : 0}</span>
                    </div>
                </div>
            </div>
            
            <div class="keywords-box">
                <h4><i class="fas fa-layer-group"></i> HSK词汇分布</h4>
                <div style="margin-top: 15px;">
                    ${hskDistributionHTML}
                </div>
            </div>
        </div>
        
        <div class="hsk-level-card">
            <div class="hsk-level-header">
                <div class="hsk-level-title">
                    <i class="fas fa-graduation-cap"></i>
                    <span>综合等级评估</span>
                </div>
                <div class="hsk-level-badge">${predictedLevel}</div>
            </div>
            
            <div class="probability-bar-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; color: rgba(255,255,255,0.9);">
                    <span>综合置信度</span>
                    <span><strong>${(confidence * 100).toFixed(1)}%</strong></span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${confidence * 100}%"></div>
                </div>
            </div>
            
            <div class="hsk-level-details">
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">核心等级</div>
                    <div class="hsk-detail-value">${predictedLevel}</div>
                </div>
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">扩展等级</div>
                    <div class="hsk-detail-value">${nPlus1Level}</div>
                </div>
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">教学难度</div>
                    <div class="hsk-detail-value">
                        ${levelNum <= 2 ? '初级' : levelNum <= 4 ? '中级' : '高级'}
                    </div>
                </div>
                <div class="hsk-detail-item">
                    <div class="hsk-detail-label">分析深度</div>
                    <div class="hsk-detail-value">完整</div>
                </div>
            </div>
        </div>
        
        <div class="analysis-section">
            <h4 class="section-title"><i class="fas fa-book-open"></i> 教学价值分析</h4>
            <div class="teaching-suggestion-card">
                <h5><i class="fas fa-chalkboard-teacher"></i> 课堂教学应用</h5>
                <div class="teaching-suggestion-content">
                    <p>此文本作为${predictedLevel}级别的教学材料，具有以下特点：</p>
                    <ul class="advice-list">
                        <li class="advice-item"><span class="advice-item-number">1</span> 词汇难度适中，符合${predictedLevel}水平要求</li>
                        <li class="advice-item"><span class="advice-item-number">2</span> 句式结构清晰，适合语法讲解</li>
                        <li class="advice-item"><span class="advice-item-number">3</span> 包含${nPlus1Level}级别词汇，适合扩展教学</li>
                        <li class="advice-item"><span class="advice-item-number">4</span> 文本长度适合单次课程使用</li>
                    </ul>
                </div>
            </div>
            
            <div class="teaching-suggestion-card">
                <h5><i class="fas fa-user-graduate"></i> 学习建议</h5>
                <div class="teaching-suggestion-content">
                    <ul class="advice-list">
                        <li class="advice-item"><span class="advice-item-number">1</span> 对于${predictedLevel}学生：可作为精读材料，掌握核心词汇</li>
                        <li class="advice-item"><span class="advice-item-number">2</span> 对于${nPlus1Level}学生：可作为快速阅读材料，挑战更高难度</li>
                        <li class="advice-item"><span class="advice-item-number">3</span> 建议学习时间：20-30分钟</li>
                        <li class="advice-item"><span class="advice-item-number">4</span> 配套练习：词汇填空、句子重组、阅读理解</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="analysis-section" style="background: linear-gradient(135deg, #f8f9fa 0%, #f1f5f9 100%);">
            <h4 class="section-title"><i class="fas fa-chart-bar"></i> 分析总结</h4>
            <div class="info-grid">
                <div class="info-card">
                    <div class="value">${predictedLevel}</div>
                    <div class="label">核心等级</div>
                </div>
                <div class="info-card">
                    <div class="value">${nPlus1Level}</div>
                    <div class="label">扩展目标</div>
                </div>
                <div class="info-card">
                    <div class="value">${wordCount}</div>
                    <div class="label">总字数</div>
                </div>
                <div class="info-card">
                    <div class="value">${sentenceCount}</div>
                    <div class="label">句子数</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 10px; border: 1px solid #e5e7eb; text-align: center;">
                <div style="display: inline-flex; align-items: center; gap: 10px; padding: 10px 20px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border-radius: 25px; margin-bottom: 10px;">
                    <i class="fas fa-check-circle"></i>
                    <strong>完整分析完成</strong>
                </div>
                <div style="color: #6b7280; font-size: 0.95rem; line-height: 1.6;">
                    基于HSK标准的多维度分析，提供全面的教学参考价值。建议将此文本作为${predictedLevel}级别的核心教学材料。
                </div>
            </div>
        </div>
    </div>`;
}

// ============================================
// 页面事件处理函数
// ============================================

// 绑定搭配检索按钮
document.addEventListener('DOMContentLoaded', function() {
    // 检查是否有搭配检索按钮
    const searchButton = document.getElementById('searchCollocationBtn');
    if (searchButton) {
        searchButton.addEventListener('click', function() {
            const wordInput = document.getElementById('collocationWord');
            if (!wordInput) return;
            
            const word = wordInput.value.trim();
            if (!word) {
                alert('请输入要检索的词语');
                return;
            }
            
            const resultDiv = document.getElementById('collocationResult');
            if (!resultDiv) return;
            
            // 显示加载状态
            resultDiv.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #3b82f6;">
                    <div class="spinner" style="width: 40px; height: 40px; border: 3px solid rgba(59, 130, 246, 0.3); border-top-color: #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px;"></div>
                    <h4>正在检索搭配词库...</h4>
                    <p>请稍候，正在为您查找"${word}"的相关搭配</p>
                </div>
            `;
            
            // 调用API
            hskClient.getCollocation(word).then(data => {
                resultDiv.innerHTML = hskClient.generateCollocationHTML(data);
            }).catch(error => {
                resultDiv.innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #f56565;">
                        <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 15px;"></i>
                        <h3>检索失败</h3>
                        <p>${error.message || '网络错误'}</p>
                    </div>
                `;
            });
        });
    }
});

// 创建全局变量
window.hskClient = HSKClient;

// 添加到现有的hskClient对象中
hskClient.generatePremiumResultHTML = generatePremiumResultHTML;
hskClient.generatePremiumAnalysisHTML = generatePremiumAnalysisHTML;

// 页面加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        HSKClient.init();
    });
} else {
    HSKClient.init();
}