# BigDataTCM
BigDataTCM(“大数中医”)是是由河南工业大学复杂性科学研究院与阿帕斯公司联合研发的一个中医垂直领域大模型，旨在将专业的中医药学知识、医疗信息、数据融会贯通，为中医医疗行业提供智能化的医疗问答、诊断支持和中医学知识等信息服务，提高诊疗效率和医疗服务质量。
# 更新日志
### ✨ Latest News
- [06/30/2024] We released the **quantitative version** of BigDataTCM-34B-chat-4bits.

# 目录
# 介绍
- 核心功能
  - **医学问答**：可以回答关于医学、健康、疾病等方面的问题，包括但不限于症状、治疗、药物、预防、检查等。
  - **语意理解**：理解医学术语、提供关键信息抽取和归类
  - **多轮对话**：可扮演各种医疗专业角色如医生与用户进行对话，根据上下文提供更加准确的答案。
  - **多场景支持**：支持中草药咨询、疾病症状解答、方剂知识查询、中医理论解读、有毒中药物警示、药物相互作用信息、疾病预防策略、在线问诊服务、康复期注意事项、食疗建议、中药茶饮推荐10大场景。
- 模型架构
  - 基于Transformer的340亿参数规模大语言模型, 训练采用Yi-34B作为基础预训练模型。
- 主要特点
  - 场景导向：对应多样化的医疗环境和实际需求，我们进行深度优化，量身打造解决方案，以实现更高效的落地应用。
  - 迭代优化：我们不断积累和吸纳最新医疗研究成果，以此不停地强化我们的模型能力和系统性能，以保持技术前沿地位。
# 推理
### Quantization Model

A quantized version of BigDataTCM is provided, allowing users with constrained memory or computing resources to access our BigDataTCM.
| Quantization          | Backbone      | Checkpoint |
| --------------------- | ------------- | ------------- |
| BigDataTCM-34B-chat-4bits        | Yi-34B        |  [HF Lnik](https://huggingface.co/BigDataTCM/BigDataTCM-34B-chat-4bits) |

### Model Inference

```bash
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("BigDataTCM-34B-chat-4bits", use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("BigDataTCM-34B-chat-4bits", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
messages = []
messages.append({"role": "user", "content": "阴阳失调是指什么？"})
response = model.HuatuoChat(tokenizer, messages)
print(response)
```
### 样例展示
# 训练数据
- 数据简介
  - <summary>总数据量：预训练数据约7G，中医药知识图谱、医学文献、权威中医教材及在线中医资源等3000万条;</summary>

- 继续预训练
  - 扩充模型的医疗知识库：超过百万条真实临床诊疗数据;全部中医药院校十三五规划教材,涵盖中医经典教科书等等

- 指令微调
  - 从知识图谱、临床诊疗数据、书籍等数据中自动化构建医疗指令集。
  - 立足于十大场景的指令精调/指令微调数据的构建、更好的服务于：中草药咨询、疾病症状解答、方剂知识查询、中医理论解读等等
  - 采用 Self-Instruct、Evol-Instruct 等方案，对指令集进行扩展以及丰富指令集多样化形式。

- 数据工程
  - 为了更好的对数据进行清洗、我们基于开源系统自研了一个统一的中间数据表示，支持多种文本输入格式，如txt、JSON、parquet等。同时封装了四大运算符池包括格式化器、映射器、过滤器和去重器。更好的服务于数据格式映射、多条件多场景文本过滤；文件数据去重。这些运算符为LLM数据处理提供了全面的功能。

## 评测

### 中医执业医师考试题评估

|                                                               模型                               | SCORE     |
| -------------------------------------------------------------------------------------------- | -------- | 
| [大数中医(BigDataTCM)](https://huggingface.co/BigDataTCM/BigDataTCM-34B-chat-4bits)                      |    80.3  | 
| [HuatuoGPT2](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-34B)        | 68.4     |
| [WiNGPT2](https://huggingface.co/winninghealth/WiNGPT2-14B-Chat) | 48.2     |
| [浦医2.0](https://huggingface.co/OpenMEDLab/PULSE-20bv5)               | 38.9     |
| [MMedLM](https://huggingface.co/Henrychur/MMed-Llama-3-8B)               | 33.6     |


## 应用
