# 通用多模态 RAG 框架 (Medical Multimodal RAG Example)

本项目是一个**通用多模态知识库 RAG 框架**，当前以 IU X-Ray 医学影像与报告数据作为示例场景，演示从数据清洗、图文嵌入、向量检索、关键词召回、融合排序、证据增强生成到 Web/API 服务化的完整流程。

> v2.0 定位：医学影像只是当前样例数据源，项目核心是可迁移到企业知识库、客服问答、论文问答、法务/财务文档等场景的多模态 RAG 工程框架。

**声明**：当前示例涉及医学影像，但本项目仅用于学术研究与工程演示，不可替代医生诊断。所有分析结果仅供参考。

---

## 目录结构

```
medical_multimodal_rag/
├── main.py                      # Web 前端入口（Gradio）
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── LICENSE                      # MIT 许可证
├── src/                         # 核心模块
│   ├── __init__.py
│   ├── config.py               # 配置管理（路径、模型、阈值）
│   ├── clip_encoder.py         # 多模态编码器（图像/文本）
│   ├── vector_store.py         # FAISS 向量索引管理
│   ├── data_builder.py         # 数据管线（配对、编码、索引、阈值估计）
│   ├── rag_pipeline.py         # 检索和生成核心类
│   ├── service.py              # Gradio/API 共享服务层
│   ├── llm_backend.py          # 本地/远程 LLM 后端适配
│   └── api.py                  # FastAPI 服务入口
├── scripts/
│   ├── build_assets.py         # 数据构建脚本的主入口
│   └── evaluate_retrieval.py   # 离线检索评测脚本
├── notebooks/                  # Jupyter 实验代码（已迁移到模块）
├── data/
│   └── iu_xray/               # 数据存放目录
│       ├── indiana_reports.csv
│       ├── indiana_projections.csv
│       ├── image_report_pairs.json      # 构建的配对文件
│       ├── images/
│       │   └── images_normalized/       # 标准化X光图像
│       ├── valid_pairs_full.json        # 全量有效配对
│       ├── image_vectors_full.npy       # 全量图像向量
│       ├── text_vectors_full.npy        # 全量文本向量
│       ├── faiss_index_full.bin         # FAISS 索引
│       ├── id_mapping_full.json         # ID 映射
└──     └── score_stats_full.json        # 距离分布统计 与 推荐阈值

```

---

## 核心功能

1. **图像上传 + 问题提问** → 以胸部 X 光为示例，支持图像检索、文本问答和图文联合检索
2. **多路召回** → 支持 image-only、text-only、hybrid 检索；hybrid 融合图像向量、文本向量和 BM25 关键词召回
3. **证据增强生成** → 返回带 metadata 的证据片段，LLM 只能基于 retrieved evidence 组织回答
4. **轻量重排与置信度校准** → 先召回 top-N，再基于文本语义相似度和关键词分数重排；低置信样本拒答
5. **工程化接口与评测** → 保留 Gradio Demo，同时提供 FastAPI `POST /query` 和离线检索评测脚本

---

## 技术栈

| 组件 | 技术/模型 | 说明 |
|------|---------|------|
| **视觉编码** | OpenCLIP ViT-B-32 | 512 维图像特征 |
| **文本编码** | OpenCLIP ViT-B-32 | 512 维文本特征 |
| **向量检索** | FAISS IndexFlatL2 | GPU 加速，L2 距离 |
| **语言模型** | Qwen 2.5 3B-Instruct | 中文优化，功率低 |
| **示例数据集** | IU X-Ray | 7466 张胸部 X 光 + 医学报告 |
| **前端框架** | Gradio 4.x | 无需前端开发 |
| **服务接口** | FastAPI | 提供可接入业务系统的 HTTP API |
| **向量数据库** | NumPy + FAISS | 本地索引 |

---

## 快速开始（以医学影像示例运行）

### 1. 环境安装

```bash
# 要求 Python 3.10+
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 先装 PyTorch CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 再装其他依赖
pip install -r requirements.txt
```

### 2. 数据准备

从 [Kaggle IU X-Ray 数据集](https://www.kaggle.com/raddar/chest-xrays-indiana-university) 下载，放入 `data/iu_xray/`：

```
data/iu_xray/
├── indiana_reports.csv
├── indiana_projections.csv
└── images/images_normalized/  # PNG 格式，标准化到 224×224
```

### 3. 数据构建（首次需要 10-30 分钟）

```bash
python scripts/build_assets.py --data-mode full --batch-size 32
```

**输出**：
- `valid_pairs_full.json` — 7466 条有效配对
- `image_vectors_full.npy` — 7466×512 图像向量
- `text_vectors_full.npy` — 7466×512 文本向量
- `faiss_index_full.bin` — FAISS 联合索引
- `id_mapping_full.json` — ID 映射表
- `score_stats_full.json` — 距离分布 & 推荐阈值

### 4. 启动 Web 界面

```bash
conda run -n medical_rag python main.py --data-mode full --llm-backend local
```

浏览器访问：http://localhost:7860

仅验证检索、证据和 prompt，不加载本地大模型：

```bash
conda run -n medical_rag python main.py --data-mode full --llm-backend none
```

### 5. 启动 FastAPI 服务

```bash
conda run -n medical_rag uvicorn src.api:app --host 127.0.0.1 --port 8000
```

`POST /query` 支持 `question`、可选 `image`、`top_k`、`retrieval_mode`。`retrieval_mode` 可选 `auto`、`image`、`text`、`hybrid`。

### 6. 离线检索评测

```bash
conda run -n medical_rag python scripts/evaluate_retrieval.py --data-mode full --sample-size 100 --top-k 3
```

评测会对比 `image`、`text`、`hybrid`、`hybrid+rerank`，输出 Recall@K、MRR、nDCG、正常/异常 Top1 命中率和平均延迟。

如果模型已缓存，可在演示前设置离线模式，避免 HuggingFace 网络探测拖慢启动：

```powershell
$env:HF_HUB_OFFLINE="1"
conda run -n medical_rag python main.py --data-mode full --llm-backend none
```

---

## 架构设计

### 数据流

```
CSV (reports + projections)
    ↓
[data_builder.py] build_image_report_pairs()
    ↓
image_report_pairs.json 
    ↓
[clip_encoder] encode_batch()
    ↓
image_vectors_full.npy + text_vectors_full.npy
    ↓
[vector_store] build_index()
    ↓
faiss_index_full.bin + id_mapping_full.json
    ↓
[rag_pipeline] retrieve() + generate()
    ↓
Web 输出
```

### 检索策略

v2 检索不再只依赖图像向量，而是支持三种模式：

| 模式 | 输入 | 说明 |
|------|------|------|
| **image** | 图像 | 适合相似病例检索 |
| **text** | 问题/报告文本 | 适合纯知识库问答 |
| **hybrid** | 图像 + 问题 | 融合图像向量、文本向量和 BM25，最接近真实业务 RAG |

召回后根据距离分布自动判断：

| 模式 | 触发条件 | 处理方式 |
|------|---------|---------|
| **low_confidence** | best_dist > p90 | 拒答：未找到相似病例 |
| **single_strong** | best_dist ≤ p90 && diff > (p80-p50) | 主要参考第1个病例 |
| **multi_similar** | 多个相似 | 综合参考前3个病例 |

---

## 自定义用法

### 自定义阈值

```bash
python main.py --data-mode full \
  --abs-threshold 0.08 \
  --diff-threshold 0.02 \
  --host 127.0.0.1 \
  --port 8000
```

### 样本模式（快速测试）

```bash
python scripts/build_assets.py --data-mode sample --batch-size 32
python main.py --data-mode sample
```

### 共享链接（临时演示）

```bash
python main.py --data-mode full --share
```

---

## 迁移到其他助手或知识库

本项目的医学影像数据只是示例。如果要改成其他助手，例如企业文档助手、课程问答助手、论文问答助手，可以按下面几个位置替换。

### 1. 更换数据库/数据源

当前数据源位于 `data/iu_xray/`，由 `src/data_builder.py` 读取 CSV 并生成图文配对。

迁移时需要准备统一的样本结构，至少包含：

```json
{
  "uid": "唯一ID",
  "image_path": "可选，多模态场景使用",
  "full_text": "可检索文本",
  "metadata": {
    "source": "文档来源",
    "title": "标题或业务字段"
  }
}
```

如果是纯文本知识库，可以不提供图像，只保留 `full_text` 和 metadata。

### 2. 更换嵌入方式

当前图像和文本嵌入都在 `src/clip_encoder.py`，默认使用 OpenCLIP ViT-B-32。

- 图文场景：可以继续使用 CLIP 类模型。
- 纯文本场景：建议替换为 bge、e5、gte 等文本 embedding 模型。
- 领域场景：可以替换为医学、法律、金融等领域 embedding。

替换后重新运行：

```bash
python scripts/build_assets.py --data-mode full --batch-size 32
```

### 3. 更换向量库

当前使用 `src/vector_store.py` 中的 FAISS 本地索引，适合个人开发和本地演示。

如果迁移到线上业务，可以把这一层替换为 Milvus、Qdrant、pgvector、Elasticsearch dense vector 等服务，保持 `retrieve()` 返回 evidence + metadata 即可。

### 4. 更换提示词

生成提示词位于 `src/rag_pipeline.py` 的 `build_prompt()`。

迁移到其他助手时，重点修改：

- 助手身份：例如课程助教、企业知识库助手、论文阅读助手。
- 回答边界：明确只能基于证据回答。
- 输出格式：是否需要引用来源、给步骤、给结论、给风险提示。
- 拒答策略：证据不足时如何说明。

### 5. 更换 LLM

LLM 后端位于 `src/llm_backend.py` 和 `src/service.py`。

当前支持：

- `--llm-backend local`：本地 Transformers 模型，如 Qwen。
- `--llm-backend api`：OpenAI-compatible API，如 aihubmix 等中转服务。
- `--llm-backend none`：不加载 LLM，只调试检索和 prompt。

本地运行示例：

```bash
conda run -n medical_rag python main.py --data-mode full --llm-backend local
```

API 运行时需要设置环境变量：

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://your-api-base/v1"
$env:OPENAI_MODEL="your-model"
conda run -n medical_rag python main.py --data-mode full --llm-backend api
```

---

## 优势

- 本项目实现了一个端到端的通用多模态 RAG 框架，从原始数据到图文编码、FAISS 索引构建、多路召回、融合排序、证据增强生成全流程打通。
- 支持 image-only、text-only、hybrid 三种检索模式，用户问题会进入检索阶段，而不只是进入最终 prompt。
- 提供离线检索评测脚本，可以对比不同检索策略的 Recall@K、MRR、nDCG 和延迟。
- 采用模块化设计，编码器、向量库、LLM、提示词和接口层都便于替换。
- 成本低，支持本地GPU运行，无需云服务费用；选用3B参数量的轻量级大模型，单张消费级显卡即可流畅推理。

---

## 限制与建议

### 当前限制与更改建议

- 算力瓶颈：建议之后采用批量编码加速或分布式构建。
- 数据规模：仅7,466条样本，泛化性未充分验证。生产环境建议接入MIMIC-CXR等更大规模医学数据集。
- 模型性能：3B模型生成质量一般，CLIP非医学专用。建议后续升级。
- 部署方式：当前提供 Gradio 和 FastAPI，仍属于本地演示级部署；生产环境建议增加鉴权、日志、监控和持久化任务队列。

- 替换编码器——`clip_encoder.py`
- 改LLM—— `src/llm_backend.py`
- 改向量库—— `vector_store.py`
- 改前端—— `main.py`

### 其他建议

本项目适合初学者理解 RAG 全流程，也适合用于展示 AI 应用工程中的检索、评测、证据约束和服务化能力。
若无网络代理，建议使用huggingface国内镜像网址部署模型。


---

## 相关资源与使用声明

- **CLIP 论文**：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.14030)
- **OpenCLIP**：[OpenCLIP: OpenAI's CLIP reproduced on open-source data](https://github.com/mlfoundations/open_clip)
- **FAISS**：[Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **Qwen**：[阿里巴巴开源 LLM](https://github.com/QwenLM/Qwen)
- **IU X-Ray**：[Indiana University Chest X-ray Collection](https://pubmed.ncbi.nlm.nih.gov/26133894/)


---

##  许可证

MIT License — 见 [LICENSE](LICENSE) 文件

---

## 更新情况

**v2.0** 将项目主题升级为“以医学影像为例的通用多模态 RAG 框架”。新增图文联合检索、BM25 关键词召回、融合排序、轻量重排、证据引用、离线评测、FastAPI 接口和 LLM 后端切换。

**v1.3** 原计划完善文搜文、文搜图能力，现已并入 v2.0。

**v1.2** 进一步更新了网页端设计。

**v1.1** 完成了网页端的构建，优化了RAG回答。

**v1.0** 初步完成医学影像相似病例 RAG Demo：上传胸片、检索相似病例、调用 LLM 生成分析。

---

**作者**：@Elliot Hou | **更新**：2026-05 | **版本**：v2.0
