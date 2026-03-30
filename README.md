# 医学影像智能分析系统 (Medical Image RAG Assistant)

本项目基于 IU X-Ray 数据集，实现了**端到端的医学影像检索增强生成（RAG）**功能。

**声明**：本项目仅用于学术研究与演示，不可替代医生诊断。所有分析结果仅供参考。

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
│   └── service.py              # HTTP 服务接口
├── scripts/
│   └── build_assets.py         # 数据构建脚本的主入口
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

1. **图像上传 + 问题提问** → 用户可上传胸部 X 光 PNG，输入任意问题
2. **相似病例检索** → 图像/文本联合索引，返回 Top-K 最相似病例（默认top-3）
3. **智能生成分析** → 基于检索病例与问题，LLM 生成中文诊断参考
4. **策略自适应** → 根据距离分布自动判断是否单强匹配、低置信拒答或多样本综合

---

## 技术栈

| 组件 | 技术/模型 | 说明 |
|------|---------|------|
| **视觉编码** | OpenCLIP ViT-B-32 | 512 维图像特征 |
| **文本编码** | OpenCLIP ViT-B-32 | 512 维文本特征 |
| **向量检索** | FAISS IndexFlatL2 | GPU 加速，L2 距离 |
| **语言模型** | Qwen 2.5 3B-Instruct | 中文优化，功率低 |
| **数据集** | IU X-Ray | 7466 张胸部 X 光 + 医学报告 |
| **前端框架** | Gradio 4.x | 无需前端开发 |
| **向量数据库** | NumPy + FAISS | 本地索引 |

---

## 快速开始

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
python main.py --data-mode full
```

浏览器访问：http://localhost:7860

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

检索后根据距离分布自动判断：

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

## 优势

- 本项目实现了一个端到端完整的医学影像RAG系统，从原始数据到向量编码、FAISS索引构建、相似病例检索直至大模型生成诊断分析全流程打通。
- 采用模块化设计，所有参数通过配置文件管理，便于扩展与调试。
- 成本低，支持本地GPU运行，无需云服务费用；选用3B参数量的轻量级大模型，单张消费级显卡即可流畅推理。

---

## 限制与建议

### 当前限制与更改建议

- 算力瓶颈：建议之后采用批量编码加速或分布式构建。
- 数据规模：仅7,466条样本，泛化性未充分验证。生产环境建议接入MIMIC-CXR等更大规模医学数据集。
- 模型性能：3B模型生成质量一般，CLIP非医学专用。建议后续升级。
- 部署方式：Gradio进程停止即服务中断，不支持分布式。建议转向FastAPI + React全栈架构

- 替换编码器——`clip_encoder.py`
- 改LLM—— `rag_pipeline.generate()`
- 改向量库—— `vector_store.py`
- 改前端—— `main.py`

### 其他建议

本项目适合初学者理解 RAG 全流程，用 Gradio + 轻量模型快速验证想法。
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


**v1.2** 进一步更新了网页端设计。

**v1.1** 完成了网页端的构建，优化了RAG回答。

**v1.0** 初步完成基本的RAG功能。

>v1.3 （进行中）将文搜文、文搜图功能完善、纳入 —— 用户可以在提供胸片的基础上，提供有效的辅助描述文字，以帮助系统更好地判断

---

**作者**：@Elliot Hou | **更新**：2025-03 | **版本**：v1.2
