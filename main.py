import argparse
import gradio as gr
from src.config import load_config
from src.service import infer, load_rag_system


def build_ui(rag):
    title = "🏥 医学影像智能分析系统"
    subtitle = "Medical Image RAG Assistant"
    
    desc = (
        "### 📋 系统说明\n\n"
        "上传胸部X光图像并提出您的问题，系统将智能检索相似病例并结合大语言模型生成专业分析报告。\n\n"
        "⚠️ **重要提示**：本系统仅用于学术研究和演示目的，**不可替代专业医生的诊断**。所有分析结果仅供参考。"
    )
    
    # 模型信息
    model_info = (
        "### 🔧 技术架构\n\n"
        "| 组件 | 模型/技术 |\n"
        "|------|-----------|\n"
        "| 视觉编码器 | CLIP (ViT-B-32) |\n"
        "| 文本编码器 | CLIP (ViT-B-32) |\n"
        "| 向量检索 | FAISS (GPU加速) |\n"
        "| 语言模型 | Qwen2.5-3B-Instruct |\n"
        "| 数据集 | IU X-Ray (Indiana University Chest X-ray Collection) |\n\n"
        "**开发者**：@Elliot Hou | 版本：v1.2 | 最后更新：2025-03"
    )

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin: 10px 0;
        }
        footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
        }
        """
    ) as demo:
        # 自定义 CSS 样式
        gr.HTML("""
        <style>
        .gradio-container {
            font-family: 'Inter', sans-serif;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5em;
            font-weight: bold;
        }
        </style>
        """)
        
        # 头部区域
        with gr.Row(elem_id="header"):
            gr.Markdown(f"# {title}\n## {subtitle}")
        
        # 系统说明
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(desc)
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 功能特点\n\n"
                           "- 🔍 智能检索相似病例\n"
                           "- 💬 自动生成诊断分析\n"
                           "- 📊 基于7,466+真实病例\n"
                           "- ⚡ GPU加速实时响应")
        
        gr.Markdown("---")
        
        # 主要交互区域
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="pil", 
                    label="📷 上传X光图像",
                    elem_id="image-input",
                    height=400,
                    show_label=True,
                    show_download_button=True,
                    show_share_button=True
                )
                
                with gr.Row():
                    # 示例图像
                    gr.Markdown("### 📂 示例图像")
                    example_images = gr.Examples(
                        examples=[
                            ["data/iu_xray/images/images_normalized/1_IM-0001-3001.dcm.png"],
                            ["data/iu_xray/images/images_normalized/1_IM-0001-4001.dcm.png"],
                        ],
                        inputs=img_input,
                        label="点击示例加载",
                        cache_examples=False
                    )
            
            with gr.Column(scale=1):
                q_input = gr.Textbox(
                    label="💬 您的问题",
                    placeholder="例如：这张片子有无肺部感染迹象？\n\n其他问题示例：\n- 心脏大小是否正常？\n- 肺部是否有阴影？\n- 气管位置有无异常？",
                    lines=5,
                    show_label=True,
                    container=True
                )
                
                with gr.Row():
                    run_btn = gr.Button(
                        "🚀 开始分析", 
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ 清空",
                        variant="secondary",
                        size="lg"
                    )
        
        gr.Markdown("---")
        
        # 结果展示区域
        with gr.Row():
            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label="📝 智能诊断分析",
                    lines=15,
                    show_label=True,
                    container=True,
                    placeholder="点击「开始分析」后，诊断结果将显示在这里..."
                )
            
            with gr.Column(scale=1):
                retrieval_output = gr.Textbox(
                    label="📚 检索到的相似病例",
                    lines=15,
                    show_label=True,
                    container=True,
                    placeholder="相似病例信息将显示在这里..."
                )
        
        # 使用说明和提示
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 如何使用本系统？
            
            1. **上传图像**：点击上传区域选择胸部X光图像，或使用示例图像
            2. **输入问题**：在文本框中输入您想了解的问题
            3. **开始分析**：点击「开始分析」按钮，等待系统处理
            4. **查看结果**：系统将显示智能诊断分析和相关相似病例
            
            ### 支持的提问类型
            
            - **诊断类**：这张片子显示什么异常？
            - **特征类**：心脏大小是否正常？肺部有无阴影？
            - **比较类**：和正常X光片相比有何差异？
            - **建议类**：需要进一步做什么检查？
            
            ### 注意事项
            
            - 系统基于 IU X-Ray 数据集训练，主要针对胸部X光片
            - 结果仅供参考，请勿用于临床诊断
            - 如有疑问，请咨询专业医生
            """)
        
        # 技术信息区域
        with gr.Accordion("🔧 技术详情", open=False):
            gr.Markdown(model_info)
        
        # 页脚
        gr.HTML("""
        <footer>
            <p>🏥 Medical Image RAG System | 基于检索增强生成的医学影像智能分析</p>
            <p>⚠️ 仅供学术研究 | 不可用于临床诊断 | 数据来源: IU X-Ray Dataset</p>
            <p style="font-size: 0.85em">© 2025 Medical Image RAG Project | MIT License</p>
        </footer>
        """)
        
        # 定义处理函数
        def process(image, question):
            return infer(rag, image, question)
        
        def clear_all():
            return None, "", "", ""
        
        # 绑定事件
        run_btn.click(
            fn=process,
            inputs=[img_input, q_input],
            outputs=[answer_output, retrieval_output],
            api_name="analyze"
        )
        
        # 清空按钮事件
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[img_input, q_input, answer_output, retrieval_output],
            api_name="clear"
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="医学影像 RAG 智能分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用示例:
          python main.py --data-mode full
          python main.py --data-mode full --port 8080 --share
          python main.py --data-mode full --abs-threshold 0.05 --diff-threshold 0.015
        """
    )
    parser.add_argument("--data-mode", default="full", choices=["sample", "full"],
                        help="数据模式: full(完整数据) / sample(采样数据)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批处理大小 (默认: 32)")
    parser.add_argument("--abs-threshold", type=float, default=None,
                        help="绝对相似度阈值 (默认: 自动计算)")
    parser.add_argument("--diff-threshold", type=float, default=None,
                        help="差异阈值 (默认: 自动计算)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="服务器主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务器端口 (默认: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="生成公网链接 (24小时有效)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = "full" if args.data_mode == "full" else "sample"

    cfg = load_config(
        data_mode=mode,
        batch_size=args.batch_size,
        abs_threshold=args.abs_threshold,
        diff_threshold=args.diff_threshold,
    )
    print(f"运行模式: {cfg.data_mode}")
    print(f"ABS_THRESHOLD: {cfg.abs_threshold}")
    print(f"DIFF_THRESHOLD: {cfg.diff_threshold}")

    rag = load_rag_system(cfg)
    app = build_ui(rag)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        debug=False
    )