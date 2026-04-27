import gradio as gr
import nibabel as nib

from vlm_client import vlm_infer
from registration.infer import run_registration
from registration.visualize import plot_slice

def chat_fn(image1, image2, text, history):

    if history is None:
        history = []

    reply = vlm_infer(image1, image2, text)

    history.append(
        {"role": "user", "content": text}
    )

    history.append(
        {"role": "assistant", "content": reply}
    )

    return history, history, ""


def reg_fn(moving_file, fixed_file):

    flow, moving, fixed = run_registration(moving_file, fixed_file)

    coronal = plot_slice(moving, fixed, flow, axis=1)
    sagittal = plot_slice(moving, fixed, flow, axis=0)
    axial = plot_slice(moving, fixed, flow, axis=2)

    return coronal, sagittal, axial


with gr.Blocks(theme=gr.themes.Base()) as demo:

    gr.Markdown("# 🏥 智慧医疗智能体平台 V1.0")

    with gr.Row():

        # 左侧 多模态模型
        with gr.Column():
            gr.Markdown("## 🔬 多模态医疗大模型")
            chatbot = gr.Chatbot(height=550, label="VLM多轮对话")
            state = gr.State([])
            with gr.Row():
                image1 = gr.Image(
                    type="pil",
                    label="图片1",
                    height=200
                )
                image2 = gr.Image(
                    type="pil",
                    label="图片2",
                    height=200
                )
            text = gr.Textbox(label="文本输入", placeholder="你好,有什么可以帮助你的?")
            send_btn = gr.Button("📡 发送", variant="primary")
            send_btn.click(
                chat_fn,
                inputs=[image1, image2, text, state],
                outputs=[chatbot, state]
            )

        # 右侧 医学影像配准
        with gr.Column():

            gr.Markdown("## 🚑 医学影像配准")
            with gr.Row():
                moving = gr.File(label="浮动图像", height=130)
                fixed = gr.File(label="固定图像", height=130)

            btn2 = gr.Button("🔬 配准", variant="primary")

            coronal = gr.Plot(label="冠状面", scale=0.8)

            sagittal = gr.Plot(label="矢状面", scale=0.8)

            axial = gr.Plot(label="轴向面", scale=0.8)

            btn2.click(
                reg_fn,
                inputs=[moving, fixed],
                outputs=[coronal, sagittal, axial]
            )

demo.launch(server_port=7860)