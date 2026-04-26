import gradio as gr
from PIL import Image

from system1_demo_infer import System1DemoDetector
from explanation_template import build_explanation

detector = System1DemoDetector(device="cuda")

def predict(image):
    if image is None:
        return None, "请先上传一张图片。", {}

    result, overlay = detector.predict_pil(image)
    explanation = build_explanation(result)

    summary = (
        f"判断结果：{result['pred_label']}\n"
        f"fake_prob：{result['fake_prob']:.3f}\n"
        f"证据等级：{result['status']}"
    )

    return overlay, summary + "\n\n" + explanation, result


with gr.Blocks(title="可解释生成式图像检测 Demo") as demo:
    gr.Markdown("# 可解释生成式图像检测 Demo")
    gr.Markdown(
        "上传一张图片，系统将基于 DINOv3 patch-token 证据输出真假判断、可疑区域和解释文字。"
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="上传图片")
            btn = gr.Button("开始检测")
        with gr.Column():
            overlay_img = gr.Image(type="pil", label="证据区域可视化")
            text_out = gr.Textbox(label="检测结果与解释", lines=12)
            json_out = gr.JSON(label="结构化证据 JSON")

    btn.click(
        fn=predict,
        inputs=input_img,
        outputs=[overlay_img, text_out, json_out]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)