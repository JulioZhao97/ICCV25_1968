import os
os.environ["GRADIO_TEMP_DIR"] = "./tmp"

import sys
import torch
import torchvision
import gradio as gr
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from visualization import visualize_bbox

# == select device ==
device = 'cuda' if torch.cuda.is_available() else 'cpu'

id_to_names = {
    0: 'title', 
    1: 'plain text',
    2: 'abandon', 
    3: 'figure', 
    4: 'figure_caption', 
    5: 'table', 
    6: 'table_caption', 
    7: 'table_footnote', 
    8: 'isolate_formula', 
    9: 'formula_caption'
}

def recognize_image(input_img, conf_threshold, iou_threshold):
    det_res = model.predict(
        input_img,
        imgsz=1024,
        conf=conf_threshold,
        device=device,
    )[0]
    boxes = det_res.__dict__['boxes'].xyxy
    classes = det_res.__dict__['boxes'].cls
    scores = det_res.__dict__['boxes'].conf
    indices = torchvision.ops.nms(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),iou_threshold=iou_threshold)
    boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
        scores = np.expand_dims(scores, 0)
        classes = np.expand_dims(classes, 0)
    vis_result = visualize_bbox(input_img, boxes, classes, scores, id_to_names)
    return vis_result
    
def gradio_reset():
    return gr.update(value=None), gr.update(value=None)

    
if __name__ == "__main__":
    root_path = os.path.abspath(os.getcwd())
    # == load model ==
    from doclayout_yolo import YOLOv10
    print(f"Using device: {device}")
    model = YOLOv10("./doclayout_yolo_docstructbench_imgsz1024.pt")  # load an official model
    
    with open("header.html", "r") as file:
        header = file.read()
    with gr.Blocks() as demo:
        gr.HTML(header)
        
        with gr.Row():
            with gr.Column():
                
                input_img = gr.Image(label=" ", interactive=True)
                with gr.Row():
                    clear = gr.Button(value="Clear")
                    predict = gr.Button(value="Detect", interactive=True, variant="primary")
                    
                with gr.Row():
                    conf_threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                    )
                    
                with gr.Row():
                    iou_threshold = gr.Slider(
                        label="NMS IOU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.45,
                    )
                    
                with gr.Accordion("Examples:"):
                    example_root = os.path.join(os.path.dirname(__file__), "assets", "app")
                    gr.Examples(
                        examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                    _.endswith("jpg")],
                        inputs=[input_img],
                    )
            with gr.Column():
                gr.Button(value="Predict Result:", interactive=False)
                output_img = gr.Image(label=" ", interactive=False)
    
        clear.click(gradio_reset, inputs=None, outputs=[input_img, output_img])
        predict.click(recognize_image, inputs=[input_img,conf_threshold,iou_threshold], outputs=[output_img])
    
    demo.launch(server_name="0.0.0.0", server_port=10001, debug=True, share=True)