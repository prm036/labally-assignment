import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/projects/e33188/praneeth/labally/lab_parser_outputs_v3/Example_lab_notebook_page_vlm_input_1600px.png"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
kwargs["images"] = image_inputs
inputs = processor(**kwargs).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    
print("Success!")
