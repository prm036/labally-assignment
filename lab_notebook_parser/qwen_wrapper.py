from typing import Optional
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVLExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", max_new_tokens: int = 1024):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {model_name}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded.")

    def ask_image(self, image_path: str, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, max_new_tokens=max_new_tokens)

    def ask_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._generate(messages, max_new_tokens=max_new_tokens)

    def _generate(self, messages, max_new_tokens: Optional[int] = None) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
        if image_inputs is not None:
            kwargs["images"] = image_inputs
        if video_inputs is not None:
            kwargs["videos"] = video_inputs

        inputs = self.processor(**kwargs).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
