"""Minimal OpenAI-compatible chat server around the AwareVLN (VILA / Llama-3 8B) checkpoint.

Run inside the ``awarevln-eval`` conda env:
    CUDA_VISIBLE_DEVICES=3 python integrations/v3/serve_awarevln.py --ckpt .../ck/awarevln --port 8600

Accepts ``/v1/chat/completions`` with one system message and one user message whose content is a
list of ``text`` / ``image_url`` (data URL) parts, exactly what ``RemoteQwen3VL.build_messages`` sends.
Images are inserted as ``<image>`` tokens at their positions; generation is greedy and serialised.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

AWAREVLN_ROOT = os.environ.get("AWAREVLN_ROOT", "/data/pengyh/workspace/Reproductions/AwareVLN")
sys.path.insert(0, AWAREVLN_ROOT)

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=os.path.join(AWAREVLN_ROOT, "ck/awarevln"))
parser.add_argument("--served-name", default="awarevln")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8600)
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

tokenizer, model, image_processor, _ = load_pretrained_model(args.ckpt, os.path.basename(args.ckpt.rstrip("/")))
model.eval()
LOCK = threading.Lock()
app = FastAPI()


def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        url = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")


def _build(messages):
    system_text = None
    user_text, images = "", []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            system_text = content if isinstance(content, str) else " ".join(p.get("text", "") for p in content)
        elif role == "user":
            if isinstance(content, str):
                user_text += content
            else:
                for part in content:
                    if part.get("type") == "text":
                        user_text += part.get("text", "")
                    elif part.get("type") == "image_url":
                        images.append(_decode_image(part["image_url"]["url"]))
                        user_text += "<image>\n"
    conv = conv_templates["llama_3"].copy()
    if system_text:
        conv.system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_text
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt(), images


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": args.served_name, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(body: dict):
    conv, prompt, images = _build(body.get("messages", []))
    max_new = int(min(int(body.get("max_tokens") or args.max_new_tokens), args.max_new_tokens))
    started = time.perf_counter()
    with LOCK, torch.inference_mode():
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        kwargs = {}
        if images:
            kwargs["images"] = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        out = model.generate(
            input_ids,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    text = tokenizer.batch_decode(out, skip_special_tokens=False)[0].strip()
    for s in (stop_str, conv.sep2, "<|eot_id|>", "<|end_of_text|>"):
        if text.endswith(s):
            text = text[: -len(s)].strip()
    if text.startswith("<|begin_of_text|>"):
        text = text[len("<|begin_of_text|>"):].strip()
    completion_tokens = int(out.shape[1])
    finish = "length" if completion_tokens >= max_new else "stop"
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": int(input_ids.shape[1]) + 196 * len(images),
                "completion_tokens": completion_tokens,
                "total_tokens": int(input_ids.shape[1]) + completion_tokens,
                "latency_ms": (time.perf_counter() - started) * 1000,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
