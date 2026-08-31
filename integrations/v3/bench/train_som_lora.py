"""LoRA SFT of Qwen3-VL-4B on the SoM direction-choice dataset.

Run in FreeAskAgent/.venv-train:
  python integrations/v3/bench/train_som_lora.py --data datasets/som_v1 --out .../models/lora-som-v1
"""
import argparse, glob, json, math, os, random, sys, time

import torch
from PIL import Image

AGENT = "/data/pengyh/workspace/FreeAskAgent"
sys.path.insert(0, AGENT)
SOM_PROMPT = None
for line in [None]:
    import re
    src = open(AGENT + "/agentflow/agents/models_embodied_v2/skiils/protocol.py").read()
    m = re.search(r'SOM_PROMPT = """(.*?)"""', src, re.S)
    SOM_PROMPT = m.group(1)

def load_rows(data_dirs, holdout_scenes=2):
    rows = []
    for data_dir in data_dirs.split(","):
        for f in sorted(glob.glob(os.path.join(data_dir, "shard_*.jsonl"))):
            for line in open(f):
                try:
                    row = json.loads(line)
                    row["image"] = os.path.join(data_dir, row["image"])
                    rows.append(row)
                except json.JSONDecodeError:
                    continue  # a shard may still be appending
    scenes = sorted({r["meta"]["scene"] for r in rows})
    eval_scenes = set(scenes[-holdout_scenes:])
    train = [r for r in rows if r["meta"]["scene"] not in eval_scenes]
    val = [r for r in rows if r["meta"]["scene"] in eval_scenes]
    return train, val

def build_inputs(processor, data_dir, row, *, with_target=True, device="cuda:0"):  # noqa: D401
    image = Image.open(row["image"] if os.path.isabs(row["image"]) or os.path.exists(row["image"]) else os.path.join(data_dir, row["image"])).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SOM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": row["prompt"]}, {"type": "image"}]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not with_target:
        return processor(text=[prompt_text], images=[image], return_tensors="pt").to(device)
    full_text = prompt_text + row["target"] + "<|im_end|>\n"
    inputs = processor(text=[full_text], images=[image], return_tensors="pt").to(device)
    prompt_len = processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"].shape[1]
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    inputs["labels"] = labels
    return inputs

@torch.no_grad()
def eval_accuracy(model, processor, data_dir, rows, n=60, log=print):
    model.eval()
    correct = 0
    sample = rows[:n]
    for row in sample:
        inputs = build_inputs(processor, data_dir, row, with_target=False)
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
        text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = json.loads(text[text.index("{"): text.rindex("}") + 1]) if "{" in text and "}" in text else {}
        if str(m.get("choice", "")).strip().upper() == row["label"].upper():
            correct += 1
    model.train()
    acc = correct / max(1, len(sample))
    log(f"eval acc {acc:.3f} on {len(sample)}")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/som_v1")
    ap.add_argument("--out", default=AGENT + "/models/lora-som-v1")
    ap.add_argument("--model", default=AGENT + "/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = "cuda:{}".format(int(os.environ.get("LOCAL_RANK", "0")))
    if world > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(device)
    train, val = load_rows(args.data)
    random.Random(42).shuffle(train)   # shards are scene-ordered; mix before slicing
    if args.max_rows:
        train = train[: args.max_rows]
    if rank == 0:
        print(f"train rows {len(train)}, val rows {len(val)}, world {world}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    ddp = None
    if world > 1:
        # find_unused_parameters=True + gradient checkpointing double-marks
        # parameters in the reducer ("Expected to mark a variable ready only
        # once", seen at step 301); every trainable parameter is used, so False.
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
            find_unused_parameters=False,
        )
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = int(len(train) * args.epochs) // (args.accum * world)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, steps_total))))
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train.log"), "a") if rank == 0 else None
    def log(*a):
        if rank == 0:
            print(*a, flush=True); print(*a, file=logf, flush=True)
    log(f"start: {len(train)} rows, {steps_total} optimizer steps, accum {args.accum}")
    random.seed(0)
    order = list(range(len(train)))
    step = micro = 0
    t0 = time.time()
    running = 0.0
    for epoch in range(math.ceil(args.epochs)):
        random.shuffle(order)
        for i in order[rank::world]:
            if step >= steps_total:
                break
            try:
                inputs = build_inputs(processor, args.data, train[i], device=device)
                net = ddp if ddp is not None else model
                loss = net(**inputs).loss / args.accum
                loss.backward()
                running += float(loss) * args.accum
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); optim.zero_grad(set_to_none=True); micro = 0
                log("OOM, batch skipped"); continue
            micro += 1
            if micro >= args.accum:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                micro = 0; step += 1
                if step % 20 == 0:
                    log(f"step {step}/{steps_total} loss {running/ (20*args.accum):.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"{(time.time()-t0)/max(1,step):.1f}s/step")
                    running = 0.0
                if step % args.eval_every == 0 and val:
                    # Both ranks pause here so the backward/allreduce lockstep
                    # is never broken by a rank-0-only evaluation.
                    if world > 1:
                        torch.distributed.barrier()
                    if rank == 0:
                        eval_accuracy(model, processor, args.data, val, log=log)
                        if step % args.save_every == 0:
                            model.save_pretrained(os.path.join(args.out, f"step_{step}"))
                    if world > 1:
                        torch.distributed.barrier()
    if rank == 0:
        model.save_pretrained(args.out)
        if val:
            eval_accuracy(model, processor, args.data, val, n=400, log=log)
    log("done")

if __name__ == "__main__":
    main()
