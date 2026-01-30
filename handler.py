import time
import re
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MarianTokenizer,
    MarianMTModel,
)

# =====================================================
# CUDA SAFETY (RTX 4090 + RTX 5090)
# =====================================================
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# GPU-aware CUDA optimizations (NO logic change)
# =====================================================
def enable_cuda_optimizations():
    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)

    log(f"GPU detected: {gpu_name} (SM {major}.{minor})")

    # RTX 4090 (Ada) and older
    if major <= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log("TF32 ENABLED (Ada/Ampere)")

    # RTX 5090 (Blackwell)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        log("TF32 DISABLED (Blackwell safe mode)")

# =====================================================
# Model paths
# =====================================================
SUMMARY_MODEL_PATH = "/models/hf/qwen"
TRANSLATE_MODEL_PATH = "/models/hf/marian-ru-en"

summary_tokenizer = None
summary_model = None
translate_tokenizer = None
translate_model = None

# =====================================================
# Default system prompt
# =====================================================
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional legal assistant.\n"
    "Produce a single-paragraph summary of the ENTIRE document in clear English.\n"
    "STRICT RULES:\n"
    "- Output MUST be one paragraph only\n"
    "- Do NOT use headings, titles, bullet points, or lists\n"
    "- Do NOT classify the document type unless explicitly stated in the text\n"
    "- Do NOT invent or infer information\n"
    "- Mention only facts that are explicitly present in the document\n"
    "- Cover all major sections evenly if the document is long\n"
    "- Focus on parties, purpose, key obligations, payments, terms, penalties, and dispute resolution if present\n"
    "- Ignore layout, tables, formatting, and section numbering\n"
    "- Write in neutral legal English\n\n"
)

# =====================================================
# Load SUMMARY model (Qwen 2.5 7B)
# =====================================================
def load_summary_model():
    global summary_tokenizer, summary_model
    if summary_model is not None:
        return

    log("Loading SUMMARY model (Qwen-2.5-7B-Instruct, FP16)")

    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    summary_model = AutoModelForCausalLM.from_pretrained(
        SUMMARY_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    summary_model.eval()
    enable_cuda_optimizations()

    log(f"SUMMARY model loaded on device: {summary_model.device}")

# =====================================================
# Load TRANSLATION model (Marian RU → EN)
# =====================================================
def load_translate_model():
    global translate_tokenizer, translate_model
    if translate_model is not None:
        return

    log("Loading TRANSLATION model (Marian RU → EN)")

    translate_tokenizer = MarianTokenizer.from_pretrained(
        TRANSLATE_MODEL_PATH,
        local_files_only=True
    )

    translate_model = MarianMTModel.from_pretrained(
        TRANSLATE_MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")

    translate_model.eval()
    enable_cuda_optimizations()

    log("TRANSLATION model loaded")

# =====================================================
# Detect layout separators
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

# =====================================================
# TRANSLATION - OPTIMIZED WITH BATCHING
# =====================================================
def translate_text(text: str) -> str:
    lines = text.split("\n")
    out_lines = []
    batch_lines = []
    batch_indices = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            out_lines.append(line)
            continue

        if re.match(r"^[\u2022•\-\*\u00B7]+$", stripped):
            out_lines.append(line)
            continue

        if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 2:
            out_lines.append(line)
            continue

        if re.match(r"^\|\s*[-\s_\.]+\|\s*[-\s_\.]+\|\s*$", line):
            out_lines.append(line)
            continue

        if is_layout_line(line):
            out_lines.append(line)
            continue

        if "|" in line:
            cells = line.split("|")
            new_cells = []

            for cell in cells:
                cell_text = cell.strip()

                if not cell_text or re.match(r"^[-\s_\.]+$", cell_text):
                    new_cells.append(cell)
                    continue

                if len(re.findall(r"[A-Za-zА-Яа-я]", cell_text)) < 2:
                    new_cells.append(cell)
                    continue

                inputs = translate_tokenizer(
                    cell_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(translate_model.device)

                with torch.no_grad():
                    output = translate_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )

                translated = translate_tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                new_cells.append(f" {translated} ")

            out_lines.append("|".join(new_cells))
            continue

        batch_lines.append(line)
        batch_indices.append(len(out_lines))
        out_lines.append(None)

    if batch_lines:
        chunk_size = 16
        for i in range(0, len(batch_lines), chunk_size):
            chunk = batch_lines[i:i+chunk_size]
            chunk_idx = batch_indices[i:i+chunk_size]

            inputs = translate_tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(translate_model.device)

            with torch.no_grad():
                outputs = translate_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            translated = translate_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            for j, trans in enumerate(translated):
                out_lines[chunk_idx[j]] = trans

    return "\n".join(out_lines)

# =====================================================
# OCR cleanup
# =====================================================
def clean_ocr_noise(text: str) -> str:
    cleaned = []
    seen = set()

    for raw in text.split("\n"):
        line = raw.strip()
        upper = line.upper()

        if not line:
            continue
        if is_layout_line(line):
            continue
        if len(re.findall(r"[A-Za-z]", line)) < 5:
            continue
        if upper in seen:
            continue

        seen.add(upper)
        cleaned.append(line)

    return "\n".join(cleaned)

# =====================================================
# Word limiter
# =====================================================
def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])

# =====================================================
# SUMMARY
# =====================================================
def summarize_all_pages(pages, max_words: int, system_prompt: str):
    full_text = "\n\n".join(
        cleaned
        for p in pages
        if (cleaned := clean_ocr_noise(p["text"]))
        and len(re.findall(r"[A-Za-z]", cleaned)) > 20
    )

    if not full_text.strip():
        log("ERROR: No valid text found for summary")
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_text}
    ]

    try:
        prompt = summary_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{full_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    inputs = summary_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(summary_model.device)

    with torch.no_grad():
        output = summary_model.generate(
            **inputs,
            max_new_tokens=max_words * 3,
            min_new_tokens=max(50, max_words // 2),
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=summary_tokenizer.pad_token_id,
            eos_token_id=summary_tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    decoded = summary_tokenizer.decode(new_tokens, skip_special_tokens=True)
    decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()

    return limit_words(decoded, max_words)

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    input_data = event["input"]
    pages = input_data["pages"]
    max_words = int(input_data.get("n_words", 100))
    system_prompt = input_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    load_translate_model()
    load_summary_model()

    for p in pages:
        p["text"] = translate_text(p["text"])

    summary = summarize_all_pages(pages, max_words, system_prompt)

    return {
        "summary": summary,
        "pages": pages
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
