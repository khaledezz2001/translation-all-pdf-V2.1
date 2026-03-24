import time
import re
import runpod
from vllm import LLM, SamplingParams

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

MODEL_PATH = "/models/hf/qwen"
llm_engine = None
tokenizer = None

# =====================================================
# System prompts (unchanged)
# =====================================================
DEFAULT_SUMMARY_PROMPT = (
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

def build_translate_prompt(target_language: str) -> str:
    prompt = (
        f"You are a certified professional legal translator.\n"
        f"Auto-detect the language of the input text and translate it into {target_language}.\n"
        f"STRICT RULES:\n"
        f"- Translate ONLY — do NOT summarize, paraphrase, or add commentary\n"
        f"- Preserve the original meaning, tone, and structure as closely as possible\n"
        f"- Keep proper nouns, names, dates, and numbers unchanged\n"
        f"- Preserve paragraph breaks and line structure\n"
        f"- If a word or phrase is already in {target_language}, keep it as-is\n"
        f"- Output ONLY the {target_language} translation, nothing else\n"
        f"- Do NOT include any notes, explanations, or metadata about the translation\n"
        f"- Do NOT mix languages: every word in the output MUST be in {target_language} "
        f"(except proper nouns, names, and abbreviations)\n"
        f"- Do NOT use words from other variants or related languages "
        f"(e.g. if {target_language} is Spanish, do NOT use Catalan, Portuguese, or Italian words)\n"
        f"CONSISTENCY RULES (CRITICAL):\n"
        f"- Every transliterated name MUST be spelled EXACTLY the same way every time it appears\n"
        f"- Preserve ALL letters in transliterated names — do NOT drop, swap, or shorten syllables\n"
        f"- Example: if 'Пожитков' → 'Pozhitkov', it must ALWAYS be 'Pozhitkov' (NEVER 'Pogotikov', 'Pozhikov', etc.)\n"
        f"- Example: if 'Курина' → 'Kurina', it must ALWAYS be 'Kurina' (NEVER 'Kurna')\n"
        f"- Russian street names: preserve the full genitive form. Example: 'ул. Герасима Курина' → 'calle Gerasima Kurina' (NOT 'calle Gerasim Kurna')\n"
        f"- Do NOT repeat or duplicate content blocks — translate each section exactly once\n"
    )

    prompt += (
        "CYRILLIC TRANSLITERATION RULES (apply ONLY if source text contains Cyrillic script):\n"
        "- Use phonetic transliteration that matches the Cyrillic spelling letter-by-letter\n"
        "- Complete mapping: А→A, Б→B, В→V, Г→G, Д→D, Е→E, Ё→Yo, Ж→Zh, З→Z, "
        "И→I, Й→Y, К→K, Л→L, М→M, Н→N, О→O, П→P, Р→R, С→S, Т→T, "
        "У→U, Ф→F, Х→Kh, Ц→Ts, Ч→Ch, Ш→Sh, Щ→Shch, Ъ→(omit), Ы→Y, Ь→(omit), Э→E, Ю→Yu, Я→Ya\n"
        "- 'Кс' in Russian names → 'Ks'. Example: Ксенофонтов → Ksenofontov (NOT Xenofontov)\n"
        "- BUT if a Cyrillic name is a phonetic rendering of a known foreign name, restore the original spelling. "
        "Example: КСАВЬЕР → XAVIER (NOT KSAVIER)\n"
        "- 'Е' → always 'E' (NEVER 'I'). Example: ГРИСЕН → GRISEN (NOT GRISIN)\n"
        "- 'Ж' → always 'Zh' (NEVER skip it). Example: Пожитков → Pozhitkov (NEVER Pogotikov)\n"
        "- Company names in Cyrillic are phonetic transcriptions — transliterate them back faithfully\n"
        "- For foreign place names, streets, and districts written phonetically in Cyrillic, "
        "ALWAYS restore the official English name — do NOT transliterate.\n"
        "HONG KONG: Сёнвань → Sheung Wan, Коулун → Kowloon, Цим Ша Цуй → Tsim Sha Tsui, "
        "Бонэм Стрэнд → Bonham Strand, Ванчай → Wan Chai, Монгкок → Mong Kok, "
        "Централ → Central, Абердин → Aberdeen, Чайвань → Chai Wan, Куорри Бей → Quarry Bay\n"
        "UAE: Дубай → Dubai, Абу-Даби → Abu Dhabi, Шарджа → Sharjah, "
        "Джебел Али → Jebel Ali, Дейра → Deira, Бур Дубай → Bur Dubai, "
        "Аджман → Ajman, Рас-эль-Хайма → Ras Al Khaimah, Фуджейра → Fujairah\n"
        "UK: Лондон → London, Вестминстер → Westminster, Кэнэри Уорф → Canary Wharf, "
        "Эдинбург → Edinburgh, Манчестер → Manchester, Бирмингем → Birmingham\n"
        "CYPRUS: Никосия → Nicosia, Лимассол → Limassol, Ларнака → Larnaca, Пафос → Paphos\n"
        "SINGAPORE: Сингапур → Singapore, Раффлз Плейс → Raffles Place\n"
        "BVI: Тортола → Tortola, Род Таун → Road Town\n"
        "SEYCHELLES: Маэ → Mahe, Виктория → Victoria, Праслин → Praslin\n"
        "OTHER: Панама → Panama, Белиз → Belize, Гибралтар → Gibraltar, "
        "Лихтенштейн → Liechtenstein, Люксембург → Luxembourg, Мальта → Malta, "
        "Каймановы острова → Cayman Islands, Бермуды → Bermuda\n"
        "COMMON TERMS: Стрит/Стрэнд → Street/Strand, Билдинг → Building, "
        "Башня/Тауэр → Tower, Авеню → Avenue, Плаза → Plaza, Роуд → Road\n"
    )

    prompt += (
        "RUSSIAN ABBREVIATIONS AND INSTITUTIONS:\n"
        "- ОВД (Отдел Внутренних Дел) → Departamento de Policía / Police Department (NOT 'Oficina de Investigación de Delitos')\n"
        "- ЗАГС → Registro Civil / Civil Registry\n"
        "- ИНН → NIF (Número de Identificación Fiscal) / TIN (Tax Identification Number)\n"
        "- ОГРН → Número de Registro Estatal / State Registration Number\n"
    )

    if target_language.lower() in ("spanish", "español", "espanol"):
        prompt += (
            "SPANISH LEGAL TERMINOLOGY (MANDATORY — use these exact terms):\n"
            "PARTIES IN LEASE/RENTAL AGREEMENTS (CRITICAL — be consistent throughout):\n"
            "- Tenant → Arrendatario (NEVER 'Inquilino' — use 'Arrendatario' EVERYWHERE in the document)\n"
            "- Landlord (singular) → Arrendador\n"
            "- Landlords (plural) → Arrendadores\n"
            "- CRITICAL: If the document uses plural 'Landlords', ALWAYS use 'los Arrendadores' (NEVER 'el Arrendador')\n"
            "- CRITICAL: Pick ONE term for each party and use it CONSISTENTLY throughout the ENTIRE document. "
            "Do NOT alternate between 'Inquilino' and 'Arrendatario' — ALWAYS use 'Arrendatario'.\n"
            "PARTIES IN LOAN AGREEMENTS:\n"
            "- Lender / Займодавец → Prestamista (NEVER 'Cedente', NEVER 'Acreedor', NEVER 'Creditor')\n"
            "- Borrower / Заемщик → Prestatario (NEVER 'Deudor')\n"
            "- Creditor / Кредитор → Acreedor\n"
            "- Debtor / Должник → Deudor\n"
            "- Цедент → Cedente (ONLY in cession/assignment agreements)\n"
            "- Цессионарий → Cesionario (ONLY in cession/assignment agreements)\n"
            "OTHER LEGAL PARTIES:\n"
            "- party (legal) → parte (NEVER 'partido')\n"
            "- parties → partes (NEVER 'partidos')\n"
            "- trespasser → ocupante ilegal (NEVER 'intruso')\n"
            "- witnesses → testigos\n"
            "CONTRACT STRUCTURE TERMS:\n"
            "- Schedule (contract appendix) → Anexo (NEVER 'Programa')\n"
            "- Schedule A, Schedule B → Anexo A, Anexo B\n"
            "- Clause → Cláusula\n"
            "- Exhibit → Exhibición / Anexo\n"
            "- Addendum → Adenda\n"
            "- Amendment → Enmienda\n"
            "COMPANY TYPES:\n"
            "- ОАО (Открытое Акционерное Общество) → Sociedad Anónima (S.A.) — NEVER 'Societat Anónima'\n"
            "- ЗАО (Закрытое Акционерное Общество) → Sociedad Anónima Cerrada\n"
            "- ООО (Общество с Ограниченной Ответственностью) → Sociedad de Responsabilidad Limitada (S.R.L.) — NEVER 'Sociedad con Limitación'\n"
            "- Limited / Ltd → Limitada / Ltda.\n"
            "- АО (Акционерное Общество) → Sociedad Anónima (S.A.)\n"
            "- ИП (Индивидуальный Предприниматель) → Empresario Individual\n"
            "- Международная Акционерная Компания → Compañía Internacional Sociedad Anónima — NEVER 'Compañía Internacional de Acciones'\n"
            "- УК (Управляющая Компания) → Sociedad Gestora / Compañía Gestora\n"
            "REAL ESTATE AND LEASE TERMS:\n"
            "- lease → contrato de arrendamiento\n"
            "- rent → renta / alquiler\n"
            "- premises → local / instalaciones\n"
            "- nuisance → molestias / actividades molestas (NEVER leave as 'nuisance' in English)\n"
            "- shareholding / equity stake → participación accionaria / porcentaje de acciones\n"
            "- remedies → recursos legales / acciones legales (NEVER 'remedios')\n"
            "- written notice → notificación escrita\n"
            "- 'three (3) months notice' → 'con tres (3) meses de antelación' (NEVER 'tres meses antes')\n"
            "- 'it is hereby agreed as follows' → 'EN CONSECUENCIA, LAS PARTES ACUERDAN LO SIGUIENTE'\n"
            "- act of God → fuerza mayor (preferred) or acto de fuerza mayor\n"
            "ARCHITECTURAL AND BUILDING TERMS:\n"
            "- basement / underground floor → sótano (NEVER 'planta baja subterránea')\n"
            "- mezzanine / mezzanine floor → entresuelo (NEVER 'plaza media')\n"
            "- ground floor → planta baja\n"
            "- floor plan → plano de planta\n"
            "FINANCIAL AND LEGAL TERMS:\n"
            "- расчеты / settlements → pagos / liquidaciones (NEVER 'cálculo')\n"
            "- Договор займа → Contrato de Préstamo\n"
            "- Договор цессии → Contrato de Cesión\n"
            "- Устав → Estatutos Sociales\n"
            "- Доверенность → Poder Notarial\n"
            "- Протокол → Acta\n"
            "- Решение → Resolución / Decisión\n"
            "- по решению / по усмотрению → por decisión de (NEVER 'a discreción')\n"
            "- месторождения → yacimientos de recursos naturales\n"
            "- новые области природных ресурсов → nuevas áreas de recursos naturales (NOT just 'yacimientos')\n"
            "- прошито и пронумеровано → cosido y numerado (NOT 'pegado')\n"
            "- Наблюдательный совет / Технический комитет → Comité Técnico (NOT 'Comité de Supervisión' unless context is a supervisory board)\n"
            "GENERAL RULES FOR SPANISH:\n"
            "- Use standard Castilian Spanish (castellano) — NEVER Catalan, Galician, or other variants\n"
            "- Use formal legal register: use 'deberá' for obligations, 'por la presente' for declarations\n"
            "- Use standard Spanish date format: '__ de octubre de 2025'\n"
            "- Maintain formal legal phrasing: 'representado por su director', 'actuando en virtud de'\n"
            "- ALL English words MUST be translated — do NOT leave any English terms in the output "
            "(except proper nouns, company names, and internationally recognized abbreviations)\n"
            "- If Greek text appears (e.g. architectural labels), add Spanish translation in brackets: e.g. 'ΚΑΤΟΨΗ ΥΠΟΓΕΙΟΥ [PLANO DEL SÓTANO]'\n"
        )

    prompt += (
        "LEGAL TRANSLATION STANDARDS:\n"
        f"- Use formal legal register in {target_language}\n"
        "- Preserve civil law terminology accurately\n"
        "- Maintain formal legal phrasing and tone\n"
        f"- Use standard date format for {target_language}\n"
        "- Do NOT leave any terms in English unless they are proper nouns or internationally recognized abbreviations\n"
    )

    return prompt


# =====================================================
# Load model with vLLM — auto-detects GPU
# =====================================================
def load_model():
    global llm_engine, tokenizer
    if llm_engine is not None:
        return

    # IMPORTANT: Do NOT call torch.cuda.* before vLLM init!
    # It initializes CUDA which forces 'spawn' multiprocessing and crashes.
    log("Loading model with vLLM engine...")
    t0 = time.time()

    llm_engine = LLM(
        model=MODEL_PATH,
        dtype="auto",                    # auto-selects BF16 on Ampere+
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        trust_remote_code=True,
        enable_prefix_caching=True,       # Caches system prompt KV across pages
    )

    tokenizer = llm_engine.get_tokenizer()

    # Log GPU info AFTER vLLM has initialized CUDA
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    log(f"vLLM engine ready in {time.time()-t0:.1f}s")


# =====================================================
# Helper: build prompt from messages
# =====================================================
def build_prompt(messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n/nothink\n")
        return "\n".join(parts)


# =====================================================
# Layout / OCR helpers
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

def clean_ocr_noise(text: str) -> str:
    cleaned, seen = [], set()
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or is_layout_line(line):
            continue
        if len(re.findall(r"[A-Za-z]", line)) < 5:
            continue
        upper = line.upper()
        if upper in seen:
            continue
        seen.add(upper)
        cleaned.append(line)
    return "\n".join(cleaned)

def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    if truncated.rstrip().endswith("."):
        return truncated.rstrip()
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    last_excl = truncated.rfind("! ")
    last_quest = truncated.rfind("? ")
    best = max(last_period, last_excl, last_quest)
    if best > len(truncated) * 0.6:
        return truncated[:best + 1].strip()
    return truncated.rstrip()

def clean_output(decoded: str) -> str:
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
    decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()
    for marker in ["STRICT RULES:", "LEGAL TRANSLATION STANDARDS:"]:
        if marker in decoded:
            idx_m = decoded.find(marker)
            lines = decoded[idx_m:].split("\n")
            last_rule = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("- "):
                    last_rule = i
            decoded = "\n".join(lines[last_rule + 1:]).strip()
    return decoded


# =====================================================
# TRANSLATION — vLLM parallel batch
# =====================================================
def translate_text_batch(texts, target_language="English"):
    translate_prompt = build_translate_prompt(target_language)

    prompts = []
    valid_indices = []
    results = [""] * len(texts)

    for idx, text in enumerate(texts):
        stripped = (text or "").strip()
        if not stripped or len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 5:
            results[idx] = text or ""
            continue

        messages = [
            {"role": "system", "content": translate_prompt},
            {"role": "user", "content": stripped}
        ]
        prompts.append(build_prompt(messages))
        valid_indices.append(idx)

    if not prompts:
        return results

    log(f"Translating {len(prompts)} pages in parallel with vLLM...")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
    )

    t0 = time.time()
    outputs = llm_engine.generate(prompts, sampling_params)
    gen_time = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    log(f"Translation: {total_tokens} tokens in {gen_time:.1f}s "
        f"({total_tokens/gen_time:.1f} tok/s effective)")

    for i, output in enumerate(outputs):
        results[valid_indices[i]] = clean_output(output.outputs[0].text)

    return results


# =====================================================
# SUMMARY — vLLM single call
# =====================================================
def summarize_all_pages(pages, max_words, system_prompt):
    full_text = "\n\n".join(
        cleaned for p in pages
        if (cleaned := clean_ocr_noise(p["text"]))
        and len(re.findall(r"[A-Za-z]", cleaned)) > 20
    )

    if not full_text.strip():
        log("ERROR: No valid text found for summary")
        return ""

    doc_word_count = len(full_text.split())
    actual_target = max(50, min(max_words, doc_word_count // 3))
    log(f"Summary target: {actual_target} words (doc has {doc_word_count} words)")

    user_content = (
        f"Summarize the following document in approximately {actual_target} words. "
        f"Make sure to complete all sentences properly.\n\n"
        f"DOCUMENT:\n{full_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    prompt = build_prompt(messages)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=min(actual_target * 5, 4096),
    )

    t0 = time.time()
    outputs = llm_engine.generate([prompt], sampling_params)
    gen_time = time.time() - t0

    decoded = clean_output(outputs[0].outputs[0].text)
    result = limit_words(decoded, actual_target)

    log(f"Summary: {len(result.split())} words in {gen_time:.1f}s")
    return result


# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    input_data = event["input"]
    pages = input_data["pages"]
    max_words = int(input_data.get("n_words", 500))
    system_prompt = input_data.get("system_prompt", DEFAULT_SUMMARY_PROMPT)
    target_language = input_data.get("target_language", "English")

    log(f"Processing {len(pages)} pages, target: {max_words} words, translate to: {target_language}")

    load_model()

    # 1) Translate all pages in parallel
    log(f"Starting batch translation to {target_language}...")
    start = time.time()
    page_texts = [p["text"] for p in pages]
    translated_texts = translate_text_batch(page_texts, target_language)
    for i, p in enumerate(pages):
        p["text"] = translated_texts[i]
    log(f"Translation done in {time.time()-start:.2f}s")

    # 2) Summarize
    log(f"Creating summary ({max_words} words)")
    start = time.time()
    summary = summarize_all_pages(pages, max_words, system_prompt)
    log(f"Summary done in {time.time()-start:.2f}s")

    if not summary:
        log("WARNING: Summary is empty!")

    log("Handler finished")
    return {"summary": summary, "pages": pages}

runpod.serverless.start({"handler": handler})
