#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Intent taxonomy
# ---------------------------
INTENTS = [
    "compliment",
    "ask_question",
    "answer_question",
    "add_information",
    "share_personal_experience",
    "agree",
    "disagree",
    "joke_or_playful",
    "empathy_or_support",
    "critique_or_suggestion",
    "request_source_or_proof",
    "meta_moderation_or_rules",
]

# ---------------------------
# Text helpers
# ---------------------------
WS_RE = re.compile(r"\s+")
EMOJI_MISC = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
REPEAT_CHARS = re.compile(r"(.)\1{2,}")  # heyyy -> heyy
FILLER = re.compile(r"\b(lol|lmao|lmfao|rofl|bro|bruh|ngl|fr|tf|wtf)\b", re.I)
PUNCT_SPAM = re.compile(r"[!?.,]{3,}")

def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())

def proto_reply(text: str, max_chars: int = 180) -> str:
    """
    Rudimentary-but-specific English: remove emojis, remove filler slang tokens,
    collapse repeated chars, reduce punctuation spam, trim length.
    """
    t = normalize_text(text)
    t = EMOJI_MISC.sub("", t)
    t = FILLER.sub("", t)
    t = REPEAT_CHARS.sub(r"\1\1", t)
    t = PUNCT_SPAM.sub(lambda m: m.group(0)[0], t)
    t = normalize_text(t)
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return t

def safe_json_load(s: str) -> Optional[dict]:
    """
    Attempt to parse JSON. If model wraps the JSON in extra text, extract
    the first {...} block.
    """
    s = (s or "").strip()
    if not s:
        return None
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None

def renormalize_distribution(dist: Any, topk: int) -> List[dict]:
    """
    Accepts any of:
      - [{"intent": "...", "prob": 0.5}, ...]
      - [{"intent": "...", "p": 0.5}, ...]
      - [{"label": "...", "prob": 0.5}, ...]
      - ["intent1", "intent2", ...]  -> uniform
      - ["intent1: 0.7", "intent2: 0.3", ...]
      - mixed lists (strings + dicts)

    Returns cleaned top-k [{"intent":..., "prob":...}] with probs summing to 1.
    """
    if not dist:
        return []

    cleaned: List[dict] = []

    # List of strings only
    if isinstance(dist, list) and all(isinstance(x, str) for x in dist):
        parsed: List[dict] = []
        for s in dist:
            s = s.strip()
            if not s:
                continue
            if ":" in s:
                left, right = s.split(":", 1)
                intent = left.strip()
                try:
                    prob = float(right.strip())
                except Exception:
                    prob = None
                if intent in INTENTS and prob is not None:
                    parsed.append({"intent": intent, "prob": prob})
            else:
                if s in INTENTS:
                    parsed.append({"intent": s, "prob": 1.0})
        if not parsed:
            return []
        cleaned = parsed

    # List of dict-like (or mixed)
    elif isinstance(dist, list):
        for item in dist:
            if isinstance(item, str):
                intent = item.strip()
                if intent in INTENTS:
                    cleaned.append({"intent": intent, "prob": 1.0})
                continue
            if not isinstance(item, dict):
                continue
            intent = item.get("intent") or item.get("label")
            prob = item.get("prob")
            if prob is None:
                prob = item.get("p")
            if intent in INTENTS and prob is not None:
                try:
                    cleaned.append({"intent": str(intent), "prob": float(prob)})
                except Exception:
                    pass
    else:
        return []

    cleaned = sorted(cleaned, key=lambda x: x["prob"], reverse=True)[:topk]
    s = sum(x["prob"] for x in cleaned)
    if s <= 0:
        return []
    for x in cleaned:
        x["prob"] = x["prob"] / s
    return cleaned

def iter_shard_jsonl(path: Path, shard_id: int, num_shards: int) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % num_shards != shard_id:
                continue
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# ---------------------------
# Qwen prompt
# ---------------------------
def build_messages(row: dict, topk: int) -> List[Dict[str, str]]:
    def sanitize_for_prompt(text: str) -> str:
        t = text or ""
        html_replacements = [
            ("&gt;", ">"),
            ("&lt;", "<"),
            ("&amp;", "&"),
            ("&quot;", '"'),
            ("&#39;", "'"),
            ("&nbsp;", " "),
            ("& gt ;", ">"),
            ("& lt ;", "<"),
            ("& amp ;", "&"),
            ("& quot ;", '"'),
            ("& #39 ;", "'"),
            ("& nbsp ;", " "),
        ]
        for src, dst in html_replacements:
            t = t.replace(src, dst)
        return normalize_text(t)

    post = row.get("post") or {}
    subreddit = sanitize_for_prompt(post.get("subreddit") or row.get("subreddit") or "")
    title = sanitize_for_prompt(post.get("title") or "")
    text = sanitize_for_prompt(post.get("text") or "")
    comment = sanitize_for_prompt(row.get("body") or row.get("comment") or "")

    user_obj = {
        "task": "Produce one JSON object in two steps: Step A intent labeling, Step B strict paraphrase.",
        "steps": {
            "step_a": (
                "Label intent of the GIVEN comment (not the post). Use the post context only when needed "
                "to disambiguate likely intent."
            ),
            "step_b": (
                "Write reply_proto as a strict paraphrase of the GIVEN comment only."
            ),
        },
        "grounding_mode": "comment_only",
        "intents": INTENTS,
        "topk": topk,
        "input": {
            "post_context_for_intent_only": {
                "subreddit": subreddit,
                "title": title,
                "text": text,
                "rule": "Never use post context as a source for reply_proto content.",
            },
            "comment": comment,
        },
        "sanitization": [
            "Before reasoning, decode common HTML escapes in inputs (&gt;, &lt;, &amp;, &quot;, &#39;, &nbsp;).",
            "Normalize excessive spaces to single spaces.",
            "If '>' quote markers are present, preserve them in reply_proto.",
        ],
        "critical_rules": [
            "Output only one valid JSON object with the exact schema keys and no extras.",
            "Grounding is comment_only: reply_proto may use only information in comment.",
            "If comment uses deixis (e.g., 'that', 'this', 'it'), keep the deixis; do not resolve references.",
            "Do not invent facts, entities, settings, subreddit lore, or external knowledge.",
            "Keep original stance and pragmatic force while neutralizing slang wording.",
        ],
        "copy_constraints": [
            "stance invariant: support/oppose/neutral/mixed must match the comment's stance toward its target.",
            "sentiment_polarity invariant: positive/negative/neutral polarity must be preserved.",
            "sarcasm_flag invariant: sarcasm true/false must reflect the original comment.",
            "speech_act invariant: question/request/assertion/command type must stay the same.",
        ],
        "banned_behaviors": [
            "Do not write a new reply to the post; paraphrase the given comment only.",
            "Do not add suggestions, advice, calls to action, or next steps not in the comment.",
            "Do not add examples, analogies, or explanations not in the comment.",
            "Do not add or change named entities, people, places, products, dates, or numbers.",
            "Do not add subreddit names unless already present in the comment text.",
            "Do not import facts from post text into reply_proto.",
            "Do not output HTML escapes like '&gt;' in reply_proto when a character form is available.",
            "Do not change a non-question into a question or vice versa.",
        ],
        "calibration": [
            "intent_topk must contain exactly topk entries.",
            "Each intent_topk item must be {\"intent\": <from intents>, \"prob\": <number in [0,1]>}.",
            "Sum of intent_topk probs must be 1.0 +/- 0.02.",
            "Use calibrated uncertainty: avoid overconfident spikes when intent is ambiguous.",
        ],
        "output_schema": {
            "type": "object",
            "keys_exactly": [
                "reply_proto",
                "intent_topk",
                "stance",
                "tone",
                "sarcasm",
                "rhetorical_question",
                "confidence",
                "violations",
                "notes",
            ],
            "field_rules": {
                "reply_proto": "string",
                "intent_topk": "array of {intent, prob}",
                "stance": ["support", "oppose", "neutral", "mixed"],
                "tone": ["hostile", "neutral", "friendly", "sarcastic", "sad", "angry", "amused", "anxious", "other"],
                "sarcasm": "bool",
                "rhetorical_question": "bool",
                "confidence": "number in [0,1]",
                "violations": "array of strings",
                "notes": "string",
            },
        },
        "self_check": [
            "Step A done: intent_topk length == topk and probs sum within 1.0 +/- 0.02.",
            "Step B done: reply_proto contains no information absent from comment.",
            "Deixis preserved where unresolved in comment.",
            "Final object has exactly the required keys and valid types.",
        ],
        "in_distribution_examples": [
            {
                "input_comment": "> that take is wild",
                "output_shape": {
                    "reply_proto": "> that opinion is extreme",
                    "sarcasm": False,
                    "violations": [],
                },
                "why": "Quoted marker '>' is preserved.",
            },
            {
                "input_comment": "yeah great idea, genius",
                "output_shape": {
                    "reply_proto": "Yes, that is definitely a great idea.",
                    "tone": "sarcastic",
                    "sarcasm": True,
                },
                "why": "Sarcasm remains sarcastic while wording is neutralized.",
            },
            {
                "input_comment": "r/AskNYC already covered this yesterday",
                "output_shape": {
                    "reply_proto": "r/AskNYC already discussed this yesterday.",
                    "violations": [],
                },
                "why": "Keeps existing subreddit mention; adds no new subreddit names.",
            },
        ],
    }

    return [
        {
            "role": "system",
            "content": (
                "You are a strict JSON generator.\n"
                "Return exactly one JSON object with keys: "
                "reply_proto, intent_topk, stance, tone, sarcasm, rhetorical_question, confidence, violations, notes.\n"
                "Do Step A then Step B in order:\n"
                "Step A: label intent_topk for the GIVEN comment only.\n"
                "Step B: write reply_proto as a strict paraphrase of the GIVEN comment only.\n"
                "grounding_mode is comment_only: post context is for intent disambiguation only and NEVER for paraphrase.\n"
                "No markdown, no prose outside JSON, no extra keys."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(user_obj, ensure_ascii=False),
        },
    ]

def format_sft(row: dict, reply_proto_text: str, intent_dist: List[dict]) -> Dict[str, str]:
    post = row.get("post") or {}
    subreddit = normalize_text(post.get("subreddit") or row.get("subreddit") or "")
    title = normalize_text(post.get("title") or "")
    text = normalize_text(post.get("text") or "")

    instruction = (
        "Predict plausible reply intents for a social media post and write a simple, specific prototype reply.\n"
        "The prototype reply should be plain English (no slang), grounded in the post.\n"
        "Output an ordered list of intents with probabilities (most likely first).\n\n"
        f"Subreddit: {subreddit}\n"
        f"Title: {title}\n"
        f"Text: {text}\n"
    )

    lines = [f"ReplyProto: {reply_proto_text}", "Intents:"]
    for item in intent_dist:
        lines.append(f"- {item['intent']}: {item['prob']:.3f}")

    return {"instruction": instruction, "output": "\n".join(lines)}

# ---------------------------
# HF chat wrapper
# ---------------------------
def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

def generate_one(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    prompt = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--qwen-min-conf", type=float, default=0.55)
    ap.add_argument("--max-rows", type=int, default=-1)
    args = ap.parse_args()

    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        print(f"[info] HF_HOME={hf_home}")

    print(f"[load] model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    kept = 0
    dropped = 0
    total = 0

    with args.out_path.open("w", encoding="utf-8") as out_f:
        for row in iter_shard_jsonl(args.in_path, args.shard_id, args.num_shards):
            total += 1
            if args.max_rows > 0 and total > args.max_rows:
                break

            msgs = build_messages(row, args.topk)

            data = None
            for attempt in range(3):
                text = generate_one(model, tokenizer, msgs, args.max_new_tokens)
                data = safe_json_load(text)
                if data is not None:
                    break
                # Retry with a JSON repair message
                msgs = [
                    msgs[0],
                    {"role": "user", "content": "Your previous output was invalid JSON. Output ONLY valid JSON for the same task."},
                ]

            if data is None:
                dropped += 1
                continue

            # Extract + normalize fields
            reply = proto_reply(str(data.get("reply_proto", "")))
            dist = renormalize_distribution(data.get("intent_distribution", []), args.topk)
            try:
                conf = float(data.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0

            if conf < args.qwen_min_conf or not reply or not dist:
                dropped += 1
                continue

            out_row = dict(row)
            out_row["label_source"] = "qwen"
            out_row["sarcasm"] = bool(data.get("sarcasm", False))
            out_row["rhetorical_question"] = bool(data.get("rhetorical_question", False))
            out_row["reply_proto"] = reply
            out_row["intent_distribution"] = dist
            out_row["intent_scored"] = [{"intent": x["intent"], "score": x["prob"]} for x in dist]
            out_row["sft"] = format_sft(out_row, reply, dist)

            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[done] shard {args.shard_id}/{args.num_shards} total={total} kept={kept} dropped={dropped}")

if __name__ == "__main__":
    main()
