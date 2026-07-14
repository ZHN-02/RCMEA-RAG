"""Shared helpers for cleaning generated pancreas report text."""

from __future__ import annotations

import json
import re
from typing import Any, Tuple


PANCREAS_PREFIX = "Pancreas:"
END_MARKER_RE = re.compile(r"(?is)</s>|<\|endoftext\|>|###")


def normalize_section(text: Any, prefix: str | None = PANCREAS_PREFIX) -> str:
    value = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()
    value = END_MARKER_RE.split(value)[0].strip()
    if not value:
        return ""
    if prefix is None:
        return value
    return value if value.lower().startswith(prefix.lower()) else f"{prefix} {value}"


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = str(text or "").find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : pos + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def extract_json_string_prefix(raw: str, field: str = "report_section") -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', raw)
    if not match:
        return ""
    pos = match.end()
    chars: list[str] = []
    escape = False
    while pos < len(raw):
        ch = raw[pos]
        if escape:
            if ch == "n":
                chars.append("\n")
            elif ch == "r":
                chars.append("\r")
            elif ch == "t":
                chars.append("\t")
            elif ch in {'"', "\\", "/"}:
                chars.append(ch)
            elif ch == "u" and pos + 4 < len(raw):
                hex_value = raw[pos + 1 : pos + 5]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hex_value):
                    try:
                        chars.append(chr(int(hex_value, 16)))
                        pos += 4
                    except ValueError:
                        chars.append("u")
                else:
                    chars.append("u")
            else:
                chars.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            break
        else:
            chars.append(ch)
        pos += 1
    return "".join(chars).strip()


def extract_report_section_source(row: dict[str, Any], parsed: dict[str, Any] | None = None) -> tuple[str, str]:
    parsed = parsed if isinstance(parsed, dict) else {}
    value = parsed.get("report_section")
    if isinstance(value, str) and value.strip():
        return value, "parsed_json"

    raw = str(row.get("raw_generation") or "")
    parsed_raw = extract_first_json_object(raw)
    if isinstance(parsed_raw, dict):
        value = parsed_raw.get("report_section")
        if isinstance(value, str) and value.strip():
            return value, "raw_json"

    prefix = extract_json_string_prefix(raw, "report_section")
    if prefix:
        return prefix, "raw_prefix"

    if raw.strip():
        return raw, "raw_text"
    return "", "empty"


def split_sentences(section: str, prefix: str | None = PANCREAS_PREFIX) -> list[str]:
    section = normalize_section(section, prefix=prefix)
    if not section:
        return []
    if prefix is not None and section.lower().startswith(prefix.lower()):
        body = section[len(prefix) :].strip()
    else:
        body = section.strip()
    if not body:
        return []
    parts = re.split(r"(?<=[.!?])\s+", body)
    sentences = [part.strip().strip('"') for part in parts if part.strip().strip('"')]
    return sentences


def normalize_for_dedupe(sent: str, remove_numbers: bool = False) -> str:
    text = sent.lower()
    text = re.sub(r"\([^)]*\)", "", text)
    if remove_numbers:
        text = re.sub(r"\b\d+(?:\.\d+)?\b", "#", text)
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return " ".join(text.split())


def finish_sentence(sentence: str) -> str:
    sent = " ".join(sentence.strip().strip('"').split())
    sent = sent.rstrip(" ,;:")
    if not sent:
        return ""
    if sent[-1] not in ".!?":
        sent += "."
    return sent


def is_truncated_tail_fragment(sentence: str) -> bool:
    sent = " ".join(str(sentence or "").strip().strip('"').split())
    if not sent:
        return False
    bare = sent.rstrip(".!?").strip()
    lower = bare.lower()
    words = re.findall(r"[a-zA-Z]+", bare)
    if not words:
        return False
    if bare[-1:] in {",", ";", ":"}:
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*x\s*$", lower):
        return True
    if re.fullmatch(
        r"(?:there\s+(?:is|are)\s+(?:a|an)\s+)?"
        r"(?:stable|unchanged|previously|measuring|measures?|up to)?\s*"
        r"\d+(?:\.\d+)?\s*(?:mm|cm)",
        lower,
    ):
        return True
    if words[-1].lower() in {
        "and",
        "or",
        "with",
        "without",
        "within",
        "from",
        "to",
        "of",
        "in",
        "at",
        "by",
        "for",
        "as",
        "than",
        "the",
        "a",
        "an",
        "measures",
        "measure",
        "measuring",
        "demonstrates",
        "extends",
        "courses",
        "abuts",
        "encases",
        "involves",
        "communicates",
    }:
        return True
    if len(words[-1]) <= 2 and words[-1].lower() not in {"no", "mm", "cm"}:
        return True
    if len(words) <= 4 and words[0].lower() in {"the", "this", "that", "these", "those"}:
        has_verb = any(word.lower() in {"is", "are", "was", "were", "has", "have", "had"} for word in words)
        if not has_verb:
            return True
    if lower in {"there is red", "there are red"}:
        return True
    return False


def dedupe_sentences(sentences: list[str], remove_numbers: bool = False) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sent in sentences:
        key = normalize_for_dedupe(sent, remove_numbers=remove_numbers)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sent)
    return out


def clean_report_section(
    section: str,
    remove_numbers: bool = False,
    prefix: str | None = PANCREAS_PREFIX,
    drop_truncated_tail: bool = True,
) -> str:
    text = normalize_section(section, prefix=prefix)
    if not text:
        return ""
    if prefix is not None and text.lower().startswith(prefix.lower()):
        body = text[len(prefix) :].strip()
    else:
        body = text.strip()
    sentences = split_sentences(body, prefix=None)
    if not sentences:
        cleaned = " ".join(body.split()).strip()
        return normalize_section(cleaned, prefix=prefix)
    deduped = dedupe_sentences(sentences, remove_numbers=remove_numbers)
    if drop_truncated_tail and len(deduped) > 1 and is_truncated_tail_fragment(deduped[-1]):
        deduped = deduped[:-1]
    cleaned_body = " ".join(finish_sentence(sent) for sent in deduped if finish_sentence(sent))
    if not cleaned_body:
        return ""
    return normalize_section(cleaned_body, prefix=prefix)
