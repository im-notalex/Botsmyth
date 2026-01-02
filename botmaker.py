#!/usr/bin/env python3

import ast

import base64

import io

import json

import mimetypes

import os

import random

import re

import uuid

import zipfile

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

from urllib.parse import urlparse



import requests

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_from_directory, url_for

from werkzeug.utils import secure_filename



try:

    from PIL import Image, PngImagePlugin



    HAS_PIL = True

except Exception:

    HAS_PIL = False



APP_TITLE = "Botsmyth"

MAX_IMAGES = 6



BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

IMAGES_DIR = DATA_DIR / "images"

EXPORTS_DIR = DATA_DIR / "exports"

DB_PATH = DATA_DIR / "bots.json"

SETTINGS_PATH = DATA_DIR / "settings.json"



PROVIDERS = [

    {"id": "openai", "label": "OpenAI"},

    {"id": "gemini", "label": "Gemini"},

    {"id": "anthropic", "label": "Anthropic"},

    {"id": "grok", "label": "Grok"},

    {"id": "openrouter", "label": "OpenRouter"},

    {"id": "openai_compatible", "label": "OpenAI Compatible"},

]



PROVIDER_DEFAULTS = {

    "openai": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},

    "gemini": {"model": "gemini-1.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta"},

    "anthropic": {"model": "claude-3-5-sonnet-latest", "base_url": "https://api.anthropic.com/v1"},

    "grok": {"model": "grok-2-latest", "base_url": "https://api.x.ai/v1"},

    "openrouter": {"model": "openai/gpt-4o-mini", "base_url": "https://openrouter.ai/api/v1"},

    "openai_compatible": {"model": "your-model", "base_url": "http://localhost:1234/v1"},

}



DEFAULT_SETTINGS = {

    "provider": "openai",

    "api_key": "",

    "model": PROVIDER_DEFAULTS["openai"]["model"],

    "base_url": PROVIDER_DEFAULTS["openai"]["base_url"],

    "temperature": 0.7,

    "max_tokens": 1200,

    "use_images": True,

    "creator_name": "",

    "autosave_seconds": 30,

}



TOGGLE_OPTIONS = [

    "Smut",

    "Fluff",

    "Romance",

    "Angst",

    "Comfort",

    "Adventure",

    "Comedy",

    "Horror",

    "Mystery",

    "Action",

    "Slice of Life",

    "Slow Burn",

    "Enemies to Lovers",

    "Friends to Lovers",

    "Dom",

    "Sub",

    "Teasing",

    "Wholesome",

    "Dark",

    "Cute",

]



COMMON_TAGS = [

    "SFW",

    "Suggestive",

    "NSFW",

    "Extreme",

    "Romance",

    "Fluff",

    "Angst",

    "Slice of Life",

    "Fantasy",

    "Sci-Fi",

    "Modern",

    "Historical",

    "Adventure",

    "Mystery",

    "Horror",

    "Comedy",

    "Action",

    "Drama",

    "Slow Burn",

    "Enemies to Lovers",

    "Friends to Lovers",

    "Wholesome",

    "Dark",

    "Cute",

    "Male",

    "Female",

    "Nonbinary",

]



DEFAULT_FIELDS = {

    "name": "",

    "alias": "",

    "description": "",

    "appearance": "",

    "personality": "",

    "age": "",

    "species": "",

    "gender": "",

    "pronouns": "",

    "height": "",

    "body_type": "",

    "outfit": "",

    "distinguishing_features": "",

    "voice": "",

    "speech_style": "",

    "mannerisms": "",

    "catchphrases": "",

    "occupation": "",

    "setting": "",

    "relationship": "",

    "backstory": "",

    "current_scenario": "",

    "goals": "",

    "motivations": "",

    "values": "",

    "likes": "",

    "dislikes": "",

    "skills": "",

    "weaknesses": "",

    "flaws": "",

    "fears": "",

    "secrets": "",

    "user_role": "",

    "bot_role": "",

    "response_length": "Medium",

    "pov": "Second person",

    "narration_style": "Mixed",

    "emoji_use": "None",

    "formatting": "",

    "style_rules": "",

    "consent_rules": "",

    "boundaries": "",

    "limits": "",

    "kinks": "",

    "author_notes": "",

    "memory_notes": "",

    "simple_input": "",

    "simple_name": "",

    "simple_age": "",

    "simple_species": "",

    "simple_current_first_messages": "",

    "simple_current_scenarios": "",

    "simple_current_dialogues": "",

    "min_tokens_description": "auto",

    "min_tokens_first_messages": "auto",

    "min_tokens_scenario": "auto",

    "min_tokens_dialogues": "auto",

    "system_prompt": "",

    "post_history_instructions": "",

    "rules": "",

    "world_lore": "",

    "creator": "",

    "character_version": "",

    "primary_first_message": "",

    "language": "English",

    "rating": "Unrated",

}



GEN_SECTIONS = [

    {

        "id": "core",

        "label": "Core Identity",

        "hint": "Name, description, personality, age, species, relationships.",

        "fields": [

            "name",

            "alias",

            "description",

            "personality",

            "age",

            "species",

            "gender",

            "pronouns",

            "occupation",

            "relationship",

        ],

    },

    {

        "id": "appearance",

        "label": "Appearance",

        "hint": "Body, outfit, and visible traits.",

        "fields": [

            "appearance",

            "distinguishing_features",

            "height",

            "body_type",

            "outfit",

        ],

    },

    {

        "id": "voice",

        "label": "Voice and Style",

        "hint": "Voice, speech, and formatting.",

        "fields": [

            "voice",

            "speech_style",

            "mannerisms",

            "catchphrases",

            "formatting",

            "style_rules",

        ],

    },

    {

        "id": "world",

        "label": "World and Motivation",

        "hint": "Setting, backstory, and goals.",

        "fields": [

            "setting",

            "current_scenario",

            "backstory",

            "goals",

            "motivations",

            "values",

            "world_lore",

        ],

    },

    {

        "id": "traits",

        "label": "Traits and Limits",

        "hint": "Likes, flaws, and boundaries.",

        "fields": [

            "likes",

            "dislikes",

            "skills",

            "weaknesses",

            "flaws",

            "fears",

            "secrets",

            "consent_rules",

            "boundaries",

            "limits",

            "kinks",

        ],

    },

    {

        "id": "roleplay",

        "label": "Roleplay Controls",

        "hint": "POV, response length, and narration style.",

        "fields": [

            "user_role",

            "bot_role",

            "response_length",

            "pov",

            "narration_style",

            "emoji_use",

        ],

    },

    {

        "id": "guidance",

        "label": "System Guidance",

        "hint": "System prompts, rules, and post-history instructions.",

        "fields": [

            "system_prompt",

            "post_history_instructions",

            "rules",

        ],

    },

    {

        "id": "metadata",

        "label": "Metadata",

        "hint": "Language, rating, and notes.",

        "fields": [

            "creator",

            "character_version",

            "language",

            "rating",

            "author_notes",

            "memory_notes",

        ],

    },

]



DEFAULT_GEN_SECTIONS = {section["id"]: True for section in GEN_SECTIONS}

DEFAULT_GEN_SECTIONS["guidance"] = False

DEFAULT_GEN_SECTIONS["metadata"] = False



APPENDABLE_FIELDS = {

    "description",

    "appearance",

    "personality",

    "backstory",

    "current_scenario",

    "goals",

    "motivations",

    "values",

    "likes",

    "dislikes",

    "skills",

    "weaknesses",

    "flaws",

    "fears",

    "secrets",

    "formatting",

    "style_rules",

    "consent_rules",

    "boundaries",

    "limits",

    "kinks",

    "system_prompt",

    "post_history_instructions",

    "rules",

    "world_lore",

    "author_notes",

    "memory_notes",

}



NON_GENERATABLE_FIELDS = {

    "response_length",

    "pov",

    "narration_style",

    "emoji_use",

    "rating",

    "language",

    "creator",

    "character_version",

    "primary_first_message",

    "simple_input",

    "simple_name",

    "simple_age",

    "simple_species",

    "simple_current_first_messages",

    "simple_current_scenarios",

    "simple_current_dialogues",

    "min_tokens_description",

    "min_tokens_first_messages",

    "min_tokens_scenario",

    "min_tokens_dialogues",

}



GENERATABLE_FIELDS = {key for key in DEFAULT_FIELDS if key not in NON_GENERATABLE_FIELDS}



app = Flask(__name__)





def now_iso() -> str:

    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")





def increment_version(value: str) -> str:

    text = str(value or "").strip()

    if not text:

        return "1"

    match = re.search(r"(\\d+)(?!.*\\d)", text)

    if not match:

        return f"{text} 2"

    number = int(match.group(1)) + 1

    return f"{text[:match.start()]}{number}{text[match.end():]}"





def ensure_dirs() -> None:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)





def load_json(path: Path, default: Any) -> Any:

    if not path.exists():

        return default

    try:

        with path.open("r", encoding="utf-8") as f:

            return json.load(f)

    except Exception:

        return default





def save_json(path: Path, data: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)





def load_db() -> Dict[str, Any]:

    db = load_json(DB_PATH, {"bots": []})

    if not isinstance(db, dict) or "bots" not in db or not isinstance(db["bots"], list):

        return {"bots": []}

    return db





def save_db(db: Dict[str, Any]) -> None:

    save_json(DB_PATH, db)





def default_gen_sections() -> Dict[str, bool]:

    return DEFAULT_GEN_SECTIONS.copy()





def normalize_gen_sections(raw: Any) -> Dict[str, bool]:

    data = default_gen_sections()

    if isinstance(raw, dict):

        for key in data:

            if key in raw:

                value = raw[key]

                if isinstance(value, bool):

                    data[key] = value

                elif isinstance(value, str):

                    data[key] = value.strip().lower() in {"1", "true", "yes", "on"}

                else:

                    data[key] = bool(value)

    return data





def enabled_fields_from_sections(sections: Dict[str, bool]) -> List[str]:

    ordered = []

    for section in GEN_SECTIONS:

        if sections.get(section["id"], True):

            ordered.extend(section["fields"])

    seen = set()

    out: List[str] = []

    for field in ordered:

        if field not in seen:

            seen.add(field)

            out.append(field)

    return out





def load_settings() -> Dict[str, Any]:

    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS.copy())

    merged = DEFAULT_SETTINGS.copy()

    if isinstance(settings, dict):

        merged.update({k: v for k, v in settings.items() if v is not None})

    merged["use_images"] = bool(merged.get("use_images", True))

    try:

        merged["temperature"] = float(merged.get("temperature", DEFAULT_SETTINGS["temperature"]))

    except Exception:

        merged["temperature"] = DEFAULT_SETTINGS["temperature"]

    try:

        merged["max_tokens"] = int(merged.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))

    except Exception:

        merged["max_tokens"] = DEFAULT_SETTINGS["max_tokens"]

    try:

        merged["autosave_seconds"] = int(merged.get("autosave_seconds", DEFAULT_SETTINGS["autosave_seconds"]))

    except Exception:

        merged["autosave_seconds"] = DEFAULT_SETTINGS["autosave_seconds"]

    provider = merged.get("provider", "openai")

    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])

    if not merged.get("model"):

        merged["model"] = defaults["model"]

    if not merged.get("base_url"):

        merged["base_url"] = defaults["base_url"]

    return merged





def save_settings(settings: Dict[str, Any]) -> None:

    save_json(SETTINGS_PATH, settings)





def ensure_bot_defaults(bot: Dict[str, Any]) -> Dict[str, Any]:

    bot.setdefault("fields", {})

    for key, value in DEFAULT_FIELDS.items():

        bot["fields"].setdefault(key, value)

    bot["gen_sections"] = normalize_gen_sections(bot.get("gen_sections"))

    for key in (

        "toggles",

        "tags",

        "images",

        "first_messages",

        "scenarios",

        "example_dialogues",

        "prompt_pairs",

        "memory",

        "lorebook",

    ):

        if not isinstance(bot.get(key), list):

            bot[key] = []

    return bot





def new_bot() -> Dict[str, Any]:

    bot_id = uuid.uuid4().hex[:10]

    return {

        "id": bot_id,

        "created_at": now_iso(),

        "updated_at": now_iso(),

        "fields": DEFAULT_FIELDS.copy(),

        "gen_sections": default_gen_sections(),

        "toggles": [],

        "tags": [],

        "images": [],

        "first_messages": [],

        "scenarios": [],

        "example_dialogues": [],

        "prompt_pairs": [],

        "memory": [],

        "lorebook": [],

    }





def get_bot(db: Dict[str, Any], bot_id: str) -> Optional[Dict[str, Any]]:

    for bot in db["bots"]:

        if bot.get("id") == bot_id:

            return bot

    return None





def upsert_bot(db: Dict[str, Any], bot: Dict[str, Any]) -> None:

    for idx, existing in enumerate(db["bots"]):

        if existing.get("id") == bot.get("id"):

            db["bots"][idx] = bot

            return

    db["bots"].append(bot)





def delete_bot(db: Dict[str, Any], bot_id: str) -> None:

    db["bots"] = [b for b in db["bots"] if b.get("id") != bot_id]





def duplicate_bot(db: Dict[str, Any], bot_id: str) -> Optional[Dict[str, Any]]:

    bot = get_bot(db, bot_id)

    if not bot:

        return None

    copied = json.loads(json.dumps(bot))

    copied["id"] = uuid.uuid4().hex[:10]

    copied["created_at"] = now_iso()

    copied["updated_at"] = now_iso()

    copied["fields"]["name"] = f"{copied['fields'].get('name', '')} Copy".strip()

    copied["fields"]["character_version"] = increment_version(copied["fields"].get("character_version", ""))

    db["bots"].append(copied)

    return copied





def normalize_json_text(text: str) -> str:

    return (

        text.replace("\u201c", '"')

        .replace("\u201d", '"')

        .replace("\u2018", "'")

        .replace("\u2019", "'")

    )





def strip_trailing_commas(text: str) -> str:

    return re.sub(r",\s*([}\]])", r"\1", text)





def parse_json_loose(text: str) -> Optional[Any]:

    for variant in (text, normalize_json_text(text)):

        cleaned = strip_trailing_commas(variant)

        try:

            return json.loads(cleaned)

        except Exception:

            pass

        try:

            return ast.literal_eval(cleaned)

        except Exception:

            pass

    return None





def find_matching_json(text: str, start: int) -> Optional[int]:

    stack: List[str] = []

    in_string = False

    escape = False

    quote_char = ""

    for i in range(start, len(text)):

        ch = text[i]

        if in_string:

            if escape:

                escape = False

                continue

            if ch == "\\":

                escape = True

                continue

            if ch == quote_char:

                in_string = False

            continue

        if ch in "\"'":

            in_string = True

            quote_char = ch

            continue

        if ch in "{[":

            stack.append(ch)

            continue

        if ch in "}]":

            if not stack:

                return None

            opener = stack.pop()

            if opener == "{" and ch != "}":

                return None

            if opener == "[" and ch != "]":

                return None

            if not stack:

                return i

    return None





def find_json_candidate(text: str) -> Optional[str]:

    for i, ch in enumerate(text):

        if ch in "{[":

            end = find_matching_json(text, i)

            if end is not None:

                return text[i : end + 1]

    return None





def extract_json(text: str) -> Any:

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(r"^```[a-zA-Z0-9]*", "", text).strip()

        if text.endswith("```"):

            text = text[: -len("```")].strip()

    parsed = parse_json_loose(text)

    if parsed is not None:

        return parsed

    candidate = find_json_candidate(text)

    if candidate:

        parsed = parse_json_loose(candidate)

        if parsed is not None:

            return parsed

    return {}





def normalize_str_list(items: Any) -> List[str]:

    if not isinstance(items, list):

        return []

    out = []

    for item in items:

        if isinstance(item, str):

            cleaned = item.strip()

            if cleaned:

                out.append(cleaned)

    return out





def normalize_dialogues(items: Any) -> List[Dict[str, str]]:

    if not isinstance(items, list):

        return []

    out = []

    for item in items:

        if not isinstance(item, dict):

            continue

        user = str(item.get("user", "")).strip()

        bot = str(item.get("bot", "")).strip()

        if user or bot:

            out.append({"user": user, "bot": bot})

    return out





def normalize_lorebook(items: Any) -> List[Dict[str, Any]]:

    if not isinstance(items, list):

        return []

    out = []

    for item in items:

        if not isinstance(item, dict):

            continue

        key = str(item.get("key", "")).strip()

        content = str(item.get("content", "")).strip()

        enabled = item.get("enabled", True)

        if key or content:

            out.append({"key": key, "content": content, "enabled": bool(enabled)})

    return out





def apply_field_merge(existing: str, new_value: str, mode: str, field_key: str) -> str:

    existing = (existing or "").strip()

    new_value = (new_value or "").strip()

    if not new_value:

        return existing

    if mode == "overwrite":

        return new_value

    if mode == "append" and field_key in APPENDABLE_FIELDS:

        if existing:

            return f"{existing}\n\n{new_value}"

        return new_value

    if not existing:

        return new_value

    return existing





def clamp_sentences(text: str, max_sentences: int = 3) -> str:

    text = (text or "").strip()

    if not text:

        return ""

    parts = re.split(r"(?<=[.!?])\s+", text)

    if len(parts) <= max_sentences:

        return text

    return " ".join(parts[:max_sentences]).strip()





def parse_bool(value: Any) -> bool:

    if isinstance(value, bool):

        return value

    if isinstance(value, (int, float)):

        return value != 0

    if isinstance(value, str):

        return value.strip().lower() in {"1", "true", "yes", "on"}

    return False





EMOJI_RE = re.compile(

    "["

    "\U0001F1E0-\U0001F1FF"

    "\U0001F300-\U0001FAFF"

    "\u2600-\u26FF"

    "\u2700-\u27BF"

    "\uFE0F"

    "\u200D"

    "]+",

    flags=re.UNICODE,

)





def strip_emojis(text: str) -> str:

    return EMOJI_RE.sub("", text or "")





def sanitize_text(text: str, allow_emojis: bool) -> str:

    cleaned = (text or "").strip()

    if allow_emojis:

        return cleaned

    return strip_emojis(cleaned).strip()





def sanitize_list(items: List[str], allow_emojis: bool) -> List[str]:

    out = []

    for item in items:

        cleaned = sanitize_text(item, allow_emojis)

        if cleaned:

            out.append(cleaned)

    return out





def sanitize_dialogues(items: List[Dict[str, str]], allow_emojis: bool) -> List[Dict[str, str]]:

    out = []

    for item in items:

        user = sanitize_text(item.get("user", ""), allow_emojis)

        bot = sanitize_text(item.get("bot", ""), allow_emojis)

        if user or bot:

            out.append({"user": user, "bot": bot})

    return out





DESCRIPTION_SECTION_LABELS = [

    "Name",

    "Age",

    "Species",

    "Appearance",

    "Personality",

    "Relationships",

    "Abilities",

    "Extra",

]





def normalize_description_sections(text: str) -> str:

    cleaned = (text or "").strip()

    if not cleaned:

        return ""

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    if "\\n" in cleaned and "\n" not in cleaned:

        cleaned = cleaned.replace("\\n", "\n")

    pattern = re.compile(r"(?i)\\b(" + "|".join(map(re.escape, DESCRIPTION_SECTION_LABELS)) + r"):")

    matches = list(pattern.finditer(cleaned))

    if not matches:

        return cleaned

    sections: List[str] = []

    prefix = cleaned[:matches[0].start()].strip()

    if prefix:

        if not re.match(r"(?i)^extra:", prefix):

            prefix = f"Extra: {prefix}"

        sections.append(prefix)

    for idx, match in enumerate(matches):

        start = match.start()

        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)

        chunk = cleaned[start:end].strip()

        if chunk:

            sections.append(chunk)

    normalized = "\n\n".join(sections)

    normalized = re.sub(r"[ \t]+\n", "\n", normalized)

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()





def sync_simple_identity_fields(bot: Dict[str, Any]) -> None:

    fields = bot.get("fields", {})

    pairs = (

        ("simple_name", "name"),

        ("simple_age", "age"),

        ("simple_species", "species"),

    )

    for simple_key, main_key in pairs:

        simple_val = str(fields.get(simple_key, "") or "").strip()

        main_val = str(fields.get(main_key, "") or "").strip()

        if simple_val and not main_val:

            fields[main_key] = simple_val

        elif main_val and not simple_val:

            fields[simple_key] = main_val




def apply_payload(bot: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:

    ensure_bot_defaults(bot)

    fields = payload.get("fields", {}) if isinstance(payload, dict) else {}

    if isinstance(fields, dict):

        for key in DEFAULT_FIELDS:

            if key in fields:

                bot["fields"][key] = fields[key]

    gen_sections = payload.get("gen_sections")

    if isinstance(gen_sections, dict):

        bot["gen_sections"] = normalize_gen_sections(gen_sections)

    toggles = payload.get("toggles", [])

    if isinstance(toggles, list):

        bot["toggles"] = [str(t).strip() for t in toggles if str(t).strip()]

    tags = payload.get("tags", [])

    if isinstance(tags, list):

        bot["tags"] = [str(t).strip() for t in tags if str(t).strip()]

    bot["first_messages"] = normalize_str_list(payload.get("first_messages", []))

    bot["scenarios"] = normalize_str_list(payload.get("scenarios", []))

    bot["memory"] = normalize_str_list(payload.get("memory", []))

    bot["lorebook"] = normalize_lorebook(payload.get("lorebook", []))

    bot["example_dialogues"] = normalize_dialogues(payload.get("example_dialogues", []))

    bot["prompt_pairs"] = normalize_dialogues(payload.get("prompt_pairs", []))

    sync_simple_identity_fields(bot)

    return bot





def load_image_data(bot: Dict[str, Any]) -> List[Dict[str, str]]:

    images = []

    for filename in bot.get("images", [])[:MAX_IMAGES]:

        path = IMAGES_DIR / filename

        if not path.exists():

            continue

        mime = mimetypes.guess_type(str(path))[0] or "image/png"

        with path.open("rb") as f:

            data = base64.b64encode(f.read()).decode("ascii")

        images.append({"mime": mime, "data": data})

    return images





def build_context(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    lines = []

    def add(label: str, key: str) -> None:

        value = str(f.get(key, "")).strip()

        if value:

            lines.append(f"{label}: {value}")



    add("Name", "name")

    add("Alias", "alias")

    add("Description", "description")

    add("Original input", "simple_input")

    add("Appearance", "appearance")

    add("Personality", "personality")

    add("Age", "age")

    add("Species", "species")

    add("Gender", "gender")

    add("Pronouns", "pronouns")

    add("Height", "height")

    add("Body type", "body_type")

    add("Outfit", "outfit")

    add("Distinguishing features", "distinguishing_features")

    add("Voice", "voice")

    add("Speech style", "speech_style")

    add("Mannerisms", "mannerisms")

    add("Catchphrases", "catchphrases")

    add("Occupation", "occupation")

    add("Setting", "setting")

    add("Relationship to user", "relationship")

    add("Backstory", "backstory")

    add("Current scenario", "current_scenario")

    add("Goals", "goals")

    add("Motivations", "motivations")

    add("Values", "values")

    add("Likes", "likes")

    add("Dislikes", "dislikes")

    add("Skills", "skills")

    add("Weaknesses", "weaknesses")

    add("Flaws", "flaws")

    add("Fears", "fears")

    add("Secrets", "secrets")

    add("System prompt", "system_prompt")

    add("Post history instructions", "post_history_instructions")

    add("Rules", "rules")

    add("User role", "user_role")

    add("Bot role", "bot_role")

    add("Formatting", "formatting")

    add("Style rules", "style_rules")

    add("Consent rules", "consent_rules")

    add("Boundaries", "boundaries")

    add("Limits", "limits")

    add("Kinks", "kinks")

    add("Author notes", "author_notes")

    add("Memory notes", "memory_notes")

    add("World lore", "world_lore")

    add("Creator", "creator")

    add("Character version", "character_version")

    add("Language", "language")

    add("Rating", "rating")

    if bot.get("toggles"):

        lines.append(f"Toggles: {', '.join(bot['toggles'])}")

    if bot.get("tags"):

        lines.append(f"Tags: {', '.join(bot['tags'])}")

    if bot.get("memory"):

        lines.append(f"Memory anchors: {' | '.join(bot['memory'])}")

    if bot.get("first_messages"):

        lines.append(f"Existing first messages: {len(bot['first_messages'])}")

    if bot.get("scenarios"):

        lines.append(f"Existing scenarios: {len(bot['scenarios'])}")

    if bot.get("example_dialogues"):

        lines.append(f"Example dialogues: {len(bot['example_dialogues'])}")

    return "\n".join(lines) if lines else "None"





def build_profile_prompt(bot: Dict[str, Any], notes: str, allowed_fields: List[str]) -> str:

    context = build_context(bot)

    notes = notes.strip() if notes else "None"

    keys = ", ".join(allowed_fields)

    min_line = ""

    if "description" in allowed_fields:

        min_tokens = get_min_tokens(bot, "min_tokens_description")

        if min_tokens:

            min_line = f"Description should be at least ~{min_tokens} tokens.\\n"

    return (

        "You are a character profile generator for RP bots.\n"

        "Use the details below and any attached images.\n"

        "Return JSON only with these keys:\n"

        f"{keys}\n"

        "Only include keys from the list. Use empty strings if unknown.\n\n"

        f"{min_line}"

        f"Known details:\n{context}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only."

    )





def build_field_prompt(bot: Dict[str, Any], field_key: str, notes: str) -> str:

    context = build_context(bot)

    notes = notes.strip() if notes else "None"

    label = field_key.replace("_", " ").title()

    min_line = ""

    if field_key == "description":

        min_tokens = get_min_tokens(bot, "min_tokens_description")

        if min_tokens:

            min_line = f"Description should be at least ~{min_tokens} tokens.\\n"

    return (

        "You are updating a single field in an RP bot profile.\n"

        f"Field: {label}\n"

        "Return JSON only in the form:\n"

        f"{{\"{field_key}\": \"...\"}}\n\n"

        f"{min_line}"

        f"Known details:\n{context}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only."

    )





def get_reference_first_message(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    primary = str(f.get("primary_first_message", "")).strip()

    if primary:

        return primary

    for msg in bot.get("first_messages", []):

        cleaned = str(msg).strip()

        if cleaned:

            return cleaned

    return ""





TOKEN_PRESET_MAP = {

    "low": 250,

    "medium": 500,

    "high": 750,

    "very_high": 1000,

    "very high": 1000,

    "extreme": 1500,

}




def parse_min_tokens(value: Any) -> Optional[int]:

    if value is None:

        return None

    if isinstance(value, (int, float)):

        num = int(value)

    else:

        text = str(value).strip().lower()

        if not text or text in {"auto", "default", "none"}:

            return None

        if text in TOKEN_PRESET_MAP:

            return TOKEN_PRESET_MAP[text]

        text = text.replace("_", " ")

        if text in TOKEN_PRESET_MAP:

            return TOKEN_PRESET_MAP[text]

        try:

            num = int(float(text))

        except Exception:

            return None

    if num <= 0:

        return None

    return min(num, 10000)




def get_min_tokens(bot: Dict[str, Any], field_key: str) -> Optional[int]:

    fields = bot.get("fields", {})

    return parse_min_tokens(fields.get(field_key))




def build_first_messages_prompt(bot: Dict[str, Any], count: int, notes: str) -> str:

    context = build_context(bot)

    notes = notes.strip() if notes else "None"

    reference = get_reference_first_message(bot)

    min_line = ""

    min_tokens = get_min_tokens(bot, "min_tokens_first_messages")

    if min_tokens:

        min_line = (

            f"Aim for at least ~{min_tokens} tokens total across all first messages while preserving style.\n"

        )

    reference_block = ""

    if reference:

        reference_block = f"Reference first message (style only, do not copy):\n{reference}\n\n"

    return (

        f"Write {count} distinct first messages for this RP bot.\n"

        "Each message should feel in-character and match the style settings.\n"

        "Write high-quality, clean prose with careful detail.\n"

        "If a reference first message is provided, match its tone, POV, formatting, and length closely.\n"

        f"{min_line}"

        "If you reference names, use {{char}} for the bot and {{user}} for the user.\n"

        "Return JSON only: {\"first_messages\": [\"...\"]}.\n\n"

        f"{reference_block}"

        f"Known details:\n{context}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only."

    )





def build_scenarios_prompt(bot: Dict[str, Any], count: int, notes: str) -> str:

    context = build_context(bot)

    notes = notes.strip() if notes else "None"

    min_line = ""

    min_tokens = get_min_tokens(bot, "min_tokens_scenario")

    if min_tokens:

        min_line = (

            f"Aim for at least ~{min_tokens} tokens total across all scenarios. "

            "If that conflicts with the sentence limit, maximize detail while keeping 1 to 3 sentences.\n"

        )

    return (

        f"Create {count} short scenario hooks for this RP bot.\n"

        "Each scenario should be 1 to 3 sentences.\n"

        f"{min_line}"

        "If you reference names, use {{char}} and {{user}}.\n"

        "Return JSON only: {\"scenarios\": [\"...\"]}.\n\n"

        f"Known details:\n{context}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only."

    )





def build_dialogues_prompt(bot: Dict[str, Any], count: int, notes: str) -> str:

    context = build_context(bot)

    notes = notes.strip() if notes else "None"

    min_line = ""

    min_tokens = get_min_tokens(bot, "min_tokens_dialogues")

    if min_tokens:

        min_line = f"Aim for at least ~{min_tokens} tokens total across all dialogue pairs.\n"

    return (

        f"Create {count} example dialogue pairs between user and bot.\n"

        f"{min_line}"

        "If you reference names, use {{user}} and {{char}} instead of proper names.\n"

        "Return JSON only: {\"dialogues\": [{\"user\": \"...\", \"bot\": \"...\"}]}.\n\n"

        f"Known details:\n{context}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only."

    )





def build_simple_context(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    lines = []

    name = str(f.get("simple_name") or f.get("name") or "").strip()

    age = str(f.get("simple_age") or f.get("age") or "").strip()

    species = str(f.get("simple_species") or f.get("species") or "").strip()

    personality = str(f.get("personality") or "").strip()

    if name:

        lines.append(f"Name: {name}")

    if age:

        lines.append(f"Age: {age}")

    if species:

        lines.append(f"Species: {species}")

    if personality:

        lines.append(f"Personality: {personality}")

    current_first = str(f.get("simple_current_first_messages") or "").strip()

    if not current_first:

        current_first = "\n".join(bot.get("first_messages", [])[:3])

    if current_first:

        lines.append(f"Current first messages:\n{current_first}")

    current_scenarios = str(f.get("simple_current_scenarios") or "").strip()

    if not current_scenarios:

        current_scenarios = "\n".join(bot.get("scenarios", [])[:3])

    if current_scenarios:

        lines.append(f"Current scenarios:\n{current_scenarios}")

    current_dialogues = str(f.get("simple_current_dialogues") or "").strip()

    if not current_dialogues:

        snippets = []

        for pair in bot.get("example_dialogues", [])[:3]:

            user = str(pair.get("user", "")).strip()

            bot_line = str(pair.get("bot", "")).strip()

            if user:

                snippets.append(f"{{{{user}}}}: {user}")

            if bot_line:

                snippets.append(f"{{{{char}}}}: {bot_line}")

        current_dialogues = "\n".join(snippets)

    if current_dialogues:

        lines.append(f"Current example dialogues:\n{current_dialogues}")

    return "\n\n".join(lines).strip()





def build_simple_prompt(bot: Dict[str, Any], simple_input: str, dialogue_count: int, notes: str) -> str:

    simple_input = simple_input.strip() if simple_input else "None"

    notes = notes.strip() if notes else "None"

    context = build_simple_context(bot)

    context_block = f"Existing bot details:\n{context}\n\n" if context else ""

    token_lines = []

    min_description = get_min_tokens(bot, "min_tokens_description")

    if min_description:

        token_lines.append(f"- Description: at least ~{min_description} tokens.")

    min_first = get_min_tokens(bot, "min_tokens_first_messages")

    if min_first:

        token_lines.append(f"- First message: at least ~{min_first} tokens.")

    min_scenario = get_min_tokens(bot, "min_tokens_scenario")

    if min_scenario:

        token_lines.append(f"- Scenario: at least ~{min_scenario} tokens.")

    min_dialogues = get_min_tokens(bot, "min_tokens_dialogues")

    if min_dialogues:

        token_lines.append(f"- Dialogues: at least ~{min_dialogues} tokens total.")

    token_block = ""

    if token_lines:

        token_block = "Minimum token targets (total per section):\n" + "\n".join(token_lines) + "\n\n"

    return (

        "You are creating a minimal RP bot starter kit.\n"

        "Use the user input and notes below plus any attached images.\n"

        "Return JSON only with keys:\n"

        "description, personality, scenario, first_message, dialogues.\n"

        "Prioritize the description; it should hold most of the detail and tokens.\n"

        "Scenario must be 1 to 2 sentences.\n"

        "If a minimum token target is set, aim for it while keeping the sentence limit.\n"

        f"{token_block}"

        "If you reference names, use {{user}} and {{char}}.\n"

        "If existing messages, scenarios, or dialogues are provided, create new ones and avoid repeats.\n"

        "If existing first messages are provided, match their tone and formatting while keeping content new.\n"

        f"dialogues should include {dialogue_count} pairs of {{\"user\": \"...\", \"bot\": \"...\"}}.\n\n"

        f"{context_block}"

        f"User input:\n{simple_input}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only."

    )





def build_compile_prompt(bot: Dict[str, Any], notes: str) -> str:

    context = build_context(bot)

    notes = notes.strip() if notes else "None"

    reference = get_reference_first_message(bot)

    reference_block = ""

    if reference:

        reference_block = f"Reference first message (style only, do not copy):\n{reference}\n\n"

    token_lines = []

    min_description = get_min_tokens(bot, "min_tokens_description")

    if min_description:

        token_lines.append(f"- Description: at least ~{min_description} tokens.")

    min_first = get_min_tokens(bot, "min_tokens_first_messages")

    if min_first:

        token_lines.append(f"- First messages: at least ~{min_first} tokens total across both messages.")

    min_scenario = get_min_tokens(bot, "min_tokens_scenario")

    if min_scenario:

        token_lines.append(f"- Scenario: at least ~{min_scenario} tokens total.")

    min_dialogues = get_min_tokens(bot, "min_tokens_dialogues")

    if min_dialogues:

        token_lines.append(f"- Dialogues: at least ~{min_dialogues} tokens total across all pairs.")

    token_block = ""

    total_line = "Aim for roughly 1200 to 2400 tokens total across all fields.\n"

    if token_lines:

        token_block = "Minimum token targets (total per section):\n" + "\n".join(token_lines) + "\n"

        total_line = "Meet the minimum targets while keeping the description dominant.\n"

    description_line = (

        "- Description: 24 to 36 paragraphed labeled sections; aim for 80 to 90 percent of the tokens"

    )

    if not min_description:

        description_line += ", and a minimum of 800+ tokens"

    description_line += ".\n"

    first_line = "- First messages: 2 distinct openers"

    if not min_first:

        first_line += ", each 40 to 80 tokens"

    first_line += ".\n"

    return (

        "Create a polished RP bot pack with clean, high-quality writing and meticulous detail.\n"

        "Use the known details below and any attached images.\n"

        "Generate all four categories at once and return JSON only with keys:\n"

        "description, first_messages, scenario, dialogues.\n"

        "Keep the description dominant; it must carry the vast majority of facts, lore, and characterization.\n"

        "Other sections must be concise and should not introduce new facts beyond the description.\n"

        "Stay tightly focused on the core concept; avoid unrelated tangents or extra characters.\n"

        "Format the description as labeled sections instead of a single paragraph.\n"

        "Use plain-text labels like:\n"

        "Name:, Age:, Species:, Appearance:, Personality:, Relationships:, Abilities:, Extra:\n"

        "Each label should be followed by 1 to 2 sentences.\n"

        "Put each label on its own line, separated by a blank line.\n"

        "Follow these structure targets:\n"

        f"{description_line}"

        f"{first_line}"

        "- Scenario: 1 to 2 sentences, concise hook only, no extra worldbuilding.\n"

        "- Dialogues: 4 pairs of {\"user\": \"...\", \"bot\": \"...\"}, each bot reply 1 to 2 sentences.\n"

        "If a minimum token target conflicts with the sentence limit, keep the sentence limit.\n"

        f"{token_block}"

        f"{total_line}"

        "If a reference first message is provided, match its tone, POV, formatting, and length closely.\n"

        "If you reference names, use {{user}} and {{char}}.\n"

        "Keep it clean: no meta commentary, no markdown, no bullet points in output strings.\n\n"

        f"{reference_block}"

        f"Known details:\n{context}\n\n"

        f"User notes:\n{notes}\n\n"

        "Output JSON only. Start with '{' and end with '}'."

    )





def build_chat_system(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    toggles = ", ".join(bot.get("toggles", [])) or "None"

    parts = []

    if f.get("system_prompt"):

        parts.append(str(f.get("system_prompt")))

    parts.append("You are roleplaying as the bot described below. Stay in character.")

    parts.append(f"Name: {f.get('name', '')}")

    parts.append(f"Description: {f.get('description', '')}")

    parts.append(f"Appearance: {f.get('appearance', '')}")

    parts.append(f"Personality: {f.get('personality', '')}")

    parts.append(f"Voice: {f.get('voice', '')}")

    parts.append(f"Speech style: {f.get('speech_style', '')}")

    parts.append(f"World lore: {f.get('world_lore', '')}")

    parts.append(f"Rules: {f.get('rules', '')}")

    parts.append(f"Formatting: {f.get('formatting', '')}")

    parts.append(f"Style rules: {f.get('style_rules', '')}")

    parts.append(f"Consent rules: {f.get('consent_rules', '')}")

    parts.append(f"Boundaries: {f.get('boundaries', '')}")

    parts.append(f"Limits: {f.get('limits', '')}")

    parts.append(f"Post history instructions: {f.get('post_history_instructions', '')}")

    parts.append(f"Toggles: {toggles}")

    parts.append(f"Response length: {f.get('response_length', 'Medium')}")

    parts.append(f"POV: {f.get('pov', 'Second person')}")

    parts.append(f"Narration style: {f.get('narration_style', 'Mixed')}")

    parts.append(f"Emoji use: {f.get('emoji_use', 'None')}")

    return "\n".join(parts)





def build_chat_system_assistant(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    parts = [

        "You are a helpful assistant for testing this bot and its assets.",

        "Answer clearly and directly. Do not roleplay or speak in-character.",

        "If the user asks for in-character responses, ask them to switch to In Character mode.",

        f"Bot name: {f.get('name', '')}",

        f"Description: {f.get('description', '')}",

        f"Personality: {f.get('personality', '')}",

        f"Scenario: {f.get('current_scenario', '')}",

        f"Voice: {f.get('voice', '')}",

        f"Speech style: {f.get('speech_style', '')}",

        f"World lore: {f.get('world_lore', '')}",

    ]

    return "\n".join(parts)





def dedupe_list(items: List[str]) -> List[str]:

    seen = set()

    out = []

    for item in items:

        if item and item not in seen:

            seen.add(item)

            out.append(item)

    return out





def split_greetings(bot: Dict[str, Any]) -> Tuple[str, List[str]]:

    f = bot.get("fields", {})

    primary = str(f.get("primary_first_message", "")).strip()

    greetings = [str(m).strip() for m in bot.get("first_messages", []) if str(m).strip()]

    greetings = dedupe_list(greetings)

    if not primary and greetings:

        primary = greetings[0]

    alternates = [g for g in greetings if g != primary]

    return primary, alternates





def build_mes_example(bot: Dict[str, Any]) -> str:

    lines = []

    for pair in bot.get("example_dialogues", []):

        user = str(pair.get("user", "")).strip()

        bot_line = str(pair.get("bot", "")).strip()

        if user:

            lines.append(f"{{{{user}}}}: {user}")

        if bot_line:

            lines.append(f"{{{{char}}}}: {bot_line}")

    return "\n".join(lines)





def build_export_description(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    parts = []

    if f.get("description"):

        parts.append(f["description"])

    basics = []

    if f.get("age"):

        basics.append(f"Age: {f['age']}")

    if f.get("species"):

        basics.append(f"Species: {f['species']}")

    if f.get("gender"):

        basics.append(f"Gender: {f['gender']}")

    if f.get("pronouns"):

        basics.append(f"Pronouns: {f['pronouns']}")

    if f.get("occupation"):

        basics.append(f"Occupation: {f['occupation']}")

    if basics:

        parts.append(", ".join(basics))

    if f.get("appearance"):

        parts.append(f"Appearance: {f['appearance']}")

    if f.get("distinguishing_features"):

        parts.append(f"Distinguishing features: {f['distinguishing_features']}")

    if f.get("backstory"):

        parts.append(f"Backstory: {f['backstory']}")

    return "\n\n".join(parts).strip()





def build_export_personality(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    parts = []

    if f.get("personality"):

        parts.append(f["personality"])

    if f.get("voice"):

        parts.append(f"Voice: {f['voice']}")

    if f.get("speech_style"):

        parts.append(f"Speech style: {f['speech_style']}")

    if f.get("mannerisms"):

        parts.append(f"Mannerisms: {f['mannerisms']}")

    if f.get("catchphrases"):

        parts.append(f"Catchphrases: {f['catchphrases']}")

    if f.get("values"):

        parts.append(f"Values: {f['values']}")

    if f.get("likes"):

        parts.append(f"Likes: {f['likes']}")

    if f.get("dislikes"):

        parts.append(f"Dislikes: {f['dislikes']}")

    if f.get("flaws"):

        parts.append(f"Flaws: {f['flaws']}")

    return "\n\n".join(parts).strip()





def build_export_scenario(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    parts = []

    if f.get("current_scenario"):

        parts.append(f["current_scenario"])

    if f.get("setting"):

        parts.append(f"Setting: {f['setting']}")

    if f.get("relationship"):

        parts.append(f"Relationship to user: {f['relationship']}")

    if f.get("world_lore"):

        parts.append(f"World lore: {f['world_lore']}")

    if f.get("goals"):

        parts.append(f"Goals: {f['goals']}")

    if f.get("motivations"):

        parts.append(f"Motivations: {f['motivations']}")

    return "\n\n".join(parts).strip()





def build_system_prompt_export(bot: Dict[str, Any]) -> str:

    f = bot.get("fields", {})

    base = str(f.get("system_prompt", "")).strip()

    extras = []

    if f.get("rules"):

        extras.append(f"Rules: {f['rules']}")

    if f.get("consent_rules"):

        extras.append(f"Consent rules: {f['consent_rules']}")

    if f.get("boundaries"):

        extras.append(f"Boundaries: {f['boundaries']}")

    if f.get("limits"):

        extras.append(f"Limits: {f['limits']}")

    if extras:

        extra_block = "\n".join(extras)

        if base:

            return f"{base}\n\n{extra_block}"

        return extra_block

    return base





def rating_to_tag(rating: str) -> str:

    rating = (rating or "").strip().lower()

    mapping = {

        "sfw": "SFW",

        "suggestive": "Suggestive",

        "nsfw": "NSFW",

        "extreme": "Extreme",

        "unrated": "Unrated",

    }

    return mapping.get(rating, rating.upper() if rating else "")





def merge_tags_with_rating(tags: List[str], rating: str) -> List[str]:

    cleaned = [str(t).strip() for t in (tags or []) if str(t).strip()]

    tag = rating_to_tag(rating)

    if tag and tag not in cleaned:

        cleaned.append(tag)

    return cleaned





def build_card_v2(bot: Dict[str, Any]) -> Dict[str, Any]:

    f = bot.get("fields", {})

    primary, alternates = split_greetings(bot)

    tags = merge_tags_with_rating(bot.get("tags", []), f.get("rating", ""))

    data = {

        "name": f.get("name", ""),

        "description": build_export_description(bot),

        "personality": build_export_personality(bot),

        "scenario": build_export_scenario(bot),

        "first_mes": primary,

        "mes_example": build_mes_example(bot),

        "creator_notes": f.get("author_notes", ""),

        "system_prompt": build_system_prompt_export(bot),

        "post_history_instructions": f.get("post_history_instructions", ""),

        "alternate_greetings": alternates,

        "tags": tags,

        "creator": f.get("creator", ""),

        "character_version": f.get("character_version", ""),

        "extensions": {

            "botmaker": {

                "fields": f,

                "toggles": bot.get("toggles", []),

                "scenarios": bot.get("scenarios", []),

                "memory": bot.get("memory", []),

                "first_messages": bot.get("first_messages", []),

                "example_dialogues": bot.get("example_dialogues", []),

                "prompt_pairs": bot.get("prompt_pairs", []),

                "lorebook": bot.get("lorebook", []),

            }

        },

    }

    return {"spec": "chara_card_v2", "spec_version": "2.0", "data": data}





def build_janitor_export(bot: Dict[str, Any]) -> Dict[str, Any]:

    f = bot.get("fields", {})

    primary, alternates = split_greetings(bot)

    tags = merge_tags_with_rating(bot.get("tags", []), f.get("rating", ""))

    return {

        "name": f.get("name", ""),

        "description": build_export_description(bot),

        "personality": build_export_personality(bot),

        "scenario": build_export_scenario(bot),

        "first_message": primary,

        "alternate_greetings": alternates,

        "example_messages": build_mes_example(bot),

        "tags": tags,

        "creator_notes": f.get("author_notes", ""),

        "system_prompt": build_system_prompt_export(bot),

        "post_history_instructions": f.get("post_history_instructions", ""),

        "rules": f.get("rules", ""),

        "world_lore": f.get("world_lore", ""),

        "nsfw": "Smut" in bot.get("toggles", []),

    }





def build_prompt_export(bot: Dict[str, Any], template: str = "plain") -> str:

    f = bot.get("fields", {})

    primary, _ = split_greetings(bot)

    toggles = ", ".join(bot.get("toggles", []))

    tags = ", ".join(bot.get("tags", []))

    sections = [f"Name: {f.get('name', '')}"]

    if tags:

        sections.append(f"Tags: {tags}")

    if toggles:

        sections.append(f"Toggles: {toggles}")

    sections.extend(

        [

            f"Description: {build_export_description(bot)}",

            f"Personality: {build_export_personality(bot)}",

            f"Scenario: {build_export_scenario(bot)}",

            f"Rules: {f.get('rules', '')}",

            f"Consent rules: {f.get('consent_rules', '')}",

            f"Boundaries: {f.get('boundaries', '')}",

            f"Limits: {f.get('limits', '')}",

            f"System prompt: {f.get('system_prompt', '')}",

            f"Post history instructions: {f.get('post_history_instructions', '')}",

            f"Primary greeting: {primary}",

            f"Example dialogue:\n{build_mes_example(bot)}",

        ]

    )

    base = "\n\n".join([s for s in sections if s.strip()])

    template = (template or "plain").lower()

    if template == "alpaca":

        return f"### Instruction:\\n{base}\\n\\n### Response:\\n"

    if template == "vicuna":

        return f"### System:\\n{base}\\n\\n### User:\\nWrite the next reply.\\n\\n### Assistant:\\n"

    if template == "mistral":

        return f"<s>[INST] {base} [/INST]"

    return base





def has_url_path(base_url: str) -> bool:

    parsed = urlparse(base_url)

    return parsed.path not in {"", "/"}





def strip_endpoint_suffix(base_url: str) -> str:

    base_url = base_url.rstrip("/")

    for suffix in ("/chat/completions", "/messages"):

        if base_url.endswith(suffix):

            return base_url[: -len(suffix)].rstrip("/")

    return base_url





def normalize_base_url(provider: str, base_url: str) -> str:

    base_url = (base_url or "").strip()

    if not base_url:

        return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])["base_url"]

    base_url = strip_endpoint_suffix(base_url)

    if provider == "gemini":

        if "/models/" in base_url and base_url.endswith(":generateContent"):

            base_url = base_url.split("/models/")[0].rstrip("/")

        if not has_url_path(base_url):

            return base_url.rstrip("/") + "/v1beta"

        return base_url

    if provider == "anthropic":

        if not has_url_path(base_url):

            return base_url.rstrip("/") + "/v1"

        return base_url

    if provider == "openrouter":

        if not has_url_path(base_url):

            return base_url.rstrip("/") + "/api/v1"

        return base_url

    if provider in {"openai", "grok", "openai_compatible"}:

        if not has_url_path(base_url):

            return base_url.rstrip("/") + "/v1"

        return base_url

    return base_url





def request_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:

    resp = requests.post(url, headers=headers, json=payload, timeout=90)

    if resp.status_code >= 400:

        try:

            detail = resp.json()

        except Exception:

            detail = resp.text

        raise RuntimeError(f"{resp.status_code} {detail}")

    return resp.json()





def call_openai_like(

    settings: Dict[str, Any],

    system_prompt: str,

    user_prompt: str,

    images: List[Dict[str, str]],

    json_mode: bool = False,

) -> str:

    provider = settings["provider"]

    base_url = normalize_base_url(provider, settings.get("base_url") or PROVIDER_DEFAULTS[provider]["base_url"])

    model = settings.get("model") or PROVIDER_DEFAULTS[provider]["model"]

    headers = {"Content-Type": "application/json"}

    api_key = settings.get("api_key", "")

    if api_key:

        headers["Authorization"] = f"Bearer {api_key}"

    if provider == "openrouter":

        headers["HTTP-Referer"] = "http://localhost"

        headers["X-Title"] = APP_TITLE

    messages: List[Dict[str, Any]] = []

    if system_prompt:

        messages.append({"role": "system", "content": system_prompt})

    if images:

        content = [{"type": "text", "text": user_prompt}]

        for image in images:

            data_url = f"data:{image['mime']};base64,{image['data']}"

            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages.append({"role": "user", "content": content})

    else:

        messages.append({"role": "user", "content": user_prompt})

    payload = {

        "model": model,

        "messages": messages,

        "temperature": settings.get("temperature", 0.7),

        "max_tokens": settings.get("max_tokens", 1200),

    }

    if json_mode:

        payload["response_format"] = {"type": "json_object"}

    url = base_url.rstrip("/") + "/chat/completions"

    try:

        data = request_json(url, headers, payload)

    except RuntimeError:

        if json_mode:

            payload.pop("response_format", None)

            data = request_json(url, headers, payload)

        else:

            raise

    choices = data.get("choices", [])

    if not choices:

        return ""

    message = choices[0].get("message", {})

    return str(message.get("content", ""))





def call_anthropic(settings: Dict[str, Any], system_prompt: str, user_prompt: str, images: List[Dict[str, str]]) -> str:

    base_url = normalize_base_url("anthropic", settings.get("base_url") or PROVIDER_DEFAULTS["anthropic"]["base_url"])

    model = settings.get("model") or PROVIDER_DEFAULTS["anthropic"]["model"]

    headers = {

        "Content-Type": "application/json",

        "x-api-key": settings.get("api_key", ""),

        "anthropic-version": "2023-06-01",

    }

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]

    for image in images:

        content.append({

            "type": "image",

            "source": {

                "type": "base64",

                "media_type": image["mime"],

                "data": image["data"],

            },

        })

    payload = {

        "model": model,

        "max_tokens": settings.get("max_tokens", 1200),

        "temperature": settings.get("temperature", 0.7),

        "messages": [{"role": "user", "content": content}],

    }

    if system_prompt:

        payload["system"] = system_prompt

    url = base_url.rstrip("/") + "/messages"

    data = request_json(url, headers, payload)

    parts = data.get("content", [])

    return "".join(part.get("text", "") for part in parts if part.get("type") == "text")





def call_gemini(settings: Dict[str, Any], system_prompt: str, user_prompt: str, images: List[Dict[str, str]]) -> str:

    base_url = normalize_base_url("gemini", settings.get("base_url") or PROVIDER_DEFAULTS["gemini"]["base_url"])

    model = settings.get("model") or PROVIDER_DEFAULTS["gemini"]["model"]

    headers = {

        "Content-Type": "application/json",

        "x-goog-api-key": settings.get("api_key", ""),

    }

    parts: List[Dict[str, Any]] = [{"text": user_prompt}]

    for image in images:

        parts.append({"inline_data": {"mime_type": image["mime"], "data": image["data"]}})

    payload = {"contents": [{"role": "user", "parts": parts}]}

    if system_prompt:

        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    url = base_url.rstrip("/") + f"/models/{model}:generateContent"

    data = request_json(url, headers, payload)

    candidates = data.get("candidates", [])

    if not candidates:

        return ""

    content = candidates[0].get("content", {})

    text_parts = [p.get("text", "") for p in content.get("parts", []) if p.get("text")]

    return "".join(text_parts)





def call_llm(

    settings: Dict[str, Any],

    system_prompt: str,

    user_prompt: str,

    images: List[Dict[str, str]],

    json_mode: bool = False,

) -> str:

    provider = settings.get("provider", "openai")

    if provider != "openai_compatible" and not settings.get("api_key"):

        raise RuntimeError("API key is required for the selected provider.")

    if provider in {"openai", "openrouter", "grok", "openai_compatible"}:

        return call_openai_like(settings, system_prompt, user_prompt, images, json_mode=json_mode)

    if provider == "anthropic":

        return call_anthropic(settings, system_prompt, user_prompt, images)

    if provider == "gemini":

        return call_gemini(settings, system_prompt, user_prompt, images)

    raise RuntimeError("Unknown provider.")





@app.route("/images/<path:filename>")

def images(filename: str):

    return send_from_directory(IMAGES_DIR, filename)





@app.route("/")

def index():

    ensure_dirs()

    db = load_db()

    if not db["bots"]:

        bot = new_bot()

        db["bots"].append(bot)

        save_db(db)

        return redirect(url_for("edit_bot", bot_id=bot["id"]))

    return redirect(url_for("edit_bot", bot_id=db["bots"][0]["id"]))





@app.route("/bot/new", methods=["POST"])

def create_bot():

    ensure_dirs()

    db = load_db()

    bot = new_bot()

    db["bots"].insert(0, bot)

    save_db(db)

    return redirect(url_for("edit_bot", bot_id=bot["id"]))





@app.route("/bot/<bot_id>")

def edit_bot(bot_id: str):

    ensure_dirs()

    db = load_db()

    for item in db.get("bots", []):

        ensure_bot_defaults(item)

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    ensure_bot_defaults(bot)

    save_db(db)

    settings = load_settings()

    if not bot["fields"].get("creator") and settings.get("creator_name"):

        bot["fields"]["creator"] = settings["creator_name"]

        bot["updated_at"] = now_iso()

        upsert_bot(db, bot)

        save_db(db)

    return render_template_string(

        EDITOR_TEMPLATE,

        app_title=APP_TITLE,

        bot=bot,

        bot_json=bot,

        bots=db["bots"],

        settings=settings,

        providers=PROVIDERS,

        provider_defaults=PROVIDER_DEFAULTS,

        gen_sections=GEN_SECTIONS,

        toggles=TOGGLE_OPTIONS,

        common_tags=COMMON_TAGS,

        generatable_fields=sorted(GENERATABLE_FIELDS),

        defaults=DEFAULT_FIELDS,

        max_images=MAX_IMAGES,

    )





@app.route("/bot/<bot_id>/delete", methods=["POST"])

def delete_bot_route(bot_id: str):

    ensure_dirs()

    db = load_db()

    delete_bot(db, bot_id)

    save_db(db)

    return redirect(url_for("index"))





@app.route("/bot/<bot_id>/duplicate", methods=["POST"])

def duplicate_bot_route(bot_id: str):

    ensure_dirs()

    db = load_db()

    copied = duplicate_bot(db, bot_id)

    if copied:

        save_db(db)

        return redirect(url_for("edit_bot", bot_id=copied["id"]))

    return redirect(url_for("index"))





@app.route("/bot/<bot_id>/export", methods=["GET"])

def export_bot(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    export_path = EXPORTS_DIR / f"{bot_id}.json"

    save_json(export_path, bot)

    return send_from_directory(EXPORTS_DIR, export_path.name, as_attachment=True)





@app.route("/bot/<bot_id>/export/card_v2", methods=["GET"])

def export_card_v2(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    card = build_card_v2(bot)

    export_path = EXPORTS_DIR / f"{bot_id}_card_v2.json"

    save_json(export_path, card)

    return send_from_directory(EXPORTS_DIR, export_path.name, as_attachment=True)





@app.route("/bot/<bot_id>/export/janitor", methods=["GET"])

def export_janitor(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    card = build_janitor_export(bot)

    export_path = EXPORTS_DIR / f"{bot_id}_janitor.json"

    save_json(export_path, card)

    return send_from_directory(EXPORTS_DIR, export_path.name, as_attachment=True)





@app.route("/bot/<bot_id>/export/prompt", methods=["GET"])

def export_prompt(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    template = request.args.get("template", "plain")

    text = build_prompt_export(bot, template)

    filename = f"{bot_id}_prompt_{template}.txt"

    return Response(

        text,

        mimetype="text/plain",

        headers={"Content-Disposition": f"attachment; filename={filename}"},

    )





@app.route("/bot/<bot_id>/export/risu", methods=["GET"])

def export_risu(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    card = build_card_v2(bot)

    export_path = EXPORTS_DIR / f"{bot_id}_risu.json"

    save_json(export_path, card)

    return send_from_directory(EXPORTS_DIR, export_path.name, as_attachment=True)





@app.route("/bot/<bot_id>/export/card_v2_png", methods=["GET"])

def export_card_v2_png(bot_id: str):

    ensure_dirs()

    if not HAS_PIL:

        return jsonify({"error": "Pillow is required for PNG export."}), 400

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return redirect(url_for("index"))

    embed = request.args.get("embed", "1") == "1"

    use_image = request.args.get("use_image", "1") == "1"

    if use_image and bot.get("images"):

        img_path = IMAGES_DIR / bot["images"][0]

        if img_path.exists():

            image = Image.open(img_path).convert("RGBA")

        else:

            image = Image.new("RGBA", (512, 512), (20, 20, 20, 255))

    else:

        image = Image.new("RGBA", (512, 512), (20, 20, 20, 255))

    buffer = io.BytesIO()

    if embed:

        card = build_card_v2(bot)

        card_data = base64.b64encode(json.dumps(card, ensure_ascii=False).encode("utf-8")).decode("ascii")

        info = PngImagePlugin.PngInfo()

        info.add_text("chara", card_data)

        image.save(buffer, format="PNG", pnginfo=info)

    else:

        image.save(buffer, format="PNG")

    buffer.seek(0)

    filename = f"{bot_id}_card_v2.png"

    return Response(

        buffer.getvalue(),

        mimetype="image/png",

        headers={"Content-Disposition": f"attachment; filename={filename}"},

    )





@app.route("/export/all.zip", methods=["GET"])

def export_all():

    ensure_dirs()

    db = load_db()

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        for bot in db.get("bots", []):

            bot_id = bot.get("id", "bot")

            raw = json.dumps(bot, ensure_ascii=False, indent=2)

            zf.writestr(f"{bot_id}/raw.json", raw)

            card = json.dumps(build_card_v2(bot), ensure_ascii=False, indent=2)

            zf.writestr(f"{bot_id}/card_v2.json", card)

            janitor = json.dumps(build_janitor_export(bot), ensure_ascii=False, indent=2)

            zf.writestr(f"{bot_id}/janitor.json", janitor)

            prompt = build_prompt_export(bot, "plain")

            zf.writestr(f"{bot_id}/prompt.txt", prompt)

    buffer.seek(0)

    return Response(

        buffer.getvalue(),

        mimetype="application/zip",

        headers={"Content-Disposition": "attachment; filename=bots_export.zip"},

    )





@app.route("/api/settings", methods=["GET", "POST"])

def api_settings():

    ensure_dirs()

    if request.method == "GET":

        return jsonify(load_settings())

    existing = load_settings()

    payload = request.get_json(silent=True) or {}

    settings = DEFAULT_SETTINGS.copy()

    settings.update({k: v for k, v in payload.items() if k in settings})

    try:

        settings["temperature"] = float(settings.get("temperature", DEFAULT_SETTINGS["temperature"]))

    except Exception:

        settings["temperature"] = DEFAULT_SETTINGS["temperature"]

    try:

        settings["max_tokens"] = int(settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))

    except Exception:

        settings["max_tokens"] = DEFAULT_SETTINGS["max_tokens"]

    try:

        settings["autosave_seconds"] = int(settings.get("autosave_seconds", DEFAULT_SETTINGS["autosave_seconds"]))

    except Exception:

        settings["autosave_seconds"] = DEFAULT_SETTINGS["autosave_seconds"]

    provider = settings.get("provider", "openai")

    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])

    previous_provider = existing.get("provider", "openai")

    previous_defaults = PROVIDER_DEFAULTS.get(previous_provider, PROVIDER_DEFAULTS["openai"])

    if provider != previous_provider:

        if settings.get("model") == previous_defaults.get("model"):

            settings["model"] = defaults["model"]

        if settings.get("base_url") == previous_defaults.get("base_url"):

            settings["base_url"] = defaults["base_url"]

    if not settings.get("model"):

        settings["model"] = defaults["model"]

    if not settings.get("base_url"):

        settings["base_url"] = defaults["base_url"]

    save_settings(settings)

    return jsonify({"ok": True})





@app.route("/api/bot/<bot_id>", methods=["GET", "POST"])

def api_bot(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    if request.method == "GET":

        return jsonify(bot)

    payload = request.get_json(silent=True) or {}

    bot = apply_payload(bot, payload)

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({"ok": True})





@app.route("/api/bot/<bot_id>/upload", methods=["POST"])

def api_upload(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    files = request.files.getlist("images")

    remaining = max(0, MAX_IMAGES - len(bot.get("images", [])))

    if remaining <= 0:

        return jsonify({"error": f"Max {MAX_IMAGES} images reached."}), 400

    saved = []

    for file in files[:remaining]:

        if not file or not file.filename:

            continue

        filename = secure_filename(file.filename)

        ext = Path(filename).suffix.lower() or ".png"

        new_name = f"{uuid.uuid4().hex}{ext}"

        dest = IMAGES_DIR / new_name

        file.save(dest)

        saved.append(new_name)

    if saved:

        bot.setdefault("images", [])

        bot["images"].extend(saved)

        bot["updated_at"] = now_iso()

        upsert_bot(db, bot)

        save_db(db)

    return jsonify({"images": bot.get("images", [])})





@app.route("/api/bot/<bot_id>/remove_image", methods=["POST"])

def api_remove_image(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    filename = payload.get("filename", "")

    if filename and filename in bot.get("images", []):

        bot["images"] = [img for img in bot["images"] if img != filename]

        path = IMAGES_DIR / filename

        if path.exists():

            try:

                path.unlink()

            except Exception:

                pass

        bot["updated_at"] = now_iso()

        upsert_bot(db, bot)

        save_db(db)

    return jsonify({"images": bot.get("images", [])})





@app.route("/api/bot/<bot_id>/generate_profile", methods=["POST"])

def api_generate_profile(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    merge_mode = payload.get("merge_mode", "fill")

    empty_only = bool(payload.get("empty_only"))

    notes = payload.get("notes", "")

    bot = apply_payload(bot, payload)

    gen_sections = normalize_gen_sections(payload.get("gen_sections") or bot.get("gen_sections"))

    bot["gen_sections"] = gen_sections

    allowed_fields = enabled_fields_from_sections(gen_sections)

    if empty_only:

        allowed_fields = [key for key in allowed_fields if not str(bot["fields"].get(key, "")).strip()]

    if not allowed_fields:

        if empty_only:

            return jsonify({"error": "No empty fields found in enabled sections."}), 400

        return jsonify({"error": "Enable at least one generation section."}), 400

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_profile_prompt(bot, notes, allowed_fields)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    if not isinstance(data, dict) or not data:

        return jsonify({

            "error": "Model did not return valid JSON. Try increasing Max Tokens or switching models."

        }), 400

    for key in allowed_fields:

        if key not in data:

            continue

        value = sanitize_text(str(data[key]), allow_emojis)

        bot["fields"][key] = apply_field_merge(bot["fields"].get(key, ""), value, merge_mode, key)

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({"fields": bot["fields"]})





@app.route("/api/bot/<bot_id>/generate_field", methods=["POST"])

def api_generate_field(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    field = str(payload.get("field", "")).strip()

    merge_mode = payload.get("merge_mode", "fill")

    notes = payload.get("notes", "")

    if field not in GENERATABLE_FIELDS:

        return jsonify({"error": "Field cannot be generated."}), 400

    bot = apply_payload(bot, payload)

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_field_prompt(bot, field, notes)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    if not isinstance(data, dict) or field not in data:

        return jsonify({"error": "Model did not return the requested field."}), 400

    value = sanitize_text(str(data[field]), allow_emojis)

    bot["fields"][field] = apply_field_merge(bot["fields"].get(field, ""), value, merge_mode, field)

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({"field": field, "value": bot["fields"][field]})





@app.route("/api/bot/<bot_id>/generate_first_messages", methods=["POST"])

def api_generate_first_messages(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    merge_mode = payload.get("merge_mode", "append")

    notes = payload.get("notes", "")

    count = int(payload.get("count", 3))

    bot = apply_payload(bot, payload)

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_first_messages_prompt(bot, count, notes)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    messages = []

    if isinstance(data, dict):

        messages = normalize_str_list(data.get("first_messages", []))

    elif isinstance(data, list):

        messages = normalize_str_list(data)

    messages = sanitize_list(messages, allow_emojis)

    if merge_mode == "overwrite":

        bot["first_messages"] = messages

    else:

        bot["first_messages"] = bot.get("first_messages", []) + messages

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({"first_messages": bot["first_messages"]})





@app.route("/api/bot/<bot_id>/generate_scenarios", methods=["POST"])

def api_generate_scenarios(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    merge_mode = payload.get("merge_mode", "append")

    notes = payload.get("notes", "")

    count = int(payload.get("count", 5))

    bot = apply_payload(bot, payload)

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_scenarios_prompt(bot, count, notes)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    scenarios = []

    if isinstance(data, dict):

        scenarios = normalize_str_list(data.get("scenarios", []))

    elif isinstance(data, list):

        scenarios = normalize_str_list(data)

    scenarios = sanitize_list(scenarios, allow_emojis)

    scenarios = [clamp_sentences(item) for item in scenarios if item]

    if merge_mode == "overwrite":

        bot["scenarios"] = scenarios

    else:

        bot["scenarios"] = bot.get("scenarios", []) + scenarios

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({"scenarios": bot["scenarios"]})





@app.route("/api/bot/<bot_id>/generate_dialogues", methods=["POST"])

def api_generate_dialogues(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    merge_mode = payload.get("merge_mode", "append")

    notes = payload.get("notes", "")

    count = int(payload.get("count", 4))

    bot = apply_payload(bot, payload)

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_dialogues_prompt(bot, count, notes)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    dialogues = []

    if isinstance(data, dict):

        dialogues = normalize_dialogues(data.get("dialogues", []))

    elif isinstance(data, list):

        dialogues = normalize_dialogues(data)

    dialogues = sanitize_dialogues(dialogues, allow_emojis)

    if merge_mode == "overwrite":

        bot["example_dialogues"] = dialogues

    else:

        bot["example_dialogues"] = bot.get("example_dialogues", []) + dialogues

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({"example_dialogues": bot["example_dialogues"]})





@app.route("/api/bot/<bot_id>/generate_simple", methods=["POST"])

def api_generate_simple(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    bot = apply_payload(bot, payload)

    simple_input = str(bot.get("fields", {}).get("simple_input", "")).strip()

    if not simple_input:

        return jsonify({"error": "Original input is required."}), 400

    notes = payload.get("notes", "")

    try:

        dialogue_count = int(payload.get("dialogue_count", 3))

    except Exception:

        dialogue_count = 3

    dialogue_count = max(1, min(8, dialogue_count))

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_simple_prompt(bot, simple_input, dialogue_count, notes)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    if not isinstance(data, dict):

        data = {}

    description = sanitize_text(str(data.get("description", "")), allow_emojis)

    description = normalize_description_sections(description)

    personality = sanitize_text(str(data.get("personality", "")), allow_emojis)

    scenario = ""

    if isinstance(data.get("scenario"), str):

        scenario = data.get("scenario", "").strip()

    elif isinstance(data.get("scenario"), list):

        scenarios = normalize_str_list(data.get("scenario", []))

        scenario = scenarios[0] if scenarios else ""

    elif isinstance(data.get("scenarios"), list):

        scenarios = normalize_str_list(data.get("scenarios", []))

        scenario = scenarios[0] if scenarios else ""

    scenario = clamp_sentences(sanitize_text(scenario, allow_emojis), 2)

    first_message = ""

    if isinstance(data.get("first_message"), str):

        first_message = data.get("first_message", "").strip()

    elif isinstance(data.get("first_messages"), list):

        messages = normalize_str_list(data.get("first_messages", []))

        first_message = messages[0] if messages else ""

    first_message = sanitize_text(first_message, allow_emojis)

    dialogues: List[Dict[str, str]] = []

    if isinstance(data.get("dialogues"), list):

        dialogues = normalize_dialogues(data.get("dialogues", []))

    elif isinstance(data.get("example_dialogues"), list):

        dialogues = normalize_dialogues(data.get("example_dialogues", []))

    dialogues = sanitize_dialogues(dialogues, allow_emojis)

    if description:

        bot["fields"]["description"] = description

    if personality:

        bot["fields"]["personality"] = personality

    if scenario:

        bot["fields"]["current_scenario"] = scenario

        bot["scenarios"] = [scenario]

    if first_message:

        bot["first_messages"] = [first_message]

        if not str(bot["fields"].get("primary_first_message", "")).strip():

            bot["fields"]["primary_first_message"] = first_message

    if dialogues:

        bot["example_dialogues"] = dialogues

    bot["updated_at"] = now_iso()

    upsert_bot(db, bot)

    save_db(db)

    return jsonify({

        "fields": {

            "description": bot["fields"].get("description", ""),

            "personality": bot["fields"].get("personality", ""),

            "current_scenario": bot["fields"].get("current_scenario", ""),

            "primary_first_message": bot["fields"].get("primary_first_message", ""),

        },

        "first_messages": bot.get("first_messages", []),

        "scenarios": bot.get("scenarios", []),

        "example_dialogues": bot.get("example_dialogues", []),

    })





@app.route("/api/bot/<bot_id>/compile", methods=["POST"])

def api_compile(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    notes = payload.get("notes", "")

    bot = apply_payload(bot, payload)

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) else []

    prompt = build_compile_prompt(bot, notes)

    try:

        response = call_llm(settings, "", prompt, images, json_mode=True)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    data = extract_json(response)

    if not isinstance(data, dict):

        data = {}

    description = sanitize_text(str(data.get("description", "")), allow_emojis)

    scenario = ""

    if isinstance(data.get("scenario"), str):

        scenario = data.get("scenario", "").strip()

    elif isinstance(data.get("scenarios"), list):

        scenarios = normalize_str_list(data.get("scenarios", []))

        scenario = scenarios[0] if scenarios else ""

    scenario = clamp_sentences(sanitize_text(scenario, allow_emojis), 2)

    first_messages: List[str] = []

    if isinstance(data.get("first_messages"), list):

        first_messages = normalize_str_list(data.get("first_messages", []))

    elif isinstance(data.get("first_message"), str):

        first_messages = normalize_str_list([data.get("first_message", "")])

    first_messages = sanitize_list(first_messages, allow_emojis)[:2]

    dialogues: List[Dict[str, str]] = []

    if isinstance(data.get("dialogues"), list):

        dialogues = normalize_dialogues(data.get("dialogues", []))

    elif isinstance(data.get("example_dialogues"), list):

        dialogues = normalize_dialogues(data.get("example_dialogues", []))

    dialogues = sanitize_dialogues(dialogues, allow_emojis)[:4]

    if not (description or scenario or first_messages or dialogues):

        return jsonify({

            "error": "Model returned empty output. Try increasing Max Tokens or switching models."

        }), 400

    return jsonify({

        "compiled": {

            "description": description,

            "scenario": scenario,

            "first_messages": first_messages,

            "dialogues": dialogues,

        },

    })





@app.route("/api/bot/<bot_id>/chat", methods=["POST"])

def api_chat(bot_id: str):

    ensure_dirs()

    db = load_db()

    bot = get_bot(db, bot_id)

    if not bot:

        return jsonify({"error": "Bot not found"}), 404

    payload = request.get_json(silent=True) or {}

    allow_emojis = parse_bool(payload.get("allow_emojis"))

    chat_mode = str(payload.get("chat_mode", "assistant")).strip().lower()

    chat_use_images = parse_bool(payload.get("chat_use_images", True))

    message = str(payload.get("message", "")).strip()

    if not message:

        return jsonify({"error": "Message is required"}), 400

    bot = apply_payload(bot, payload)

    settings = load_settings()

    images = load_image_data(bot) if settings.get("use_images", True) and chat_use_images else []

    if chat_mode == "character":

        system_prompt = build_chat_system(bot)

    else:

        system_prompt = build_chat_system_assistant(bot)

    if allow_emojis:

        system_prompt = f"{system_prompt}\nEmojis are allowed."

    else:

        system_prompt = f"{system_prompt}\nNo emojis."

    try:

        response = call_llm(settings, system_prompt, message, images)

    except Exception as exc:

        return jsonify({"error": str(exc)}), 400

    response = sanitize_text(response, allow_emojis)

    return jsonify({"response": response})





EDITOR_TEMPLATE = """

<!doctype html>

<html lang="en">

<head>

  <meta charset="utf-8" />

  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <title>{{ app_title }}</title>

  <style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {

      --bg: #f6f1e6;

      --bg-2: #e8f3f1;

      --ink: #1f1d1a;

      --muted: #6e6259;

      --accent: #0f766e;

      --accent-2: #f97316;

      --card: rgba(255, 255, 255, 0.92);

      --border: #e7d8c4;

      --shadow: rgba(31, 29, 26, 0.08);

      --surface: #ffffff;

      --input-bg: #ffffff;

      --soft: #f2e8da;

      --user-bubble: #fff6ed;

      --bot-bubble: #f0faf8;

      --user-border: #f3c79f;

      --bot-border: #b9dfd8;

      --ok-bg: #e6f6f4;

      --ok-border: #8fd4c8;

      --ok-ink: #1f5f58;

      --error-bg: #fde6df;

      --error-border: #f1b7a8;

      --error-ink: #8b2e1f;

      --chip-bg: #ffffff;

      --glow-1: #ffe7c0;

      --glow-1-fade: rgba(255, 231, 192, 0);

      --glow-2: #d8f0ec;

      --glow-2-fade: rgba(216, 240, 236, 0);

      --focus-ring: rgba(15, 118, 110, 0.25);

      --fx-overlay-opacity: 0;
      --fx-overlay-bg: none;
      --fx-overlay-blur: 0px;
      --fx-overlay-size: auto;
      --fx-overlay-anim: none;
      --fx-vignette-opacity: 0;
      --fx-vignette-color: rgba(0, 0, 0, 0.6);

    }

    :root[data-theme="dark"] {

      --bg: #12100d;

      --bg-2: #1b1813;

      --ink: #f6efe5;

      --muted: #b6aa9d;

      --accent: #14b8a6;

      --accent-2: #fb923c;

      --card: rgba(26, 24, 20, 0.95);

      --border: #2e2922;

      --shadow: rgba(0, 0, 0, 0.35);

      --surface: #1b1813;

      --input-bg: #221f19;

      --soft: #2a251f;

      --user-bubble: #2c2118;

      --bot-bubble: #182320;

      --user-border: #7a4a2a;

      --bot-border: #2b6f65;

      --ok-bg: #17332f;

      --ok-border: #2b6f65;

      --ok-ink: #cfeee9;

      --error-bg: #3b1f17;

      --error-border: #7a3527;

      --error-ink: #f7c3b6;

      --chip-bg: #201c17;

      --glow-1: #2d1f12;

      --glow-1-fade: rgba(45, 31, 18, 0);

      --glow-2: #142725;

      --glow-2-fade: rgba(20, 39, 37, 0);

      --focus-ring: rgba(20, 184, 166, 0.35);

    }

    :root[data-theme="stellar"] {
      --bg: #0b1020;
      --bg-2: #101a33;
      --ink: #f5f7ff;
      --muted: #aab4cc;
      --accent: #7dd3fc;
      --accent-2: #e2e8f0;
      --card: rgba(15, 22, 40, 0.9);
      --border: #263253;
      --shadow: rgba(4, 8, 18, 0.6);
      --surface: #121a30;
      --input-bg: #101827;
      --soft: #1b2540;
      --user-bubble: #111a33;
      --bot-bubble: #0f1f2e;
      --user-border: #2e3e66;
      --bot-border: #1f5b74;
      --ok-bg: #0d2533;
      --ok-border: #3aaed8;
      --ok-ink: #cdefff;
      --error-bg: #33151b;
      --error-border: #7d2b39;
      --error-ink: #f6c3cf;
      --chip-bg: #121a30;
      --glow-1: #223b73;
      --glow-1-fade: rgba(34, 59, 115, 0);
      --glow-2: #12345a;
      --glow-2-fade: rgba(18, 52, 90, 0);
      --focus-ring: rgba(125, 211, 252, 0.35);
      --fx-overlay-opacity: 0.35;
      --fx-overlay-bg: radial-gradient(1px 1px at 20% 30%, rgba(255, 255, 255, 0.6), transparent 60%),
        radial-gradient(1px 1px at 80% 70%, rgba(255, 255, 255, 0.4), transparent 60%),
        radial-gradient(1.5px 1.5px at 60% 20%, rgba(255, 255, 255, 0.5), transparent 60%),
        radial-gradient(1px 1px at 30% 80%, rgba(255, 255, 255, 0.35), transparent 60%);
      --fx-overlay-size: 240px 240px;
      --fx-overlay-anim: twinkle 10s ease-in-out infinite;
    }

    :root[data-theme="nebula"] {
      --bg: #120a22;
      --bg-2: #1c0f33;
      --ink: #f5f1ff;
      --muted: #b6a8d3;
      --accent: #a855f7;
      --accent-2: #22d3ee;
      --card: rgba(26, 16, 44, 0.92);
      --border: #3b2657;
      --shadow: rgba(10, 4, 24, 0.55);
      --surface: #1c1430;
      --input-bg: #21183a;
      --soft: #2a1d46;
      --user-bubble: #23183e;
      --bot-bubble: #152837;
      --user-border: #4b2b70;
      --bot-border: #1f6f7a;
      --ok-bg: #112b2f;
      --ok-border: #32c4d4;
      --ok-ink: #d0f4f9;
      --error-bg: #3a1424;
      --error-border: #8a3250;
      --error-ink: #f7cadc;
      --chip-bg: #22183a;
      --glow-1: #3d1f6f;
      --glow-1-fade: rgba(61, 31, 111, 0);
      --glow-2: #123f62;
      --glow-2-fade: rgba(18, 63, 98, 0);
      --focus-ring: rgba(168, 85, 247, 0.45);
      --fx-overlay-opacity: 0.5;
      --fx-overlay-bg: radial-gradient(circle at 20% 20%, rgba(168, 85, 247, 0.35), transparent 55%),
        radial-gradient(circle at 80% 10%, rgba(34, 211, 238, 0.28), transparent 55%),
        radial-gradient(circle at 30% 80%, rgba(236, 72, 153, 0.3), transparent 60%);
      --fx-overlay-size: 200% 200%;
      --fx-overlay-anim: nebulaShift 18s ease infinite;
    }

    :root[data-theme="solar"] {
      --bg: #fff1cf;
      --bg-2: #ffe2a8;
      --ink: #3f2608;
      --muted: #8a5b2c;
      --accent: #f59e0b;
      --accent-2: #f97316;
      --card: rgba(255, 255, 255, 0.93);
      --border: #f6c687;
      --shadow: rgba(120, 68, 15, 0.15);
      --surface: #fff7e8;
      --input-bg: #fff4de;
      --soft: #ffe5be;
      --user-bubble: #fff5e4;
      --bot-bubble: #fff1d4;
      --user-border: #f2b975;
      --bot-border: #f7c48c;
      --ok-bg: #f4fbe8;
      --ok-border: #a7d67c;
      --ok-ink: #36591c;
      --error-bg: #ffe6dc;
      --error-border: #f5b29c;
      --error-ink: #7a2d1f;
      --chip-bg: #fff3da;
      --glow-1: #ffdf9b;
      --glow-1-fade: rgba(255, 223, 155, 0);
      --glow-2: #ffd0a1;
      --glow-2-fade: rgba(255, 208, 161, 0);
      --focus-ring: rgba(245, 158, 11, 0.35);
      --fx-overlay-opacity: 0.3;
      --fx-overlay-bg: radial-gradient(circle at 20% 20%, rgba(245, 158, 11, 0.25), transparent 60%),
        radial-gradient(circle at 80% 30%, rgba(249, 115, 22, 0.2), transparent 60%);
      --fx-overlay-size: 160% 160%;
      --fx-overlay-anim: solarPulse 7s ease-in-out infinite;
    }

    :root[data-theme="lunar"] {
      --bg: #eef2f7;
      --bg-2: #dfe7f2;
      --ink: #1b2430;
      --muted: #607287;
      --accent: #6ea8fe;
      --accent-2: #9aa7ff;
      --card: rgba(255, 255, 255, 0.92);
      --border: #ccd6e3;
      --shadow: rgba(25, 38, 60, 0.08);
      --surface: #f6f9fd;
      --input-bg: #f1f5fb;
      --soft: #e8eef7;
      --user-bubble: #f7f9fe;
      --bot-bubble: #eef4fb;
      --user-border: #c3d3ea;
      --bot-border: #b9c7df;
      --ok-bg: #eaf6ff;
      --ok-border: #93c5fd;
      --ok-ink: #1e3a5f;
      --error-bg: #ffe8ea;
      --error-border: #f2b3bc;
      --error-ink: #7a1f2b;
      --chip-bg: #f4f7fb;
      --glow-1: #d7e6ff;
      --glow-1-fade: rgba(215, 230, 255, 0);
      --glow-2: #cddaf7;
      --glow-2-fade: rgba(205, 218, 247, 0);
      --focus-ring: rgba(110, 168, 254, 0.35);
      --fx-overlay-opacity: 0.22;
      --fx-overlay-bg: radial-gradient(circle at 30% 30%, rgba(140, 170, 210, 0.2), transparent 60%);
      --fx-overlay-size: 160% 160%;
      --fx-overlay-anim: lunarDrift 18s ease-in-out infinite;
    }

    :root[data-theme="eclipse"] {
      --bg: #07060b;
      --bg-2: #130b20;
      --ink: #f7f1ff;
      --muted: #b7a6d1;
      --accent: #ef4444;
      --accent-2: #7c3aed;
      --card: rgba(15, 10, 22, 0.92);
      --border: #2b1c3f;
      --shadow: rgba(0, 0, 0, 0.65);
      --surface: #0f0a1a;
      --input-bg: #141025;
      --soft: #1b132e;
      --user-bubble: #1a0f22;
      --bot-bubble: #0f1422;
      --user-border: #4a2740;
      --bot-border: #2a3c5a;
      --ok-bg: #0f2a26;
      --ok-border: #22c55e;
      --ok-ink: #c7f9e8;
      --error-bg: #3d121a;
      --error-border: #8b1e2e;
      --error-ink: #f5b9c3;
      --chip-bg: #161022;
      --glow-1: #2b103f;
      --glow-1-fade: rgba(43, 16, 63, 0);
      --glow-2: #1a0b2d;
      --glow-2-fade: rgba(26, 11, 45, 0);
      --focus-ring: rgba(239, 68, 68, 0.4);
      --fx-overlay-opacity: 0.2;
      --fx-overlay-bg: radial-gradient(circle at 40% 20%, rgba(124, 58, 237, 0.2), transparent 60%),
        radial-gradient(circle at 70% 70%, rgba(239, 68, 68, 0.15), transparent 60%);
      --fx-overlay-size: 200% 200%;
      --fx-overlay-anim: eclipseFade 16s ease-in-out infinite;
      --fx-vignette-opacity: 0.55;
      --fx-vignette-color: rgba(0, 0, 0, 0.75);
    }

    :root[data-theme="aurora"] {
      --bg: #091a19;
      --bg-2: #0b2328;
      --ink: #e6fdf9;
      --muted: #9cd8cf;
      --accent: #34d399;
      --accent-2: #7c3aed;
      --card: rgba(12, 27, 32, 0.9);
      --border: #1d3a3a;
      --shadow: rgba(5, 16, 16, 0.5);
      --surface: #0f2326;
      --input-bg: #0f2326;
      --soft: #123136;
      --user-bubble: #0e1f26;
      --bot-bubble: #0c2623;
      --user-border: #1d4a4a;
      --bot-border: #1d5a5a;
      --ok-bg: #0b2f26;
      --ok-border: #34d399;
      --ok-ink: #c6f7e8;
      --error-bg: #3a1419;
      --error-border: #8b2a35;
      --error-ink: #f4b9c2;
      --chip-bg: #11262a;
      --glow-1: #0f3a2a;
      --glow-1-fade: rgba(15, 58, 42, 0);
      --glow-2: #3a1b5a;
      --glow-2-fade: rgba(58, 27, 90, 0);
      --focus-ring: rgba(52, 211, 153, 0.4);
      --fx-overlay-opacity: 0.4;
      --fx-overlay-bg: linear-gradient(120deg, rgba(52, 211, 153, 0.25), rgba(34, 211, 238, 0.2), rgba(124, 58, 237, 0.25));
      --fx-overlay-size: 220% 220%;
      --fx-overlay-anim: auroraFlow 20s ease infinite;
    }

    :root[data-theme="cosmos"] {
      --bg: #070b18;
      --bg-2: #0b152c;
      --ink: #f2f6ff;
      --muted: #a6b0cc;
      --accent: #60a5fa;
      --accent-2: #f8fafc;
      --card: rgba(13, 18, 35, 0.9);
      --border: #25325a;
      --shadow: rgba(2, 5, 15, 0.6);
      --surface: #0f172a;
      --input-bg: #111a30;
      --soft: #19243f;
      --user-bubble: #111a32;
      --bot-bubble: #0f1c2e;
      --user-border: #2b3c66;
      --bot-border: #1d5674;
      --ok-bg: #0d2a36;
      --ok-border: #38bdf8;
      --ok-ink: #dbeafe;
      --error-bg: #32131d;
      --error-border: #7a2d3b;
      --error-ink: #f7c4d3;
      --chip-bg: #121a30;
      --glow-1: #1b2f5f;
      --glow-1-fade: rgba(27, 47, 95, 0);
      --glow-2: #0c2a45;
      --glow-2-fade: rgba(12, 42, 69, 0);
      --focus-ring: rgba(96, 165, 250, 0.4);
      --fx-overlay-opacity: 0.25;
      --fx-overlay-bg: radial-gradient(1px 1px at 25% 35%, rgba(255, 255, 255, 0.5), transparent 60%),
        radial-gradient(1px 1px at 75% 55%, rgba(255, 255, 255, 0.35), transparent 60%),
        radial-gradient(1.5px 1.5px at 55% 15%, rgba(255, 255, 255, 0.45), transparent 60%);
      --fx-overlay-size: 260px 260px;
      --fx-overlay-anim: cosmosDrift 26s linear infinite;
    }

    :root[data-theme="supernova"] {
      --bg: #0a0f1f;
      --bg-2: #121a33;
      --ink: #f5f7ff;
      --muted: #a8b3d8;
      --accent: #3b82f6;
      --accent-2: #ec4899;
      --card: rgba(15, 20, 40, 0.9);
      --border: #2a355d;
      --shadow: rgba(6, 8, 20, 0.6);
      --surface: #10182d;
      --input-bg: #121b35;
      --soft: #1a2542;
      --user-bubble: #121a35;
      --bot-bubble: #101b2b;
      --user-border: #2f3e6e;
      --bot-border: #2b4f6e;
      --ok-bg: #0d2a36;
      --ok-border: #22d3ee;
      --ok-ink: #dbeafe;
      --error-bg: #351423;
      --error-border: #8b2a4f;
      --error-ink: #f7c4da;
      --chip-bg: #121b35;
      --glow-1: #1f3b6f;
      --glow-1-fade: rgba(31, 59, 111, 0);
      --glow-2: #51204f;
      --glow-2-fade: rgba(81, 32, 79, 0);
      --focus-ring: rgba(59, 130, 246, 0.45);
      --fx-overlay-opacity: 0.35;
      --fx-overlay-bg: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.25), transparent 55%),
        radial-gradient(circle at 70% 30%, rgba(236, 72, 153, 0.2), transparent 55%);
      --fx-overlay-size: 180% 180%;
      --fx-overlay-anim: supernovaPulse 8s ease-in-out infinite;
    }

    :root[data-theme="void"] {
      --bg: #050608;
      --bg-2: #0c0d12;
      --ink: #e5e7eb;
      --muted: #9aa0ac;
      --accent: #7c3aed;
      --accent-2: #64748b;
      --card: rgba(10, 12, 16, 0.92);
      --border: #20222b;
      --shadow: rgba(0, 0, 0, 0.6);
      --surface: #0c0f14;
      --input-bg: #0e1118;
      --soft: #151922;
      --user-bubble: #0e1218;
      --bot-bubble: #0b1416;
      --user-border: #262b36;
      --bot-border: #1d2e36;
      --ok-bg: #0f2a2a;
      --ok-border: #2dd4bf;
      --ok-ink: #c7f9f1;
      --error-bg: #2c121a;
      --error-border: #7a2733;
      --error-ink: #f2b6c2;
      --chip-bg: #0f1117;
      --glow-1: #0b0f1a;
      --glow-1-fade: rgba(11, 15, 26, 0);
      --glow-2: #12101a;
      --glow-2-fade: rgba(18, 16, 26, 0);
      --focus-ring: rgba(124, 58, 237, 0.35);
      --fx-overlay-opacity: 0.08;
      --fx-overlay-bg: radial-gradient(circle at 50% 50%, rgba(124, 58, 237, 0.08), transparent 60%);
      --fx-overlay-size: 160% 160%;
      --fx-overlay-anim: voidBreath 18s ease-in-out infinite;
      --fx-vignette-opacity: 0.4;
      --fx-vignette-color: rgba(0, 0, 0, 0.8);
    }

    :root[data-theme="orbit"] {
      --bg: #e9eff6;
      --bg-2: #d6e2f1;
      --ink: #1f2a37;
      --muted: #64748b;
      --accent: #60a5fa;
      --accent-2: #f97316;
      --card: rgba(255, 255, 255, 0.92);
      --border: #cbd5e1;
      --shadow: rgba(30, 41, 59, 0.1);
      --surface: #f7fafc;
      --input-bg: #f1f5f9;
      --soft: #e2e8f0;
      --user-bubble: #f5f9ff;
      --bot-bubble: #edf4ff;
      --user-border: #c2d4ef;
      --bot-border: #a7c1e8;
      --ok-bg: #eaf6ff;
      --ok-border: #60a5fa;
      --ok-ink: #1e3a8a;
      --error-bg: #ffe8e3;
      --error-border: #f5b7a0;
      --error-ink: #7a2d1f;
      --chip-bg: #f3f6fb;
      --glow-1: #c7ddff;
      --glow-1-fade: rgba(199, 221, 255, 0);
      --glow-2: #ffd9b0;
      --glow-2-fade: rgba(255, 217, 176, 0);
      --focus-ring: rgba(96, 165, 250, 0.35);
      --fx-overlay-opacity: 0.35;
      --fx-overlay-bg: radial-gradient(circle at 30% 30%, rgba(96, 165, 250, 0.25), transparent 60%),
        radial-gradient(circle at 70% 70%, rgba(249, 115, 22, 0.2), transparent 60%);
      --fx-overlay-size: 180% 180%;
      --fx-overlay-anim: orbitDrift 22s ease-in-out infinite;
    }

    :root[data-theme="plasma"] {
      --bg: #06060d;
      --bg-2: #100b1a;
      --ink: #f4f2ff;
      --muted: #b7a7d6;
      --accent: #22d3ee;
      --accent-2: #a855f7;
      --card: rgba(14, 10, 22, 0.92);
      --border: #2a1c40;
      --shadow: rgba(0, 0, 0, 0.6);
      --surface: #120d1f;
      --input-bg: #140f24;
      --soft: #1c1330;
      --user-bubble: #1a1228;
      --bot-bubble: #14202b;
      --user-border: #3a2a5a;
      --bot-border: #1f4b5a;
      --ok-bg: #0f2a36;
      --ok-border: #22d3ee;
      --ok-ink: #dbeafe;
      --error-bg: #351423;
      --error-border: #8a2e5c;
      --error-ink: #f6c4db;
      --chip-bg: #141022;
      --glow-1: #2b1f5f;
      --glow-1-fade: rgba(43, 31, 95, 0);
      --glow-2: #0e3a4a;
      --glow-2-fade: rgba(14, 58, 74, 0);
      --focus-ring: rgba(34, 211, 238, 0.45);
      --fx-overlay-opacity: 0.4;
      --fx-overlay-bg: linear-gradient(140deg, rgba(34, 211, 238, 0.25), rgba(168, 85, 247, 0.28), rgba(15, 23, 42, 0.1));
      --fx-overlay-size: 220% 220%;
      --fx-overlay-anim: plasmaSurge 12s ease-in-out infinite;
    }

    :root[data-theme="meteor"] {
      --bg: #1a1411;
      --bg-2: #231613;
      --ink: #f7f1eb;
      --muted: #c1a79a;
      --accent: #f97316;
      --accent-2: #ef4444;
      --card: rgba(28, 20, 16, 0.92);
      --border: #3b2a20;
      --shadow: rgba(0, 0, 0, 0.5);
      --surface: #221913;
      --input-bg: #281c15;
      --soft: #2f2018;
      --user-bubble: #2a1b14;
      --bot-bubble: #1f201a;
      --user-border: #5a3a2a;
      --bot-border: #3b4a3f;
      --ok-bg: #1f2a23;
      --ok-border: #34d399;
      --ok-ink: #c7f9e8;
      --error-bg: #3a1316;
      --error-border: #8b2a2a;
      --error-ink: #f4b7b7;
      --chip-bg: #231913;
      --glow-1: #3a2014;
      --glow-1-fade: rgba(58, 32, 20, 0);
      --glow-2: #3a1a16;
      --glow-2-fade: rgba(58, 26, 22, 0);
      --focus-ring: rgba(249, 115, 22, 0.4);
      --fx-overlay-opacity: 0.35;
      --fx-overlay-bg: linear-gradient(120deg, rgba(249, 115, 22, 0.15), rgba(15, 23, 42, 0) 35%, rgba(239, 68, 68, 0.12) 70%, rgba(15, 23, 42, 0));
      --fx-overlay-size: 220% 220%;
      --fx-overlay-anim: meteorShift 6s linear infinite;
    }

    @media (prefers-color-scheme: dark) {

      :root:not([data-theme]) {

        --bg: #12100d;

        --bg-2: #1b1813;

        --ink: #f6efe5;

        --muted: #b6aa9d;

        --accent: #14b8a6;

        --accent-2: #fb923c;

        --card: rgba(26, 24, 20, 0.95);

        --border: #2e2922;

        --shadow: rgba(0, 0, 0, 0.35);

        --surface: #1b1813;

        --input-bg: #221f19;

        --soft: #2a251f;

        --user-bubble: #2c2118;

        --bot-bubble: #182320;

        --user-border: #7a4a2a;

        --bot-border: #2b6f65;

        --ok-bg: #17332f;

        --ok-border: #2b6f65;

        --ok-ink: #cfeee9;

        --error-bg: #3b1f17;

        --error-border: #7a3527;

        --error-ink: #f7c3b6;

        --chip-bg: #201c17;

        --glow-1: #2d1f12;

        --glow-1-fade: rgba(45, 31, 18, 0);

        --glow-2: #142725;

        --glow-2-fade: rgba(20, 39, 37, 0);

        --focus-ring: rgba(20, 184, 166, 0.35);

      }

    }

    * {

      box-sizing: border-box;

    }

    body {

      margin: 0;

      font-family: "Space Grotesk", sans-serif;

      color: var(--ink);

      position: relative;
      isolation: isolate;

      background:

        radial-gradient(800px 420px at 10% -10%, var(--glow-1) 0%, var(--glow-1-fade) 60%),

        radial-gradient(900px 520px at 110% 0%, var(--glow-2) 0%, var(--glow-2-fade) 60%),

        linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);

      min-height: 100vh;

      transition: background 0.25s ease, color 0.25s ease;

    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
    }

    body::before {
      background: var(--fx-overlay-bg);
      background-size: var(--fx-overlay-size);
      background-position: 0% 0%;
      opacity: var(--fx-overlay-opacity);
      filter: blur(var(--fx-overlay-blur));
      mix-blend-mode: screen;
      transform-origin: center;
      animation: var(--fx-overlay-anim);
    }

    body::after {
      background: radial-gradient(ellipse at center, rgba(0, 0, 0, 0) 45%, var(--fx-vignette-color) 100%);
      opacity: var(--fx-vignette-opacity);
    }

    body > * {
      position: relative;
      z-index: 1;
    }

    @keyframes twinkle {
      0% { transform: translateY(0); }
      50% { transform: translateY(-8px); }
      100% { transform: translateY(0); }
    }

    @keyframes nebulaShift {
      0% { background-position: 0% 0%; }
      50% { background-position: 100% 60%; }
      100% { background-position: 0% 0%; }
    }

    @keyframes solarPulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.04); }
      100% { transform: scale(1); }
    }

    @keyframes lunarDrift {
      0% { background-position: 0% 0%; }
      50% { background-position: 60% 40%; }
      100% { background-position: 0% 0%; }
    }

    @keyframes eclipseFade {
      0% { opacity: 0.15; }
      50% { opacity: 0.25; }
      100% { opacity: 0.15; }
    }

    @keyframes auroraFlow {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    @keyframes cosmosDrift {
      0% { background-position: 0% 0%; }
      100% { background-position: 200% 200%; }
    }

    @keyframes supernovaPulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.06); }
      100% { transform: scale(1); }
    }

    @keyframes voidBreath {
      0% { opacity: 0.06; }
      50% { opacity: 0.12; }
      100% { opacity: 0.06; }
    }

    @keyframes orbitDrift {
      0% { transform: translateY(0) rotate(0deg); }
      50% { transform: translateY(-10px) rotate(180deg); }
      100% { transform: translateY(0) rotate(360deg); }
    }

    @keyframes plasmaSurge {
      0% { background-position: 0% 40%; }
      50% { background-position: 100% 60%; }
      100% { background-position: 0% 40%; }
    }

    @keyframes meteorShift {
      0% { background-position: 0% 0%; }
      100% { background-position: 200% 200%; }
    }

    header {

      display: flex;

      align-items: center;

      justify-content: space-between;

      padding: 20px 26px;

    }

    .brand {

      font-family: "Fraunces", serif;

      font-size: 28px;

    }

    .sub {

      color: var(--muted);

      font-size: 14px;

      margin-top: 2px;

    }

    .layout {

      display: grid;

      grid-template-columns: 320px 1fr;

      gap: 18px;

      padding: 0 20px 40px;

    }

    .sidebar {

      position: sticky;

      top: 14px;

      align-self: start;

      display: flex;

      flex-direction: column;

      gap: 16px;

    }

    .panel {

      background: var(--card);

      border: 1px solid var(--border);

      border-radius: 16px;

      padding: 16px;

      box-shadow: 0 10px 24px var(--shadow);

    }

    .panel h3 {

      margin: 0 0 10px;

      font-size: 16px;

    }

    .panel h2 {

      margin: 0 0 12px;

      font-family: "Fraunces", serif;

      font-size: 20px;

    }

    .panel .hint {

      color: var(--muted);

      font-size: 12px;

    }

    .panel-title {

      display: flex;

      align-items: center;

      justify-content: space-between;

      gap: 12px;

    }

    .main {

      display: flex;

      flex-direction: column;

      gap: 18px;

    }

    .grid {

      display: grid;

      gap: 12px;

      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));

    }

    .nav-list {

      display: grid;

      gap: 8px;

    }

    .nav-list a {

      padding: 8px 10px;

      border-radius: 10px;

      border: 1px solid var(--border);

      background: var(--surface);

      text-decoration: none;

      color: var(--ink);

      font-size: 13px;

    }

    .nav-list a:hover {

      border-color: var(--accent);

    }

    label {

      display: block;

      font-size: 12px;

      text-transform: uppercase;

      letter-spacing: 0.04em;

      margin-bottom: 6px;

      color: var(--muted);

    }

    input[type="text"], input[type="number"], select, textarea {

      width: 100%;

      padding: 10px 12px;

      border: 1px solid var(--border);

      border-radius: 12px;

      font-family: inherit;

      font-size: 14px;

      background: var(--input-bg);

      color: var(--ink);

    }

    input[type="text"]:focus, input[type="number"]:focus, select:focus, textarea:focus {

      outline: none;

      border-color: var(--accent);

      box-shadow: 0 0 0 3px var(--focus-ring);

    }

    textarea {

      min-height: 90px;

      resize: vertical;

    }

    .row {

      display: flex;

      align-items: center;

      gap: 10px;

      flex-wrap: wrap;

    }

    .theme-control {

      display: flex;

      align-items: center;

      gap: 8px;

    }

    .mode-toggle {

      display: inline-flex;

      align-items: center;

      gap: 4px;

      padding: 4px;

      border-radius: 999px;

      border: 1px solid var(--border);

      background: var(--soft);

    }

    .mode-toggle button {

      border: none;

      background: transparent;

      color: var(--muted);

      padding: 6px 14px;

      border-radius: 999px;

      font-weight: 600;

      cursor: pointer;

    }

    .mode-toggle button.active {

      background: var(--accent);

      color: #fff;

      box-shadow: 0 6px 12px rgba(15, 118, 110, 0.2);

    }

    .token-toggle {

      flex-wrap: wrap;

      justify-content: flex-start;

    }

    .token-toggle button {

      padding: 4px 10px;

      font-size: 11px;

    }

    .token-targets {

      display: grid;

      gap: 12px;

      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));

    }

    .token-target {

      display: flex;

      flex-direction: column;

      gap: 6px;

    }

    .chat-mode-toggle button {

      padding: 4px 10px;

      font-size: 12px;

    }

    :root.mode-simple .advanced-only {

      display: none !important;

    }

    :root.mode-advanced .simple-only {

      display: none !important;

    }

    .theme-control label {

      margin: 0;

    }

    .theme-control select {

      width: auto;

      min-width: 130px;

    }

    .btn {

      border: none;

      background: var(--accent);

      color: #fff;

      padding: 10px 14px;

      border-radius: 12px;

      font-weight: 600;

      cursor: pointer;

      transition: transform 0.12s ease, box-shadow 0.12s ease;

    }

    .btn.secondary {

      background: var(--soft);

      color: var(--ink);

    }

    .btn.ghost {

      background: transparent;

      border: 1px solid var(--border);

      color: var(--ink);

    }

    :root[data-theme="solar"] .chip input:checked + span,
    :root[data-theme="solar"] .mode-toggle button.active,
    :root[data-theme="solar"] .btn {
      box-shadow: 0 0 18px rgba(245, 158, 11, 0.35);
    }

    :root[data-theme="plasma"] .btn,
    :root[data-theme="plasma"] .chip input:checked + span {
      box-shadow: 0 0 20px rgba(34, 211, 238, 0.35), 0 0 32px rgba(168, 85, 247, 0.25);
    }

    :root[data-theme="supernova"] .btn:active {
      box-shadow: 0 0 24px rgba(59, 130, 246, 0.6), 0 0 34px rgba(236, 72, 153, 0.45);
    }

    :root[data-theme="lunar"] .panel {
      backdrop-filter: blur(12px);
    }

    :root[data-theme="meteor"] body {
      transition-duration: 0.15s;
    }

    :root[data-theme="meteor"] .btn {
      transition-duration: 0.08s;
    }

    :root[data-theme="void"] body {
      transition-duration: 0.5s;
    }

    .help-btn {

      width: 36px;

      height: 36px;

      padding: 0;

      border-radius: 999px;

      font-weight: 700;

    }

    .btn:active {

      transform: translateY(1px);

    }

    .gen-btn {

      border: 1px solid var(--border);

      background: var(--surface);

      color: var(--muted);

      border-radius: 8px;

      font-size: 11px;

      padding: 2px 6px;

      margin-left: 6px;

      cursor: pointer;

    }

    .bot-list {

      display: flex;

      flex-direction: column;

      gap: 8px;

      max-height: 300px;

      overflow: auto;

    }

    #bot_search {

      margin-bottom: 10px;

    }

    .bot-card {

      padding: 10px 12px;

      border-radius: 12px;

      border: 1px solid var(--border);

      text-decoration: none;

      color: var(--ink);

      background: var(--surface);

    }

    .bot-card.active {

      border-color: var(--accent);

      box-shadow: 0 6px 14px rgba(15, 118, 110, 0.15);

    }

    .bot-card .title {

      font-weight: 600;

    }

    .bot-card .meta {

      font-size: 12px;

      color: var(--muted);

    }

    .bot-card-images {

      position: relative;

      height: 44px;

      margin-top: 8px;

      perspective: 500px;

      transform-style: preserve-3d;

    }

    .bot-card-image {

      position: absolute;

      width: 62px;

      height: 44px;

      border-radius: 10px;

      border: 1px solid var(--border);

      background: var(--soft);

      background-size: cover;

      background-position: center;

      box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);

      transform:
        translateX(calc(var(--img-index) * 18px))
        translateY(calc(var(--img-index) * -2px))
        rotateY(calc(var(--img-index) * -7deg))
        rotateZ(calc(var(--img-index) * -1deg));

    }

    .bot-card-image.more {

      display: flex;

      align-items: center;

      justify-content: center;

      font-size: 12px;

      font-weight: 700;

      color: var(--ink);

      background: linear-gradient(135deg, rgba(15, 118, 110, 0.2), rgba(249, 115, 22, 0.2));

      backdrop-filter: blur(6px);

    }

    .section-head {

      display: flex;

      align-items: center;

      justify-content: space-between;

      gap: 12px;

      margin-bottom: 10px;

    }

    .modal {

      position: fixed;

      inset: 0;

      display: none;

      align-items: center;

      justify-content: center;

      z-index: 50;

    }

    .modal.open {

      display: flex;

    }

    .modal-backdrop {

      position: absolute;

      inset: 0;

      background: rgba(0, 0, 0, 0.35);

      z-index: 0;

    }

    .modal-card {

      position: relative;

      width: min(94vw, 980px);

      max-height: 85vh;

      overflow: auto;

      background: var(--card);

      border: 1px solid var(--border);

      border-radius: 16px;

      padding: 18px;

      box-shadow: 0 18px 40px var(--shadow);

      z-index: 2;

    }

    .onboarding-modal .modal-backdrop {

      background: rgba(0, 0, 0, 0.25);

    }

    .onboarding-modal {
      --ink: #1f1d1a;
      --muted: #6e6259;
      --card: rgba(255, 255, 255, 0.95);
      --surface: rgba(255, 255, 255, 0.95);
      --soft: #f0e6d9;
      --border: rgba(225, 206, 185, 0.85);
      --shadow: rgba(31, 29, 26, 0.12);
    }
    :root[data-theme="dark"] .onboarding-modal {
      --ink: #f1e7da;
      --muted: #c7b7a8;
      --card: rgba(18, 16, 14, 0.98);
      --surface: rgba(22, 19, 16, 0.98);
      --soft: #252019;
      --border: #332b23;
      --shadow: rgba(0, 0, 0, 0.6);
    }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme]) .onboarding-modal {
        --ink: #f1e7da;
        --muted: #c7b7a8;
        --card: rgba(18, 16, 14, 0.98);
        --surface: rgba(22, 19, 16, 0.98);
        --soft: #252019;
        --border: #332b23;
        --shadow: rgba(0, 0, 0, 0.6);
      }
    }
    .onboarding-modal .section-head h2,
    .onboarding-modal .onboard-hero h2,
    .onboarding-modal .onboard-step-card h3 {
      color: var(--ink);
    }
    .onboarding-modal .onboard-hero p,
    .onboarding-modal .onboard-step-card p {
      color: var(--muted);
    }
    .onboard-card-wrap {
      position: relative;
      z-index: 2;
      transition: transform 0.45s ease;
      will-change: transform;
    }
    .onboarding-card {
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
      gap: 18px;
      padding: 0;
      overflow: hidden;

      transform: translate3d(0, 0, 0);

      animation: onboard-card-drift 9s ease-in-out infinite;

      transition: transform 0.3s ease, box-shadow 0.3s ease;

    }

    .onboarding-card:hover {

      animation-play-state: paused;

      transform: translate3d(0, -3px, 0);

      box-shadow: 0 24px 50px rgba(31, 29, 26, 0.2);

    }

    .onboard-focus {

      position: absolute;

      inset: 0;

      pointer-events: none;

      z-index: 1;

      opacity: 0;

      --focus-x: 50%;

      --focus-y: 50%;

      --focus-r: 180px;

      background: radial-gradient(circle at var(--focus-x) var(--focus-y),

        rgba(255, 255, 255, 0.08) 0,

        rgba(255, 255, 255, 0.02) calc(var(--focus-r) * 0.5),

        rgba(0, 0, 0, 0.35) calc(var(--focus-r) * 1.2));

      transition: opacity 0.3s ease, background 0.3s ease;

    }

    .onboard-focus.active {

      opacity: 1;

    }

    .onboard-focus-box {

      position: absolute;

      border-radius: 16px;

      border: 2px solid rgba(255, 255, 255, 0.9);

      box-shadow: 0 0 0 9999px rgba(5, 8, 10, 0.18), 0 0 24px rgba(15, 118, 110, 0.45);

      transition: width 0.4s ease, height 0.4s ease, top 0.4s ease, left 0.4s ease;

      animation: onboard-pulse 2.6s ease-in-out infinite;

    }

    .onboard-hero {
      position: relative;
      padding: 26px;
      min-height: 320px;
      background: radial-gradient(circle at 15% 20%, rgba(255, 255, 255, 0.4), transparent 60%),
        linear-gradient(135deg, rgba(15, 118, 110, 0.24), rgba(249, 115, 22, 0.2), rgba(15, 118, 110, 0.08));
      background-size: 140% 140%, 200% 200%;

      background-position: 0% 50%, 0% 50%;

      animation: onboard-gradient 14s ease infinite;

      border-right: 1px solid var(--border);

    }

    .onboard-hero::before,

    .onboard-hero::after {

      content: "";

      position: absolute;

      width: 180px;

      height: 180px;

      border-radius: 50%;

      opacity: 0.8;

      animation: onboard-float 10s ease-in-out infinite;

    }

    .onboard-hero::before {

      top: -40px;

      left: -40px;

      background: radial-gradient(circle at 30% 30%, rgba(15, 118, 110, 0.45), transparent 70%);

    }

    .onboard-hero::after {

      bottom: -60px;

      right: -30px;

      background: radial-gradient(circle at 60% 40%, rgba(249, 115, 22, 0.45), transparent 70%);

      animation-delay: -3s;

    }

    .onboard-badge {

      display: inline-flex;

      align-items: center;

      padding: 6px 12px;

      border-radius: 999px;

      background: rgba(255, 255, 255, 0.82);

      border: 1px solid rgba(255, 255, 255, 0.7);

      box-shadow: 0 10px 22px rgba(15, 118, 110, 0.18);

      font-size: 12px;

      font-weight: 600;

      letter-spacing: 0.04em;

      text-transform: uppercase;

      color: var(--accent);

    }

    .onboard-hero h2 {
      margin: 14px 0 8px;
      font-size: 28px;
      line-height: 1.1;
      text-shadow: 0 8px 20px rgba(31, 29, 26, 0.18);
    }
    .onboard-hero p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
      max-width: 360px;
    }
    .onboard-hero-stats {

      display: grid;

      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));

      gap: 12px;

    }

    .onboard-stat {
      position: relative;
      border-radius: 12px;
      padding: 12px 14px 12px 16px;
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.85);

      box-shadow: 0 12px 24px rgba(15, 118, 110, 0.12);

      font-size: 12px;

      overflow: hidden;

    }

    .onboard-stat::before {

      content: "";

      position: absolute;

      left: 0;

      top: 0;

      bottom: 0;

      width: 4px;

      background: linear-gradient(180deg, var(--accent), var(--accent-2));

      opacity: 0.9;

    }

    .onboard-stat-label {

      display: block;

      text-transform: uppercase;

      letter-spacing: 0.08em;

      font-size: 10px;

      color: var(--muted);

      margin-bottom: 4px;

    }

    .onboard-stat-value {

      font-weight: 600;

      color: var(--ink);

      font-size: 13px;

    }

    .onboard-main {
      padding: 22px 22px 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.98));
    }
    .onboard-track-row {

      display: flex;

      align-items: center;

      justify-content: space-between;

      gap: 12px;

      flex-wrap: wrap;

    }

    .onboard-track-toggle button {

      padding: 4px 10px;

      font-size: 12px;

    }

    .onboard-track-toggle {

      background: rgba(255, 255, 255, 0.75);

      border-color: rgba(255, 255, 255, 0.8);

      box-shadow: 0 12px 24px rgba(31, 29, 26, 0.1);

    }

    .onboard-track-toggle button.active {
      box-shadow: 0 10px 18px rgba(15, 118, 110, 0.22);
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-hero {
      background: radial-gradient(circle at 20% 15%, rgba(35, 31, 26, 0.8), transparent 55%),
        linear-gradient(135deg, rgba(16, 44, 40, 0.75), rgba(66, 40, 22, 0.7), rgba(18, 16, 14, 0.85));
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-badge {
      background: rgba(26, 22, 18, 0.9);
      border-color: #3b332a;
      color: #7ddfd4;
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-hero::before {
      background: radial-gradient(circle at 30% 30%, rgba(20, 184, 166, 0.3), transparent 70%);
      opacity: 0.6;
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-hero::after {
      background: radial-gradient(circle at 60% 40%, rgba(251, 146, 60, 0.26), transparent 70%);
      opacity: 0.6;
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-hero h2 {
      text-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-focus {
      background: radial-gradient(circle at var(--focus-x) var(--focus-y),
        rgba(20, 184, 166, 0.18) 0,
        rgba(20, 184, 166, 0.05) calc(var(--focus-r) * 0.5),
        rgba(0, 0, 0, 0.6) calc(var(--focus-r) * 1.2));
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-focus-box {
      border-color: rgba(20, 184, 166, 0.85);
      box-shadow: 0 0 0 9999px rgba(4, 6, 7, 0.35), 0 0 24px rgba(20, 184, 166, 0.4);
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-stat {
      background: rgba(24, 21, 17, 0.92);
      border-color: #3b332a;
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-step-card {
      background: rgba(24, 21, 17, 0.98);
      border-color: #3b332a;
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-main {
      background: linear-gradient(180deg, rgba(22, 20, 17, 0.92), rgba(26, 23, 19, 0.98));
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-track-toggle {
      background: rgba(24, 21, 17, 0.92);
      border-color: #3b332a;
      box-shadow: 0 12px 22px rgba(0, 0, 0, 0.35);
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-track-note {
      background: #231f19;
      border-color: #3b332a;
      color: #c2b4a5;
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-progress {
      background: #2b251f;
    }
    :root[data-theme="dark"] .onboarding-modal .onboard-dot {
      background: #3b332a;
    }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme]) .onboarding-modal .onboard-hero {
        background: radial-gradient(circle at 20% 15%, rgba(35, 31, 26, 0.8), transparent 55%),
          linear-gradient(135deg, rgba(16, 44, 40, 0.75), rgba(66, 40, 22, 0.7), rgba(18, 16, 14, 0.85));
      }
      :root:not([data-theme]) .onboarding-modal .onboard-badge {
        background: rgba(26, 22, 18, 0.9);
        border-color: #3b332a;
        color: #7ddfd4;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
      }
      :root:not([data-theme]) .onboarding-modal .onboard-hero::before {
        background: radial-gradient(circle at 30% 30%, rgba(20, 184, 166, 0.3), transparent 70%);
        opacity: 0.6;
      }
      :root:not([data-theme]) .onboarding-modal .onboard-hero::after {
        background: radial-gradient(circle at 60% 40%, rgba(251, 146, 60, 0.26), transparent 70%);
        opacity: 0.6;
      }
      :root:not([data-theme]) .onboarding-modal .onboard-hero h2 {
        text-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
      }
      :root:not([data-theme]) .onboarding-modal .onboard-focus {
        background: radial-gradient(circle at var(--focus-x) var(--focus-y),
          rgba(20, 184, 166, 0.18) 0,
          rgba(20, 184, 166, 0.05) calc(var(--focus-r) * 0.5),
          rgba(0, 0, 0, 0.6) calc(var(--focus-r) * 1.2));
      }
      :root:not([data-theme]) .onboarding-modal .onboard-focus-box {
        border-color: rgba(20, 184, 166, 0.85);
        box-shadow: 0 0 0 9999px rgba(4, 6, 7, 0.35), 0 0 24px rgba(20, 184, 166, 0.4);
      }
      :root:not([data-theme]) .onboarding-modal .onboard-stat {
        background: rgba(24, 21, 17, 0.92);
        border-color: #3b332a;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
      }
      :root:not([data-theme]) .onboarding-modal .onboard-step-card {
        background: rgba(24, 21, 17, 0.98);
        border-color: #3b332a;
      }
      :root:not([data-theme]) .onboarding-modal .onboard-main {
        background: linear-gradient(180deg, rgba(22, 20, 17, 0.92), rgba(26, 23, 19, 0.98));
      }
      :root:not([data-theme]) .onboarding-modal .onboard-track-toggle {
        background: rgba(24, 21, 17, 0.92);
        border-color: #3b332a;
        box-shadow: 0 12px 22px rgba(0, 0, 0, 0.35);
      }
      :root:not([data-theme]) .onboarding-modal .onboard-track-note {
        background: #231f19;
        border-color: #3b332a;
        color: #c2b4a5;
      }
      :root:not([data-theme]) .onboarding-modal .onboard-progress {
        background: #2b251f;
      }
      :root:not([data-theme]) .onboarding-modal .onboard-dot {
        background: #3b332a;
      }
    }
    .onboard-track-note {

      font-size: 12px;

      color: var(--muted);

      padding: 4px 10px;

      border-radius: 999px;

      background: var(--soft);

      border: 1px solid var(--border);

    }

    .onboard-step-count {

      font-size: 12px;

      letter-spacing: 0.08em;

      text-transform: uppercase;

      color: var(--muted);

    }

    .onboard-step-card {

      padding: 18px;

      border-radius: 14px;

      border: 1px solid var(--border);

      background: var(--surface);

      box-shadow: 0 14px 28px rgba(31, 29, 26, 0.08);

      opacity: 0.98;

      transition: transform 0.3s ease, box-shadow 0.3s ease;

    }

    .onboard-step-card::before {

      content: "";

      display: block;

      height: 3px;

      margin: -18px -18px 14px -18px;

      border-radius: 14px 14px 0 0;

      background: linear-gradient(90deg, var(--accent), var(--accent-2));

      opacity: 0.85;

    }

    .onboard-step-card h3 {

      margin: 0 0 8px;

      font-size: 18px;

    }

    .onboard-step-card.animate {

      animation: onboard-fade 0.5s ease;

    }

    .onboard-step-card:hover {

      transform: translate3d(0, -2px, 0);

      box-shadow: 0 20px 36px rgba(31, 29, 26, 0.12);

    }

    .onboard-step-card p {

      margin: 0 0 16px;

      font-size: 14px;

      color: var(--ink);

    }

    .onboard-progress {

      position: relative;

      height: 6px;

      border-radius: 999px;

      background: var(--soft);

      overflow: hidden;

      margin-bottom: 14px;

    }

    .onboard-progress-bar {

      height: 100%;

      width: 0%;

      background: linear-gradient(90deg, var(--accent), var(--accent-2));

      border-radius: inherit;

      transition: width 0.4s ease;

      background-size: 200% 100%;

      animation: onboard-sheen 3.2s ease infinite;

    }

    .onboard-dots {

      display: flex;

      gap: 8px;

      align-items: center;

    }

    .onboard-dot {

      width: 10px;

      height: 10px;

      border-radius: 999px;

      background: var(--border);

      border: none;

      padding: 0;

      appearance: none;

      cursor: pointer;

      transition: transform 0.2s ease, background 0.2s ease;

    }

    .onboard-dot.active {

      background: var(--accent);

      transform: scale(1.2);

    }

    .onboard-actions {

      display: flex;

      align-items: center;

      justify-content: space-between;

      gap: 12px;

      flex-wrap: wrap;

    }

    @keyframes onboard-float {

      0% { transform: translate3d(0, 0, 0); }

      50% { transform: translate3d(0, 12px, 0); }

      100% { transform: translate3d(0, 0, 0); }

    }

    @keyframes onboard-fade {

      from { opacity: 0; transform: translateY(6px); }

      to { opacity: 1; transform: translateY(0); }

    }

    @keyframes onboard-card-drift {

      0% { transform: translate3d(0, 0, 0); }

      50% { transform: translate3d(0, 6px, 0); }

      100% { transform: translate3d(0, 0, 0); }

    }

    @keyframes onboard-pulse {

      0% { box-shadow: 0 0 0 9999px rgba(5, 8, 10, 0.18), 0 0 18px rgba(15, 118, 110, 0.45); }

      50% { box-shadow: 0 0 0 9999px rgba(5, 8, 10, 0.22), 0 0 32px rgba(15, 118, 110, 0.6); }

      100% { box-shadow: 0 0 0 9999px rgba(5, 8, 10, 0.18), 0 0 18px rgba(15, 118, 110, 0.45); }

    }

    @keyframes onboard-gradient {

      0% { background-position: 0% 50%; }

      50% { background-position: 100% 50%; }

      100% { background-position: 0% 50%; }

    }

    @keyframes onboard-sheen {

      0% { background-position: 0% 50%; }

      100% { background-position: 100% 50%; }

    }

    .help-content h3 {

      margin: 18px 0 8px;

      font-size: 15px;

    }

    .help-content ul {

      margin: 0 0 12px 18px;

      padding: 0;

      color: var(--ink);

      font-size: 13px;

    }

    .help-content li {

      margin-bottom: 6px;

    }

    .help-grid {

      display: grid;

      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);

      gap: 16px;

    }

    .faq-card, .help-chat {

      border: 1px solid var(--border);

      border-radius: 14px;

      padding: 14px;

      background: var(--surface);

      box-shadow: 0 16px 40px rgba(31, 29, 26, 0.08);

    }

    .faq-item + .faq-item {

      margin-top: 10px;

      padding-top: 10px;

      border-top: 1px solid var(--border);

    }

    .faq-item h4 {

      margin: 0 0 6px;

      font-size: 14px;

    }

    .faq-item p {

      margin: 0;

      color: var(--muted);

      font-size: 13px;

    }

    .help-chat {

      display: flex;

      flex-direction: column;

      gap: 10px;

    }

    .help-chat-log {

      border: 1px solid var(--border);

      border-radius: 12px;

      padding: 10px;

      background: var(--soft);

      max-height: 260px;

      overflow: auto;

      display: flex;

      flex-direction: column;

      gap: 8px;

    }

    .help-chat-bubble {

      padding: 8px 10px;

      border-radius: 12px;

      font-size: 13px;

      line-height: 1.4;

      white-space: pre-wrap;

    }

    .help-chat-bubble.user {

      align-self: flex-end;

      background: var(--user-bubble);

      border: 1px solid var(--user-border);

    }

    .help-chat-bubble.bot {

      align-self: flex-start;

      background: var(--bot-bubble);

      border: 1px solid var(--bot-border);

    }

    .help-chat-controls {

      display: flex;

      gap: 8px;

      flex-wrap: wrap;

      align-items: center;

    }

    .help-chat-input {

      width: 100%;

      min-height: 80px;

    }

    .help-mode-toggle {

      display: inline-flex;

      align-items: center;

      gap: 4px;

      padding: 4px;

      border-radius: 999px;

      border: 1px solid var(--border);

      background: var(--soft);

    }

    .help-mode-toggle button {

      border: none;

      background: transparent;

      color: var(--muted);

      padding: 6px 12px;

      border-radius: 999px;

      font-weight: 600;

      cursor: pointer;

    }

    .help-mode-toggle button.active {

      background: var(--accent-2);

      color: #fff;

    }

    .help-chat select, .help-chat textarea, .help-chat input[type="text"] {

      width: 100%;

    }

    @media (max-width: 1100px) {

      .help-grid {

        grid-template-columns: 1fr;

      }

    }

    section.panel {

      scroll-margin-top: 90px;

    }

    .chips {

      display: flex;

      flex-wrap: wrap;

      gap: 8px;

    }

    .tag-editor {

      display: flex;

      flex-direction: column;

      gap: 10px;

    }

    .tag-chip {

      display: inline-flex;

      align-items: center;

      gap: 6px;

      padding: 6px 10px;

      border-radius: 999px;

      border: 1px solid var(--border);

      background: var(--chip-bg);

      font-size: 13px;

    }

    .tag-chip button {

      border: none;

      background: transparent;

      color: var(--muted);

      cursor: pointer;

      font-weight: 700;

    }

    .chip {

      display: inline-flex;

      align-items: center;

    }

    .chip input {

      display: none;

    }

    .chip span {

      padding: 6px 12px;

      border-radius: 999px;

      border: 1px solid var(--border);

      background: var(--chip-bg);

      font-size: 13px;

      cursor: pointer;

    }

    .chip input:checked + span {

      background: var(--accent);

      color: #fff;

      border-color: var(--accent);

    }

    .image-grid {

      display: grid;

      grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));

      gap: 8px;

    }

    .image-card {

      position: relative;

      border-radius: 10px;

      overflow: hidden;

      border: 1px solid var(--border);

      background: var(--surface);

    }

    .image-card img {

      width: 100%;

      height: 90px;

      object-fit: cover;

      display: block;

    }

    .image-card button {

      position: absolute;

      top: 6px;

      right: 6px;

      background: rgba(31, 29, 26, 0.75);

      color: #fff;

      border: none;

      border-radius: 8px;

      padding: 2px 6px;

      cursor: pointer;

    }

    .image-viewer-card {

      width: min(92vw, 980px);

      padding: 16px;

      display: flex;

      flex-direction: column;

      gap: 12px;

    }

    .image-viewer-header {

      display: flex;

      align-items: center;

      justify-content: space-between;

      gap: 12px;

    }

    .image-viewer-actions {

      display: flex;

      align-items: center;

      gap: 8px;

    }

    .image-viewer-actions .btn {

      padding: 6px 10px;

      font-size: 12px;

    }

    .image-viewer-counter {

      font-size: 12px;

      color: var(--muted);

    }

    .image-viewer-header .meta {

      font-size: 12px;

      color: var(--muted);

    }

    .image-viewer-frame {

      border-radius: 14px;

      border: 1px solid var(--border);

      background: var(--soft);

      padding: 10px;

    }

    .image-viewer-frame img {

      width: 100%;

      max-height: 70vh;

      object-fit: contain;

      display: block;

      border-radius: 10px;

      background: var(--surface);

    }

    .list-stack {

      display: flex;

      flex-direction: column;

      gap: 10px;

    }

    .list-item {

      display: grid;

      grid-template-columns: auto auto 1fr auto;

      gap: 8px;

      align-items: start;

    }

    .dialogue-row, .pair-row {

      display: grid;

      grid-template-columns: auto auto 1fr 1fr auto;

      gap: 8px;

      align-items: start;

    }

    .lore-row {

      display: grid;

      grid-template-columns: auto auto 1fr 2fr auto auto;

      gap: 8px;

      align-items: start;

    }

    .status {

      padding: 10px 12px;

      border-radius: 12px;

      background: var(--surface);

      border: 1px solid var(--border);

      font-size: 13px;

      color: var(--muted);

    }

    .status-stack {

      display: flex;

      flex-direction: column;

      gap: 10px;

    }

    .progress {

      margin-top: 10px;

      height: 8px;

      background: var(--soft);

      border-radius: 999px;

      overflow: hidden;

      border: 1px solid var(--border);

    }

    .progress-bar {

      height: 100%;

      width: 0%;

      background: var(--accent);

      transition: width 0.2s ease;

    }

    .status.ok {

      border-color: var(--ok-border);

      color: var(--ok-ink);

      background: var(--ok-bg);

    }

    .status.error {

      border-color: var(--error-border);

      color: var(--error-ink);

      background: var(--error-bg);

    }

    .chat-log {

      display: flex;

      flex-direction: column;

      gap: 10px;

    }

    .chat-bubble {

      padding: 10px 12px;

      border-radius: 12px;

      background: var(--surface);

      border: 1px solid var(--border);

      font-size: 14px;

    }

    .chat-bubble.user {

      border-color: var(--user-border);

      background: var(--user-bubble);

      align-self: flex-end;

    }

    .chat-bubble.bot {

      border-color: var(--bot-border);

      background: var(--bot-bubble);

    }

    .preview {

      background: var(--surface);

      border: 1px solid var(--border);

      border-radius: 12px;

      padding: 12px;

      max-height: 280px;

      overflow: auto;

      font-size: 12px;

      white-space: pre-wrap;

    }

    .token-counter {

      margin-top: 6px;

      font-size: 12px;

      color: var(--muted);

    }

    .token-counter.caution {

      color: #a16207;

    }

    .token-counter.warn {

      color: #b45309;

    }

    .token-counter.danger {

      color: #b91c1c;

    }

    .list-tools {

      display: flex;

      align-items: center;

      gap: 8px;

      margin-top: 8px;

    }

    .drop-zone {

      border: 1px dashed var(--border);

      border-radius: 12px;

      padding: 14px;

      text-align: center;

      color: var(--muted);

      background: var(--surface);

      cursor: pointer;

    }

    .drop-zone.drag {

      border-color: var(--accent);

      color: var(--accent);

      background: var(--soft);

    }

    .drag-handle {

      cursor: grab;

      user-select: none;

      padding: 4px 6px;

      border: 1px dashed var(--border);

      border-radius: 8px;

      color: var(--muted);

      font-size: 11px;

    }

    @media (max-width: 980px) {

      .layout {

        grid-template-columns: 1fr;

      }

      .sidebar {

        position: static;

      }

    }

    @media (max-width: 900px) {

      .onboarding-card {

        grid-template-columns: 1fr;

      }

      .onboard-hero {

        border-right: none;

        border-bottom: 1px solid var(--border);

      }

    }

    @media (prefers-reduced-motion: reduce) {

      .onboarding-card,

      .onboard-hero,

      .onboard-hero::before,

      .onboard-hero::after,

      .onboard-focus-box,

      .onboard-progress-bar {

        animation: none !important;

      }

      body::before {
        animation: none !important;
      }

    }

  </style>

</head>

<body>

  <header>

    <div>

      <div class="brand">{{ app_title }}</div>

      <div class="sub">Build, tune, and store RP bots with image-aware generation.</div>

    </div>

    <div class="row">

      <div class="theme-control">

        <label for="theme_select">Theme</label>

        <select id="theme_select">

          <option value="system">System</option>

          <option value="light">Light</option>

          <option value="dark">Dark</option>

          <option value="stellar">Stellar</option>
          <option value="nebula">Nebula</option>
          <option value="solar">Solar</option>
          <option value="lunar">Lunar</option>
          <option value="eclipse">Eclipse</option>
          <option value="aurora">Aurora</option>
          <option value="cosmos">Cosmos</option>
          <option value="supernova">Supernova</option>
          <option value="void">Void</option>
          <option value="orbit">Orbit</option>
          <option value="plasma">Plasma</option>
          <option value="meteor">Meteor</option>

        </select>

      </div>

      <div class="mode-toggle" id="mode_toggle">

        <button type="button" data-mode="simple">Simple</button>

        <button type="button" data-mode="advanced">Advanced</button>

      </div>

      <button class="btn ghost" type="button" onclick="toggleOnboarding(true)" title="Guide">Guide</button>

      <button class="btn ghost help-btn" type="button" onclick="toggleHelp(true)" title="Help">?</button>

      <button class="btn" type="button" onclick="saveBot()">Save</button>

      <form method="post" action="{{ url_for('create_bot') }}">

        <button class="btn secondary" type="submit">New Bot</button>

      </form>

    </div>

  </header>



  <div class="layout">

    <aside class="sidebar">

      <div class="panel" id="library-panel">

        <h3>Library</h3>

        <input type="text" id="bot_search" placeholder="Search bots..." />

        <div class="bot-list" id="bot_list">

          {% for item in bots %}

          <a class="bot-card {% if item.id == bot.id %}active{% endif %}"

             data-search="{{ (item.fields.name ~ ' ' ~ item.fields.alias ~ ' ' ~ item.fields.species ~ ' ' ~ (item.tags | join(' ')) ) | lower }}"
             data-title="{{ item.fields.name or 'Untitled Bot' }}"
             data-images="{{ item.images | tojson | e }}"

             href="{{ url_for('edit_bot', bot_id=item.id) }}">

            <div class="title">{{ item.fields.name or 'Untitled Bot' }}</div>

            <div class="meta">{{ item.fields.species or 'Unknown' }} - {{ item.fields.age or 'Age?' }}</div>

            {% if item.images %}

            <div class="bot-card-images" aria-hidden="true">

              {% for img in item.images[:4] %}

              <div class="bot-card-image" style="--img-index: {{ loop.index0 }}; z-index: {{ 10 - loop.index0 }}; background-image: url('/images/{{ img }}');"></div>

              {% endfor %}

              {% if item.images|length > 4 %}

              {% set extra = item.images|length - 4 %}

              <div class="bot-card-image more" style="--img-index: 3; z-index: 6;">+{{ extra }}</div>

              {% endif %}

            </div>

            {% endif %}

          </a>

          {% endfor %}

        </div>

      </div>



      <div class="panel">

        <h3>Status</h3>

        <div class="status-stack">

          <div id="status" class="status">Ready.</div>

          <div id="save_status" class="status">Autosave ready.</div>

        </div>

      </div>



      <div class="panel">

        <h3>Jump To</h3>

        <div class="nav-list">

          <a href="#simple-generator">Simple Generator</a>

          <a class="advanced-only" href="#generation-controls">Generation Controls</a>

          <a href="#core-identity">Core Identity</a>

          <a class="advanced-only" href="#appearance">Appearance</a>

          <a class="advanced-only" href="#voice-style">Voice and Style</a>

          <a class="advanced-only" href="#world-motivation">World and Motivation</a>

          <a class="advanced-only" href="#traits-limits">Traits and Limits</a>

          <a class="advanced-only" href="#system-guidance">System Guidance</a>

          <a class="advanced-only" href="#roleplay-controls">Roleplay Controls</a>

          <a class="advanced-only" href="#tone-toggles">Tone Toggles</a>

          <a class="advanced-only" href="#presets">Presets</a>

          <a class="advanced-only" href="#history">History</a>

          <a href="#first-messages">First Messages</a>

          <a href="#scenarios">Scenarios</a>

          <a href="#example-dialogues">Example Dialogues</a>

          <a class="advanced-only" href="#prompt-pairs">Prompt and Response</a>

          <a class="advanced-only" href="#memory-anchors">Memory Anchors</a>

          <a class="advanced-only" href="#lorebook">Lorebook</a>

          <a class="advanced-only" href="#metadata">Metadata</a>

          <a class="advanced-only" href="#import">Import</a>

          <a class="advanced-only" href="#exports">Exports</a>

          <a class="advanced-only" href="#ai-notes">AI Notes</a>

          <a href="#test-chat">Test Chat</a>

          <a href="#compile">Compile</a>

          <a class="advanced-only" href="#utilities">Utilities</a>

        </div>

      </div>



      <div class="panel" id="ai-settings">

        <h3>AI Settings</h3>

        <div class="list-stack">

          <div>

            <label>Provider</label>

            <select id="ai_provider">

              {% for provider in providers %}

              <option value="{{ provider.id }}" {% if settings.provider == provider.id %}selected{% endif %}>{{ provider.label }}</option>

              {% endfor %}

            </select>

          </div>

          <div>

            <label>Model</label>

            <input type="text" id="ai_model" value="{{ settings.model }}" />

          </div>

          <div class="row">

            <button class="btn ghost" type="button" onclick="applyProviderDefaults(true)">Use Provider Defaults</button>

          </div>

          <div>

            <label>Use Images</label>

            <select id="ai_use_images">

              <option value="true" {% if settings.use_images %}selected{% endif %}>Enabled</option>

              <option value="false" {% if not settings.use_images %}selected{% endif %}>Disabled</option>

            </select>

          </div>

          <div>

            <label>API Key</label>

            <input type="text" id="ai_key" value="{{ settings.api_key }}" placeholder="sk-..." />

          </div>

          <div>

            <label>Default Creator</label>

            <input type="text" id="creator_name" value="{{ settings.creator_name }}" placeholder="Your handle" />

          </div>

          <div>

            <label>Base URL</label>

            <input type="text" id="ai_base_url" value="{{ settings.base_url }}" />

          </div>

          <div>

            <label>Temperature</label>

            <input type="number" id="ai_temp" min="0" max="2" step="0.1" value="{{ settings.temperature }}" />

          </div>

          <div>

            <label>Max Tokens</label>

            <input type="number" id="ai_tokens" min="128" max="4096" step="64" value="{{ settings.max_tokens }}" />

          </div>

          <div>

            <label>Autosave Seconds</label>

            <input type="number" id="autosave_seconds" min="10" max="300" step="5" value="{{ settings.autosave_seconds }}" />

          </div>

          <button class="btn secondary" type="button" onclick="saveSettings()">Save Settings</button>

        </div>

      </div>



      <div class="panel" id="images-panel">

        <h3>Images</h3>

        <div class="list-stack">

          <input type="file" id="image_upload" multiple accept="image/*" />

          <div id="image_grid" class="image-grid"></div>

          <div class="hint">Upload 1-{{ max_images }} reference images.</div>

        </div>

      </div>

    </aside>



    <main class="main">

      <section class="panel" id="simple-generator">

        <div class="section-head">

          <h2>Simple Generator</h2>

          <div class="row">

            <button class="btn" type="button" onclick="generateSimple()">Generate Simple</button>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Original Input</label>

            <textarea data-field="simple_input" id="simple_input" placeholder="Describe the character, vibe, and situation.">{{ bot.fields.simple_input }}</textarea>

            <div class="hint">Generates description, personality, scenario, first message, and example dialogues from this input and any images.</div>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Current Name</label>

            <input data-field="simple_name" id="simple_name" type="text" value="{{ bot.fields.simple_name }}" placeholder="Optional" />

          </div>

          <div>

            <label>Current Age</label>

            <input data-field="simple_age" id="simple_age" type="text" value="{{ bot.fields.simple_age }}" placeholder="Optional" />

          </div>

          <div>

            <label>Current Species</label>

            <input data-field="simple_species" id="simple_species" type="text" value="{{ bot.fields.simple_species }}" placeholder="Optional" />

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Current First Messages</label>

            <textarea data-field="simple_current_first_messages" placeholder="Optional. One per line.">{{ bot.fields.simple_current_first_messages }}</textarea>

          </div>

          <div>

            <label>Current Scenarios</label>

            <textarea data-field="simple_current_scenarios" placeholder="Optional. One per line.">{{ bot.fields.simple_current_scenarios }}</textarea>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Current Example Dialogues</label>

            <textarea data-field="simple_current_dialogues" placeholder="{{ '{{user}}' }}: ...&#10;{{ '{{char}}' }}: ...">{{ bot.fields.simple_current_dialogues }}</textarea>

            <div class="hint">Paste existing dialogue lines to steer new outputs.</div>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="generation-controls">

        <div class="section-head">

          <h2>Generation Controls</h2>

          <div class="row">

            <button class="btn secondary" type="button" onclick="setAllGenSections(true)">Enable All</button>

            <button class="btn ghost" type="button" onclick="setAllGenSections(false)">Disable All</button>

          </div>

        </div>

        <div class="chips">

          {% for section in gen_sections %}

          <label class="chip" title="{{ section.hint }}">

            <input type="checkbox" name="gen_section" value="{{ section.id }}" {% if bot.gen_sections.get(section.id, True) %}checked{% endif %} />

            <span>{{ section.label }}</span>

          </label>

          {% endfor %}

        </div>

        <div class="row" style="margin-top: 10px;">

          <button class="btn ghost" type="button" onclick="setMergePreset('fill')">Fill Empty</button>

          <button class="btn ghost" type="button" onclick="setMergePreset('append')">Append</button>

          <button class="btn ghost" type="button" onclick="setMergePreset('overwrite')">Overwrite</button>

          <button class="btn secondary" type="button" onclick="freshStart()">Fresh Start</button>

        </div>

        <div class="grid" style="margin-top: 10px;">

          <div>

            <label>Greetings Count</label>

            <input type="number" id="gen_greetings_count" min="0" max="10" value="3" />

          </div>

          <div>

            <label>Scenarios Count</label>

            <input type="number" id="gen_scenarios_count" min="0" max="12" value="5" />

          </div>

          <div>

            <label>Dialogues Count</label>

            <input type="number" id="gen_dialogues_count" min="0" max="10" value="4" />

          </div>

        </div>

        <div class="row" style="margin-top: 10px;">

          <button class="btn" type="button" onclick="generateAll()">Generate All</button>

          <button class="btn ghost" type="button" onclick="generateEmptyOnly()">Empty Only</button>

          <button class="btn ghost" type="button" onclick="randomInspiration()">Random Inspiration</button>

          <button class="btn secondary" type="button" onclick="aiDecideEverything()">AI Decide Everything</button>

        </div>

        <div class="progress">

          <div id="gen_progress" class="progress-bar"></div>

        </div>

        <div class="hint">Controls what the AI can change when generating the profile.</div>

      </section>



      <section class="panel advanced-only" id="preview">

        <div class="section-head">

          <h2>Live Preview</h2>

          <div class="row">

            <button class="btn ghost" type="button" onclick="setPreviewMode('card')">Chub v2</button>

            <button class="btn ghost" type="button" onclick="setPreviewMode('janitor')">Janitor</button>

            <button class="btn ghost" type="button" onclick="setPreviewMode('system')">System</button>

          </div>

        </div>

        <pre id="preview_output" class="preview"></pre>

        <div id="token_total" class="token-counter"></div>

      </section>



      <section class="panel" id="core-identity">

        <div class="section-head">

          <h2>Core Identity</h2>

          <div class="row advanced-only">

            <select id="profile_merge">

              <option value="fill">Fill Empty</option>

              <option value="append">Append</option>

              <option value="overwrite">Overwrite</option>

            </select>

            <button class="btn" type="button" onclick="generateProfile()">Generate Profile</button>

          </div>

        </div>

        <div class="grid advanced-only">

          <div>

            <label>Name</label>

            <input data-field="name" type="text" value="{{ bot.fields.name }}" />

          </div>

          <div>

            <label>Alias</label>

            <input data-field="alias" type="text" value="{{ bot.fields.alias }}" />

          </div>

          <div>

            <label>Age</label>

            <input data-field="age" type="text" value="{{ bot.fields.age }}" />

          </div>

          <div>

            <label>Species</label>

            <input data-field="species" type="text" value="{{ bot.fields.species }}" />

          </div>

          <div>

            <label>Gender</label>

            <input data-field="gender" type="text" value="{{ bot.fields.gender }}" />

          </div>

          <div>

            <label>Pronouns</label>

            <input data-field="pronouns" type="text" value="{{ bot.fields.pronouns }}" />

          </div>

          <div>

            <label>Occupation</label>

            <input data-field="occupation" type="text" value="{{ bot.fields.occupation }}" />

          </div>

          <div>

            <label>Relationship to User</label>

            <input data-field="relationship" type="text" value="{{ bot.fields.relationship }}" />

          </div>

        </div>

        <div class="grid">

          <div class="advanced-only">

            <label>Description</label>

            <textarea data-field="description">{{ bot.fields.description }}</textarea>

          </div>

          <div>

            <label>Personality</label>

            <textarea data-field="personality">{{ bot.fields.personality }}</textarea>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="appearance">

        <h2>Appearance</h2>

        <div class="grid">

          <div>

            <label>Appearance Summary</label>

            <textarea data-field="appearance">{{ bot.fields.appearance }}</textarea>

          </div>

          <div>

            <label>Distinguishing Features</label>

            <textarea data-field="distinguishing_features">{{ bot.fields.distinguishing_features }}</textarea>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Height</label>

            <input data-field="height" type="text" value="{{ bot.fields.height }}" />

          </div>

          <div>

            <label>Body Type</label>

            <input data-field="body_type" type="text" value="{{ bot.fields.body_type }}" />

          </div>

          <div>

            <label>Outfit / Style</label>

            <input data-field="outfit" type="text" value="{{ bot.fields.outfit }}" />

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="voice-style">

        <h2>Voice and Style</h2>

        <div class="grid">

          <div>

            <label>Voice</label>

            <textarea data-field="voice">{{ bot.fields.voice }}</textarea>

          </div>

          <div>

            <label>Speech Style</label>

            <textarea data-field="speech_style">{{ bot.fields.speech_style }}</textarea>

          </div>

          <div>

            <label>Mannerisms</label>

            <textarea data-field="mannerisms">{{ bot.fields.mannerisms }}</textarea>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Catchphrases</label>

            <input data-field="catchphrases" type="text" value="{{ bot.fields.catchphrases }}" />

          </div>

          <div>

            <label>Formatting</label>

            <input data-field="formatting" type="text" value="{{ bot.fields.formatting }}" placeholder="Use *italics* for actions" />

          </div>

          <div>

            <label>Style Rules</label>

            <input data-field="style_rules" type="text" value="{{ bot.fields.style_rules }}" placeholder="No out-of-character text" />

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="world-motivation">

        <h2>World and Motivation</h2>

        <div class="grid">

          <div>

            <label>Setting</label>

            <input data-field="setting" type="text" value="{{ bot.fields.setting }}" />

          </div>

          <div>

            <label>Current Scenario</label>

            <input data-field="current_scenario" type="text" value="{{ bot.fields.current_scenario }}" />

          </div>

          <div>

            <label>Backstory</label>

            <textarea data-field="backstory">{{ bot.fields.backstory }}</textarea>

          </div>

          <div>

            <label>Goals</label>

            <textarea data-field="goals">{{ bot.fields.goals }}</textarea>

          </div>

          <div>

            <label>Motivations</label>

            <textarea data-field="motivations">{{ bot.fields.motivations }}</textarea>

          </div>

          <div>

            <label>Values</label>

            <textarea data-field="values">{{ bot.fields.values }}</textarea>

          </div>

          <div>

            <label>World Lore</label>

            <textarea data-field="world_lore">{{ bot.fields.world_lore }}</textarea>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="traits-limits">

        <h2>Traits and Limits</h2>

        <div class="grid">

          <div>

            <label>Likes</label>

            <textarea data-field="likes">{{ bot.fields.likes }}</textarea>

          </div>

          <div>

            <label>Dislikes</label>

            <textarea data-field="dislikes">{{ bot.fields.dislikes }}</textarea>

          </div>

          <div>

            <label>Skills</label>

            <textarea data-field="skills">{{ bot.fields.skills }}</textarea>

          </div>

          <div>

            <label>Weaknesses</label>

            <textarea data-field="weaknesses">{{ bot.fields.weaknesses }}</textarea>

          </div>

          <div>

            <label>Flaws</label>

            <textarea data-field="flaws">{{ bot.fields.flaws }}</textarea>

          </div>

          <div>

            <label>Fears</label>

            <textarea data-field="fears">{{ bot.fields.fears }}</textarea>

          </div>

          <div>

            <label>Secrets</label>

            <textarea data-field="secrets">{{ bot.fields.secrets }}</textarea>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Consent Rules</label>

            <textarea data-field="consent_rules">{{ bot.fields.consent_rules }}</textarea>

          </div>

          <div>

            <label>Boundaries</label>

            <textarea data-field="boundaries">{{ bot.fields.boundaries }}</textarea>

          </div>

          <div>

            <label>Limits</label>

            <textarea data-field="limits">{{ bot.fields.limits }}</textarea>

          </div>

          <div>

            <label>Kinks</label>

            <textarea data-field="kinks">{{ bot.fields.kinks }}</textarea>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="system-guidance">

        <h2>System Guidance</h2>

        <div class="grid">

          <div>

            <label>System Prompt</label>

            <textarea data-field="system_prompt" placeholder="High-level instructions that always apply.">{{ bot.fields.system_prompt }}</textarea>

          </div>

          <div>

            <label>Post History Instructions</label>

            <textarea data-field="post_history_instructions" placeholder="Guidance applied after conversation history.">{{ bot.fields.post_history_instructions }}</textarea>

          </div>

          <div>

            <label>Rules</label>

            <textarea data-field="rules" placeholder="Behavior rules, do/don't lists, or roleplay guardrails.">{{ bot.fields.rules }}</textarea>

            <div class="hint">Placeholders: {{ '{{char}}' }} for the bot, {{ '{{user}}' }} for the user.</div>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="roleplay-controls">

        <h2>Roleplay Controls</h2>

        <div class="grid">

          <div>

            <label>User Role</label>

            <input data-field="user_role" type="text" value="{{ bot.fields.user_role }}" />

          </div>

          <div>

            <label>Bot Role</label>

            <input data-field="bot_role" type="text" value="{{ bot.fields.bot_role }}" />

          </div>

          <div>

            <label>Response Length</label>

            <select data-field="response_length">

              {% for opt in ['Short', 'Medium', 'Long', 'Very Long'] %}

              <option value="{{ opt }}" {% if bot.fields.response_length == opt %}selected{% endif %}>{{ opt }}</option>

              {% endfor %}

            </select>

          </div>

          <div>

            <label>POV</label>

            <select data-field="pov">

              {% for opt in ['First person', 'Second person', 'Third person'] %}

              <option value="{{ opt }}" {% if bot.fields.pov == opt %}selected{% endif %}>{{ opt }}</option>

              {% endfor %}

            </select>

          </div>

          <div>

            <label>Narration Style</label>

            <select data-field="narration_style">

              {% for opt in ['Dialogue only', 'Actions + Dialogue', 'Mixed'] %}

              <option value="{{ opt }}" {% if bot.fields.narration_style == opt %}selected{% endif %}>{{ opt }}</option>

              {% endfor %}

            </select>

          </div>

          <div>

            <label>Emoji Use</label>

            <select data-field="emoji_use">

              {% for opt in ['None', 'Light', 'Heavy'] %}

              <option value="{{ opt }}" {% if bot.fields.emoji_use == opt %}selected{% endif %}>{{ opt }}</option>

              {% endfor %}

            </select>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="tone-toggles">

        <h2>Tone Toggles</h2>

        <div class="chips">

          {% for toggle in toggles %}

          <label class="chip">

            <input type="checkbox" name="toggle" value="{{ toggle }}" {% if toggle in bot.toggles %}checked{% endif %} />

            <span>{{ toggle }}</span>

          </label>

          {% endfor %}

        </div>

      </section>



      <section class="panel advanced-only" id="presets">

        <h2>Presets</h2>

        <div class="grid">

          <div>

            <label>Preset Name</label>

            <input type="text" id="preset_name" placeholder="e.g. Soft romance" />

          </div>

          <div>

            <label>Saved Presets</label>

            <select id="preset_select"></select>

          </div>

        </div>

        <div class="row">

          <button class="btn secondary" type="button" onclick="savePreset()">Save Preset</button>

          <button class="btn ghost" type="button" onclick="loadPreset()">Load Preset</button>

          <button class="btn ghost" type="button" onclick="deletePreset()">Delete Preset</button>

        </div>

      </section>



      <section class="panel advanced-only" id="history">

        <h2>Version History</h2>

        <div class="row">

          <select id="history_select"></select>

          <button class="btn secondary" type="button" onclick="restoreHistory()">Restore</button>

          <button class="btn ghost" type="button" onclick="deleteHistory()">Delete</button>

          <button class="btn ghost" type="button" onclick="clearHistory()">Clear All</button>

        </div>

        <div id="history_meta" class="hint"></div>

      </section>



      <section class="panel" id="compile">

        <div class="section-head">

          <h2>Compile</h2>

          <button class="btn" type="button" onclick="compileBot()">Compile</button>

        </div>

        <div class="row">

          <label class="chip">

            <input type="checkbox" id="quality_mode" />

            <span>High quality mode</span>

          </label>

          <label class="chip">

            <input type="checkbox" id="allow_emojis" />

            <span>Allow emojis</span>

          </label>

        </div>

        <div class="token-targets">

          <div class="token-target">

            <label>Description Min Tokens</label>

            <input type="hidden" data-field="min_tokens_description" id="min_tokens_description" value="{{ bot.fields.min_tokens_description or 'auto' }}" />

            <div class="mode-toggle token-toggle" data-token-target="min_tokens_description">

              <button type="button" data-value="auto">Auto</button>

              <button type="button" data-value="250">Low</button>

              <button type="button" data-value="500">Medium</button>

              <button type="button" data-value="750">High</button>

              <button type="button" data-value="1000">Very High</button>

              <button type="button" data-value="1500">Extreme</button>

            </div>

          </div>

          <div class="token-target">

            <label>First Messages Min Tokens (total)</label>

            <input type="hidden" data-field="min_tokens_first_messages" id="min_tokens_first_messages" value="{{ bot.fields.min_tokens_first_messages or 'auto' }}" />

            <div class="mode-toggle token-toggle" data-token-target="min_tokens_first_messages">

              <button type="button" data-value="auto">Auto</button>

              <button type="button" data-value="250">Low</button>

              <button type="button" data-value="500">Medium</button>

              <button type="button" data-value="750">High</button>

              <button type="button" data-value="1000">Very High</button>

              <button type="button" data-value="1500">Extreme</button>

            </div>

          </div>

          <div class="token-target">

            <label>Scenario Min Tokens</label>

            <input type="hidden" data-field="min_tokens_scenario" id="min_tokens_scenario" value="{{ bot.fields.min_tokens_scenario or 'auto' }}" />

            <div class="mode-toggle token-toggle" data-token-target="min_tokens_scenario">

              <button type="button" data-value="auto">Auto</button>

              <button type="button" data-value="250">Low</button>

              <button type="button" data-value="500">Medium</button>

              <button type="button" data-value="750">High</button>

              <button type="button" data-value="1000">Very High</button>

              <button type="button" data-value="1500">Extreme</button>

            </div>

          </div>

          <div class="token-target">

            <label>Example Dialogues Min Tokens (total)</label>

            <input type="hidden" data-field="min_tokens_dialogues" id="min_tokens_dialogues" value="{{ bot.fields.min_tokens_dialogues or 'auto' }}" />

            <div class="mode-toggle token-toggle" data-token-target="min_tokens_dialogues">

              <button type="button" data-value="auto">Auto</button>

              <button type="button" data-value="250">Low</button>

              <button type="button" data-value="500">Medium</button>

              <button type="button" data-value="750">High</button>

              <button type="button" data-value="1000">Very High</button>

              <button type="button" data-value="1500">Extreme</button>

            </div>

          </div>

        </div>

        <div class="hint">Generates description, first messages, scenario, and example dialogues in one high-quality pass. Auto keeps the default length guidance. Minimums are totals per section. Outputs appear below.</div>

      </section>



      <section class="panel" id="compile-output">

        <div class="section-head">

          <h2>Compiled Outputs</h2>

          <div class="row">

            <button class="btn secondary" type="button" onclick="applyCompileOutputs()">Apply Outputs</button>

            <button class="btn ghost" type="button" onclick="copyCompileOutputs()">Copy All</button>

            <button class="btn ghost" type="button" onclick="copyCompileJson()">Copy JSON</button>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Description</label>

            <textarea id="compile_description_output" readonly></textarea>

          </div>

          <div>

            <label>Scenario</label>

            <textarea id="compile_scenario_output" readonly></textarea>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>First Messages</label>

            <textarea id="compile_first_messages_output" readonly></textarea>

          </div>

          <div>

            <label>Example Dialogues</label>

            <textarea id="compile_dialogues_output" readonly></textarea>

          </div>

        </div>

        <div class="hint">These outputs do not change the bot until you apply them.</div>

      </section>



      <section class="panel" id="first-messages">

        <h2>First Messages</h2>

        <div class="grid">

          <div>

            <label>Primary Greeting</label>

            <textarea data-field="primary_first_message" id="primary_first_message" placeholder="Used for exports.">{{ bot.fields.primary_first_message }}</textarea>

            <select id="primary_greeting_select"></select>

          </div>

          <div>

            <label>Greeting Helper</label>

            <div class="hint">If empty, the first message in the list is used for exports.</div>

            <button class="btn ghost" type="button" onclick="useFirstMessageAsPrimary()">Use First Message</button>

          </div>

        </div>

        <div class="row">

          <input type="number" id="first_count" min="1" max="10" value="3" />

          <select id="first_merge">

            <option value="append">Append</option>

            <option value="overwrite">Overwrite</option>

          </select>

          <button class="btn" type="button" onclick="generateFirstMessages()">Generate First Messages</button>

        </div>

        <div class="hint">List items become alternate greetings for exports.</div>

        <div id="first_messages" class="list-stack"></div>

        <div id="first_messages_counter" class="token-counter"></div>

        <div class="list-tools">

          <button class="btn ghost" type="button" onclick="removeSelected('first_messages')">Remove Selected</button>

        </div>

        <button class="btn secondary" type="button" onclick="addFirstMessage()">Add Message</button>

      </section>



      <section class="panel" id="scenarios">

        <h2>Scenarios</h2>

        <div class="row">

          <input type="number" id="scenario_count" min="1" max="12" value="5" />

          <select id="scenario_merge">

            <option value="append">Append</option>

            <option value="overwrite">Overwrite</option>

          </select>

          <button class="btn" type="button" onclick="generateScenarios()">Generate Scenarios</button>

        </div>

        <div id="scenario_list" class="list-stack"></div>

        <div id="scenario_counter" class="token-counter"></div>

        <div class="list-tools">

          <button class="btn ghost" type="button" onclick="removeSelected('scenario_list')">Remove Selected</button>

        </div>

        <button class="btn secondary" type="button" onclick="addScenario()">Add Scenario</button>

      </section>



      <section class="panel" id="example-dialogues">

        <h2>Example Dialogues</h2>

        <div class="row">

          <input type="number" id="dialogue_count" min="1" max="10" value="4" />

          <select id="dialogue_merge">

            <option value="append">Append</option>

            <option value="overwrite">Overwrite</option>

          </select>

          <button class="btn" type="button" onclick="generateDialogues()">Generate Dialogues</button>

        </div>

        <div id="dialogue_list" class="list-stack"></div>

        <div id="dialogue_counter" class="token-counter"></div>

        <div class="list-tools">

          <button class="btn ghost" type="button" onclick="removeSelected('dialogue_list')">Remove Selected</button>

        </div>

        <button class="btn secondary" type="button" onclick="addDialogue()">Add Dialogue</button>

      </section>



      <section class="panel advanced-only" id="prompt-pairs">

        <h2>Prompt / Response Pairs</h2>

        <div id="pair_list" class="list-stack"></div>

        <div class="list-tools">

          <button class="btn ghost" type="button" onclick="removeSelected('pair_list')">Remove Selected</button>

        </div>

        <button class="btn secondary" type="button" onclick="addPair()">Add Pair</button>

      </section>



      <section class="panel advanced-only" id="memory-anchors">

        <h2>Memory Anchors</h2>

        <div id="memory_list" class="list-stack"></div>

        <div id="memory_counter" class="token-counter"></div>

        <div class="list-tools">

          <button class="btn ghost" type="button" onclick="removeSelected('memory_list')">Remove Selected</button>

        </div>

        <button class="btn secondary" type="button" onclick="addMemory()">Add Memory</button>

        <div class="row">

          <button class="btn ghost" type="button" onclick="convertMemoryToLorebook()">Convert to Lorebook</button>

        </div>

      </section>



      <section class="panel advanced-only" id="lorebook">

        <h2>Lorebook</h2>

        <div id="lorebook_list" class="list-stack"></div>

        <div class="list-tools">

          <button class="btn secondary" type="button" onclick="addLorebook()">Add Entry</button>

          <button class="btn ghost" type="button" onclick="removeSelected('lorebook_list')">Remove Selected</button>

        </div>

      </section>



      <section class="panel advanced-only" id="metadata">

        <h2>Metadata</h2>

        <div class="grid">

          <div>

            <label>Tags</label>

            <div class="tag-editor">

              <div id="tag_list" class="chips"></div>

              <div class="row">

                <input id="tag_input" list="tag_suggestions" type="text" placeholder="Add tag..." />

                <button class="btn ghost" type="button" onclick="addTagFromInput()">Add</button>

              </div>

              <div id="tag_quick" class="chips"></div>

              <datalist id="tag_suggestions"></datalist>

            </div>

          </div>

          <div>

            <label>Creator</label>

            <input data-field="creator" type="text" value="{{ bot.fields.creator }}" />

          </div>

          <div>

            <label>Character Version</label>

            <div class="row">

              <input data-field="character_version" id="character_version" type="text" value="{{ bot.fields.character_version }}" />

              <button class="btn ghost" type="button" onclick="incrementVersion()">+1</button>

            </div>

          </div>

          <div>

            <label>Language</label>

            <input data-field="language" type="text" value="{{ bot.fields.language }}" />

          </div>

          <div>

            <label>Rating</label>

            <select data-field="rating" id="rating_select">

              {% for opt in ['SFW', 'Suggestive', 'NSFW', 'Extreme', 'Unrated'] %}

              <option value="{{ opt }}" {% if bot.fields.rating == opt %}selected{% endif %}>{{ opt }}</option>

              {% endfor %}

            </select>

            <div class="hint">Rating tag is auto-added on export.</div>

          </div>

        </div>

        <div class="grid">

          <div>

            <label>Author Notes</label>

            <textarea data-field="author_notes">{{ bot.fields.author_notes }}</textarea>

          </div>

          <div>

            <label>Memory Notes</label>

            <textarea data-field="memory_notes">{{ bot.fields.memory_notes }}</textarea>

          </div>

        </div>

      </section>



      <section class="panel advanced-only" id="import">

        <h2>Import</h2>

        <div class="list-stack">

          <div id="import_drop" class="drop-zone">Drop a file here or click to browse.</div>

          <input type="file" id="import_file" accept=".json,.png,.txt" />

          <button class="btn secondary" type="button" onclick="importFile()">Import File</button>

          <div class="hint">Supports Chub/SillyTavern card JSON/PNG, Janitor JSON, Risu JSON, and plaintext prompts.</div>

        </div>

      </section>



      <section class="panel advanced-only" id="exports">

        <h2>Exports</h2>

        <div class="row">

          <a class="btn secondary" href="{{ url_for('export_bot', bot_id=bot.id) }}">Raw JSON</a>

          <a class="btn" href="{{ url_for('export_card_v2', bot_id=bot.id) }}">Chub / Tavern Card v2</a>

          <a class="btn secondary" href="{{ url_for('export_janitor', bot_id=bot.id) }}">Janitor JSON</a>

          <a class="btn secondary" href="{{ url_for('export_risu', bot_id=bot.id) }}">Risu JSON</a>

          <a class="btn ghost" id="export_png_link" href="{{ url_for('export_card_v2_png', bot_id=bot.id) }}?embed=1&use_image=1">Tavern PNG</a>

        </div>

        <div class="row" style="margin-top: 8px;">

          <label class="chip">

            <input type="checkbox" id="png_embed" checked />

            <span>Embed card data in PNG</span>

          </label>

          <label class="chip">

            <input type="checkbox" id="png_use_image" checked />

            <span>Use first image as base</span>

          </label>

        </div>

        <div class="row" style="margin-top: 8px;">

          <label for="prompt_template">Prompt Template</label>

          <select id="prompt_template">

            <option value="plain">Plain</option>

            <option value="alpaca">Alpaca</option>

            <option value="vicuna">Vicuna</option>

            <option value="mistral">Mistral</option>

          </select>

          <a class="btn ghost" id="export_prompt_link" href="{{ url_for('export_prompt', bot_id=bot.id) }}?template=plain">Prompt TXT</a>

          <a class="btn ghost" href="{{ url_for('export_all') }}">Export All (zip)</a>

        </div>

        <div class="hint">Character Card v2 works for Chub and SillyTavern-style apps.</div>

      </section>



      <section class="panel advanced-only" id="ai-notes">

        <h2>AI Notes</h2>

        <div class="grid">

          <div>

            <label>Generation Notes</label>

            <textarea id="ai_notes" placeholder="Add guidance for generation, e.g. themes, vibe, or constraints."></textarea>

          </div>

        </div>

      </section>



      <section class="panel" id="test-chat">

        <h2>Test Chat</h2>

        <div class="row" style="align-items: center; justify-content: space-between;">

          <div class="mode-toggle chat-mode-toggle" id="chat_mode_toggle">

            <button type="button" data-mode="assistant">Assistant</button>

            <button type="button" data-mode="character">In Character</button>

          </div>

          <label class="chip">

            <input type="checkbox" id="chat_use_images" checked />

            <span>Use images (vision)</span>

          </label>

        </div>

        <div class="chat-log" id="chat_log"></div>

        <div class="row">

          <input type="text" id="chat_input" placeholder="Ask anything or test the current bot..." />

          <button class="btn" type="button" onclick="sendChat()">Send</button>

        </div>

      </section>



      <section class="panel advanced-only" id="utilities">

        <h2>Utilities</h2>

        <div class="row">

          <form method="post" action="{{ url_for('duplicate_bot_route', bot_id=bot.id) }}">

            <button class="btn secondary" type="submit">Duplicate Bot</button>

          </form>

          <form method="post" action="{{ url_for('delete_bot_route', bot_id=bot.id) }}" onsubmit="return confirm('Delete this bot?');">

            <button class="btn ghost" type="submit">Delete Bot</button>

          </form>

        </div>

      </section>



    </main>

  </div>



  <div class="modal" id="help_modal" aria-hidden="true">

    <div class="modal-backdrop" onclick="toggleHelp(false)"></div>

    <div class="modal-card">

      <div class="section-head">

        <h2>Help</h2>

        <button class="btn ghost" type="button" onclick="toggleHelp(false)">Close</button>

      </div>

      <div class="help-content">

        <div class="help-grid">

          <div class="faq-card">

            <h3>FAQ</h3>

            <div class="faq-item">

              <h4>How do I start fast?</h4>

              <p>Fill Original Input, add images if needed, then run Simple Generator. Use Compile for a polished pack.</p>

            </div>

            <div class="faq-item">

              <h4>What does Simple vs Advanced change?</h4>

              <p>Simple shows the minimal flow and outputs. Advanced reveals all profile sections, controls, and utilities.</p>

            </div>

            <div class="faq-item">

              <h4>What does Compile do?</h4>

              <p>Compile creates Description, First Messages, Scenario, and Example Dialogues together in one pass.</p>

            </div>

            <div class="faq-item">

              <h4>Why are outputs short?</h4>

              <p>Raise Max Tokens in AI Settings, enable High quality mode, and set Output Length targets in Compile. Larger models also help.</p>

            </div>

            <div class="faq-item">

              <h4>How does autosave work?</h4>

              <p>Autosave saves to local storage and version history on an interval. Manual save is always available.</p>

            </div>

            <div class="faq-item">

              <h4>How do exports work?</h4>

              <p>Use Card v2 for Chub/Tavern, Prompt TXT for external tools, and Export All for a zip.</p>

            </div>

          </div>

          <div class="help-chat">

            <div class="section-head">

              <h3>Ask the App</h3>

              <div class="help-mode-toggle" id="help_mode_toggle">

                <button type="button" data-mode="simple">Simple</button>

                <button type="button" data-mode="advanced">Advanced</button>

              </div>

            </div>

            <div class="help-chat-controls">

              <div style="flex: 1;">

                <label>Model</label>

                <select id="help_model">

                  <option value="grok-4-fast">grok-4-fast</option>

                  <option value="gpt-5-nano">gpt-5-nano</option>

                  <option value="claude-sonnet-4.5">claude-sonnet-4.5</option>

                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>

                </select>

              </div>

              <div>

                <label>&nbsp;</label>

                <button class="btn ghost" type="button" onclick="signInHelpChat()">Sign In</button>

              </div>

              <div>

                <label>&nbsp;</label>

                <button class="btn ghost" type="button" onclick="clearHelpChat()">Clear</button>

              </div>

            </div>

            <div id="help_chat_log" class="help-chat-log"></div>

            <textarea id="help_chat_input" class="help-chat-input" placeholder="Ask anything about Simple or Advanced mode..."></textarea>

            <div class="help-chat-controls">

              <button class="btn" type="button" onclick="sendHelpChat()">Ask</button>

              <div class="hint" id="help_status_hint">Uses Puter.js. Chat history stays in your browser.</div>

            </div>

          </div>

        </div>

      </div>

    </div>

  </div>



  <div class="modal" id="image_viewer_modal" aria-hidden="true">

    <div class="modal-backdrop" onclick="closeImageViewer()"></div>

    <div class="modal-card image-viewer-card">

      <div class="image-viewer-header">

        <div>

          <h2>Image Viewer</h2>

          <div class="meta" id="image_viewer_meta"></div>

        </div>

        <div class="image-viewer-actions">

          <button class="btn ghost" type="button" id="image_viewer_prev" onclick="stepImageViewer(-1)">Prev</button>

          <span class="image-viewer-counter" id="image_viewer_counter">0 / 0</span>

          <button class="btn ghost" type="button" id="image_viewer_next" onclick="stepImageViewer(1)">Next</button>

          <button class="btn ghost" type="button" onclick="closeImageViewer()">Close</button>

        </div>

      </div>

      <div class="image-viewer-frame">

        <img id="image_viewer_img" alt="Uploaded bot reference" />

      </div>

    </div>

  </div>



  <div class="modal onboarding-modal" id="onboard_modal" aria-hidden="true">

    <div class="modal-backdrop" onclick="finishOnboarding()"></div>

    <div class="onboard-focus" id="onboard_focus">

      <div class="onboard-focus-box" id="onboard_focus_box"></div>

    </div>

    <div class="onboard-card-wrap" id="onboard_card_wrap">

      <div class="modal-card onboarding-card" id="onboard_card">

      <div class="onboard-hero">

        <span class="onboard-badge">Quick Start</span>

        <h2>Make a bot in minutes</h2>

        <p>A short walkthrough to show the fastest path from idea to a clean, export-ready bot.</p>

        <div class="onboard-hero-stats">

          <div class="onboard-stat">

            <span class="onboard-stat-label">Mode</span>

            <span class="onboard-stat-value" id="onboard_mode_value">Simple</span>

          </div>

          <div class="onboard-stat">

            <span class="onboard-stat-label">Tour</span>

            <span class="onboard-stat-value" id="onboard_track_value">Simple</span>

          </div>

          <div class="onboard-stat">

            <span class="onboard-stat-label">Flow</span>

            <span class="onboard-stat-value" id="onboard_flow_value">Input -> Generate -> Compile</span>

          </div>

        </div>

      </div>

<div class="onboard-main">

        <div class="section-head">

          <h2>Quick Tour</h2>

          <button class="btn ghost" type="button" onclick="finishOnboarding()">Close</button>

        </div>

        <div class="onboard-track-row">

          <div class="mode-toggle onboard-track-toggle" id="onboard_track_toggle">

            <button type="button" data-track="simple">Simple Tour</button>

            <button type="button" data-track="advanced">Advanced Tour</button>

          </div>

          <div class="onboard-track-note" id="onboard_track_note">Advanced tour switches to Advanced mode.</div>

        </div>

        <div class="onboard-step-count" id="onboard_step_count">Step 1</div>

        <div class="onboard-progress">

          <div class="onboard-progress-bar" id="onboard_progress"></div>

        </div>

        <div class="onboard-step-card" id="onboard_step_card">

          <h3 id="onboard_step_title">Pick a bot</h3>

          <p id="onboard_step_body">Use New Bot for a fresh character or select one on the left to continue editing.</p>

        </div>

        <div class="onboard-actions">

          <div class="onboard-dots" id="onboard_dots"></div>

          <div class="row">

            <button class="btn ghost" type="button" id="onboard_back_btn" onclick="prevOnboardStep()">Back</button>

            <button class="btn" type="button" id="onboard_next_btn" onclick="nextOnboardStep()">Next</button>

            <button class="btn secondary" type="button" onclick="finishOnboarding()">Skip</button>

          </div>

        </div>

      </div>

    </div>

  </div>



<script>

  (function() {

    try {

      if (typeof exports === 'object' && typeof module === 'undefined') {

        window.module = { exports: exports };

      }

    } catch (err) {

      // Ignore if exports/module are not present.

    }

  })();

</script>

<script src="https://js.puter.com/v2/"></script>

<script>

  (function() {

    if (!window.puter && window.module && window.module.exports) {

      window.puter = window.module.exports.puter || window.module.exports;

    }

  })();

</script>

<script>

  const BOT = {{ bot_json | tojson }};

  const PROVIDER_DEFAULTS = {{ provider_defaults | tojson }};

  const COMMON_TAGS = {{ common_tags | tojson }};

  const GEN_SECTIONS = {{ gen_sections | tojson }};

  const GENERATABLE_FIELDS = {{ generatable_fields | tojson }};

  const MAX_IMAGES = {{ max_images }};

  const INSPIRATION_SEEDS = {

    firstNames: ['Ava', 'Riley', 'Nova', 'Jade', 'Maya', 'Sage', 'Theo', 'Liam', 'Kai', 'Rowan', 'Iris', 'Ezra', 'Mira', 'Remy', 'Skye'],

    lastNames: ['Hart', 'Vale', 'Stone', 'Quinn', 'Lark', 'Monroe', 'Ash', 'Kade', 'Rivers', 'Marlow', 'Solis', 'Wren', 'Hale', 'Fox', 'Blaine'],

    species: ['Human', 'Elf', 'Vampire', 'Werewolf', 'Android', 'Faerie', 'Demon', 'Angel', 'Sorcerer', 'Alien', 'Shapeshifter', 'Merfolk', 'Dragonkin', 'Cyborg'],

    settings: ['Neon city', 'Arcane academy', 'Coastal village', 'Desert outpost', 'Skyship port', 'Ancient ruins', 'Underground club', 'Quiet bookstore', 'Space station', 'Forest enclave', 'Royal court', 'Haunted manor', 'Cyber bazaar', 'Frontier town', 'Mountain retreat'],

    occupations: ['Detective', 'Bounty hunter', 'Bartender', 'Archivist', 'Smuggler', 'Medic', 'Bodyguard', 'Mechanic', 'Scholar', 'Oracle', 'Thief', 'Pilot', 'Diplomat', 'Chef', 'Musician'],

    relationships: ['Old friend', 'New partner', 'Rival-turned-ally', 'Mysterious benefactor', 'Roommate', 'Bodyguard assignment', 'Co-conspirator', 'Client', 'Fated meeting', 'Pen pal'],

    personality: ['Calm, observant, and quietly intense', 'Playful and clever with a soft heart', 'Stoic protector with a dry sense of humor', 'Charming trickster who hides vulnerability', 'Ambitious and disciplined, loyal to a fault', 'Warm, patient, and gently teasing', 'Restless adventurer hungry for stories', 'Pragmatic strategist with hidden warmth'],

    appearance: ['Sharp eyes, neat hair, and a minimalist wardrobe', 'Weathered jacket, soft smile, and calloused hands', 'Elegant attire with a signature accessory', 'Athletic build with a few visible scars', 'Bright eyes, messy hair, and layered jewelry', 'Tailored coat, polished boots, and a quiet gaze'],

    voice: ['Low and steady, with clipped sentences', 'Lively and quick, full of playful asides', 'Warm, measured, and slightly husky', 'Precise and formal, with occasional softness', 'Relaxed and casual, with a teasing lilt'],

  };

  const statusEl = document.getElementById('status');

  const saveStatusEl = document.getElementById('save_status');

  const themeSelect = document.getElementById('theme_select');

  const modeToggleEl = document.getElementById('mode_toggle');

  const botSearch = document.getElementById('bot_search');

  const botList = document.getElementById('bot_list');

  const imageGrid = document.getElementById('image_grid');

  const imageUpload = document.getElementById('image_upload');

  const tagListEl = document.getElementById('tag_list');

  const tagInputEl = document.getElementById('tag_input');

  const tagQuickEl = document.getElementById('tag_quick');

  const progressBar = document.getElementById('gen_progress');

  const previewOutput = document.getElementById('preview_output');

  const tokenTotalEl = document.getElementById('token_total');

  const firstMessagesEl = document.getElementById('first_messages');

  const primarySelectEl = document.getElementById('primary_greeting_select');

  const scenarioListEl = document.getElementById('scenario_list');

  const dialogueListEl = document.getElementById('dialogue_list');

  const pairListEl = document.getElementById('pair_list');

  const memoryListEl = document.getElementById('memory_list');

  const lorebookListEl = document.getElementById('lorebook_list');

  const importFileEl = document.getElementById('import_file');

  const importDropEl = document.getElementById('import_drop');

  const promptTemplateEl = document.getElementById('prompt_template');

  const exportPromptLink = document.getElementById('export_prompt_link');

  const exportPngLink = document.getElementById('export_png_link');

  const pngEmbedEl = document.getElementById('png_embed');

  const pngUseImageEl = document.getElementById('png_use_image');

  const presetNameEl = document.getElementById('preset_name');

  const presetSelectEl = document.getElementById('preset_select');

  const historySelectEl = document.getElementById('history_select');

  const historyMetaEl = document.getElementById('history_meta');

  const autosaveSecondsEl = document.getElementById('autosave_seconds');

  const qualityModeEl = document.getElementById('quality_mode');

  const allowEmojisEl = document.getElementById('allow_emojis');

  const imageViewerModalEl = document.getElementById('image_viewer_modal');

  const imageViewerImgEl = document.getElementById('image_viewer_img');

  const imageViewerMetaEl = document.getElementById('image_viewer_meta');

  const imageViewerCounterEl = document.getElementById('image_viewer_counter');

  const imageViewerPrevEl = document.getElementById('image_viewer_prev');

  const imageViewerNextEl = document.getElementById('image_viewer_next');

  const helpModalEl = document.getElementById('help_modal');

  const helpModeToggleEl = document.getElementById('help_mode_toggle');

  const helpModelEl = document.getElementById('help_model');

  const helpChatLogEl = document.getElementById('help_chat_log');

  const helpChatInputEl = document.getElementById('help_chat_input');

  const helpStatusHintEl = document.getElementById('help_status_hint');

  const onboardModalEl = document.getElementById('onboard_modal');

  const onboardModeValueEl = document.getElementById('onboard_mode_value');

  const onboardTrackValueEl = document.getElementById('onboard_track_value');

  const onboardFlowValueEl = document.getElementById('onboard_flow_value');

  const onboardTrackToggleEl = document.getElementById('onboard_track_toggle');
  const onboardTrackNoteEl = document.getElementById('onboard_track_note');
  const onboardFocusEl = document.getElementById('onboard_focus');
  const onboardFocusBoxEl = document.getElementById('onboard_focus_box');
  const onboardCardWrapEl = document.getElementById('onboard_card_wrap');
  const onboardStepCountEl = document.getElementById('onboard_step_count');

  const onboardProgressEl = document.getElementById('onboard_progress');

  const onboardStepCardEl = document.getElementById('onboard_step_card');

  const onboardStepTitleEl = document.getElementById('onboard_step_title');

  const onboardStepBodyEl = document.getElementById('onboard_step_body');

  const onboardDotsEl = document.getElementById('onboard_dots');

  const onboardNextBtnEl = document.getElementById('onboard_next_btn');

  const onboardBackBtnEl = document.getElementById('onboard_back_btn');

  const chatModeToggleEl = document.getElementById('chat_mode_toggle');

  const chatUseImagesEl = document.getElementById('chat_use_images');

  const simpleNameEl = document.getElementById('simple_name');

  const simpleAgeEl = document.getElementById('simple_age');

  const simpleSpeciesEl = document.getElementById('simple_species');

  const compileDescriptionEl = document.getElementById('compile_description_output');

  const compileScenarioEl = document.getElementById('compile_scenario_output');

  const compileFirstMessagesEl = document.getElementById('compile_first_messages_output');

  const compileDialoguesEl = document.getElementById('compile_dialogues_output');

  let lastProvider = document.getElementById('ai_provider').value;



  const QUALITY_HINT = 'High quality mode: write clean, vivid, and precise prose with careful detail, consistent tone, and no filler.';

  const HELP_MODELS = ['grok-4-fast', 'gpt-5-nano', 'claude-sonnet-4.5', 'gemini-2.5-flash'];

  const HELP_HISTORY_KEY = 'botmaker_help_history';

  const HELP_MODE_KEY = 'botmaker_help_mode';

  const HELP_MODEL_KEY = 'botmaker_help_model';

  const CHAT_MODE_KEY = 'botmaker_chat_mode';

  const CHAT_IMAGES_KEY = 'botmaker_chat_images';

  const ONBOARD_KEY = 'botmaker_onboarded';

  const ONBOARD_TRACK_KEY = 'botmaker_onboard_track';

  const HELP_SYSTEM_BASE = [

    'You are the embedded help assistant for the RP Bot Maker web app.',

    'You have full knowledge of the app UI and features and must answer clearly and concretely.',

    'Never invent buttons or sections that do not exist. Be concise but thorough when asked.',

    'Key features:',

    '- Simple Generator: Original Input + optional current name/age/species + current first messages/scenarios/dialogues. Generates description, personality, scenario, first message, and example dialogues.',

    '- Advanced mode: reveals all profile sections (Core Identity, Appearance, Voice/Style, World/Motivation, Traits/Limits, System Guidance, Roleplay Controls, Tone Toggles, Presets, History, Import, Exports, AI Notes, Test Chat, Utilities).',

    '- Compile: produces Description, First Messages, Scenario, and Example Dialogues into the Compiled Outputs panel. Outputs are not applied until you click Apply Outputs.',

    '- Output length targets: set minimum tokens for Description, First Messages, Scenario, and Dialogues in the Compile panel.',

    '- Simple/Advanced toggle: changes visible controls. Simple shows minimal flow; Advanced shows full control.',

    '- Generation: Generate All, Generate Profile, First Messages, Scenarios, Dialogues, and per-field gen buttons.',

    '- Quality mode and Allow emojis affect generation prompts and sanitization.',

    '- Autosave and History create local snapshots.',

    '- Exports: Raw JSON, Card v2, Janitor JSON, Risu JSON, Prompt TXT, PNG.',

    '- Placeholders: use {{user}} and {{char}} in dialogues and outputs.',

  ].join('\\n');



  const ONBOARD_STEPS_SIMPLE = [
    {
      title: 'Pick or create a bot',
      body: 'Use the Library list to open a bot or hit New Bot in the header. Use the Simple/Advanced toggle for more controls.',
      target: '#library-panel',
    },
    {
      title: 'Help and FAQ',
      body: 'Use the ? button for the FAQ and the in-app help chat.',
      target: '.help-btn',
    },
    {
      title: 'AI settings and API key',
      body: 'Choose a provider, paste your API key, and set max tokens. Save settings before generating.',
      target: '#ai-settings',
    },
    {
      title: 'Images for vision',
      body: 'Upload reference images to steer generation and enable vision testing.',
      target: '#images-panel',
    },
    {
      title: 'Simple generator',
      body: 'Describe the character, vibe, and scenario. Add name, age, and species if known.',
      target: '#simple-generator',
    },
    {
      title: 'Core identity',
      body: 'Edit name, description, and personality for the core character profile.',
      target: '#core-identity',
    },
    {
      title: 'Compile the pack',
      body: 'Compile creates Description, First Messages, Scenario, and Dialogues together. Toggle High quality and Allow emojis as needed.',
      target: '#compile',
    },
    {
      title: 'Review outputs',
      body: 'Use Compiled Outputs to copy or Apply Outputs into the main fields.',
      target: '#compile-output',
    },
    {
      title: 'First messages',
      body: 'Manage primary and alternate greetings in one list.',
      target: '#first-messages',
    },
    {
      title: 'Scenarios',
      body: 'Add or generate multiple scenarios for variety.',
      target: '#scenarios',
    },
    {
      title: 'Example dialogues',
      body: 'Add example dialogues to lock in voice and flow.',
      target: '#example-dialogues',
    },
    {
      title: 'Test chat',
      body: 'Switch between Assistant and In Character, and enable vision if needed.',
      target: '#test-chat',
    },
  ];
  const ONBOARD_STEPS_ADVANCED = [
    {
      title: 'Library and modes',
      body: 'Pick a bot in the Library and keep Advanced mode on for full control.',
      target: '#library-panel',
    },
    {
      title: 'Help and FAQ',
      body: 'Use the ? button for FAQ answers and the in-app help chat.',
      target: '.help-btn',
    },
    {
      title: 'AI settings and API key',
      body: 'Set provider, model, API key, base URL, and max tokens. Save settings before runs.',
      target: '#ai-settings',
    },
    {
      title: 'Images and vision',
      body: 'Upload images and toggle Use Images to control vision-aware generation.',
      target: '#images-panel',
    },
    {
      title: 'Simple generator',
      body: 'Use Simple Generator when you want a fast pass before deeper editing.',
      target: '#simple-generator',
    },
    {
      title: 'Generation controls',
      body: 'Generate all or specific sections, then choose merge behavior.',
      target: '#generation-controls',
    },
    {
      title: 'Preview',
      body: 'Check the live preview of your bot card or prompt formatting.',
      target: '#preview',
    },
    {
      title: 'Core identity',
      body: 'Name, description, and personality drive everything else.',
      target: '#core-identity',
    },
    {
      title: 'Appearance',
      body: 'Lock in visual traits and signature details.',
      target: '#appearance',
    },
    {
      title: 'Voice and style',
      body: 'Dial in voice, speech style, and mannerisms.',
      target: '#voice-style',
    },
    {
      title: 'World and motivation',
      body: 'Define setting, goals, and relationship to the user.',
      target: '#world-motivation',
    },
    {
      title: 'Traits and limits',
      body: 'Add strengths, flaws, and boundaries for clarity.',
      target: '#traits-limits',
    },
    {
      title: 'System guidance',
      body: 'Rules, formatting, and behavior constraints keep the bot on track.',
      target: '#system-guidance',
    },
    {
      title: 'Roleplay controls',
      body: 'Set response length, POV, narration, and formatting behavior.',
      target: '#roleplay-controls',
    },
    {
      title: 'Tone toggles',
      body: 'Use toggles to steer vibe and filters.',
      target: '#tone-toggles',
    },
    {
      title: 'Presets',
      body: 'Save and load presets for repeatable builds.',
      target: '#presets',
    },
    {
      title: 'History snapshots',
      body: 'Review autosaves and roll back if needed.',
      target: '#history',
    },
    {
      title: 'Compile',
      body: 'Compile a full pack with Description, First Messages, Scenario, and Dialogues.',
      target: '#compile',
    },
    {
      title: 'Compiled outputs',
      body: 'Review results and apply them in one click.',
      target: '#compile-output',
    },
    {
      title: 'First messages',
      body: 'Manage primary and alternate greetings.',
      target: '#first-messages',
    },
    {
      title: 'Scenarios',
      body: 'Store multiple scenarios and generate new ones.',
      target: '#scenarios',
    },
    {
      title: 'Example dialogues',
      body: 'Add sample exchanges to shape voice and flow.',
      target: '#example-dialogues',
    },
    {
      title: 'Prompt pairs',
      body: 'Add prompt/response pairs for fine-grained control.',
      target: '#prompt-pairs',
    },
    {
      title: 'Memory anchors',
      body: 'Store key memories and convert them to lorebook entries.',
      target: '#memory-anchors',
    },
    {
      title: 'Lorebook',
      body: 'Create structured lore entries for complex worlds.',
      target: '#lorebook',
    },
    {
      title: 'Metadata and tags',
      body: 'Add tags, creator info, and ratings to keep bots organized.',
      target: '#metadata',
    },
    {
      title: 'Import',
      body: 'Bring in existing cards or prompts from supported formats.',
      target: '#import',
    },
    {
      title: 'Exports',
      body: 'Export Card v2, PNG, or prompt TXT when ready.',
      target: '#exports',
    },
    {
      title: 'AI notes',
      body: 'Add generation notes to steer outputs for this bot.',
      target: '#ai-notes',
    },
    {
      title: 'Test chat',
      body: 'Test in Assistant/In Character and verify vision.',
      target: '#test-chat',
    },
    {
      title: 'Utilities',
      body: 'Duplicate or delete bots when you are done.',
      target: '#utilities',
    },
  ];


  let onboardIndex = 0;

  let onboardTrack = 'simple';

  let onboardActive = false;



  function isPuterClient(candidate) {

    return !!(candidate && (candidate.ai || candidate.auth || candidate.fs || candidate.ui));

  }



  function getPuterClient() {

    if (isPuterClient(window.puter)) return window.puter;

    if (window.module && window.module.exports) {

      const moduleExports = window.module.exports;

      if (isPuterClient(moduleExports?.puter)) return moduleExports.puter;

      if (isPuterClient(moduleExports)) return moduleExports;

    }

    if (window.exports && window.exports.puter && isPuterClient(window.exports.puter)) {

      return window.exports.puter;

    }

    if (window.exports && isPuterClient(window.exports)) return window.exports;

    return null;

  }



  const state = {

    images: BOT.images || [],

    tags: BOT.tags || [],

    compileOutputs: null,

    dirty: false,

    previewMode: 'card',

    saving: false,

  };

  let autosaveTimer = null;



  function setStatus(message, kind) {

    statusEl.textContent = message;

    statusEl.className = 'status';

    if (kind) {

      statusEl.classList.add(kind);

    }

  }



  function setSaveStatus(message, kind) {

    if (!saveStatusEl) return;

    saveStatusEl.textContent = message;

    saveStatusEl.className = 'status';

    if (kind) {

      saveStatusEl.classList.add(kind);

    }

  }



  function reportError(message) {

    setStatus(message || 'Unexpected error', 'error');

  }



  window.addEventListener('error', (event) => {

    reportError(event.message);

  });



  window.addEventListener('unhandledrejection', (event) => {

    const reason = event.reason && event.reason.message ? event.reason.message : String(event.reason || '');

    reportError(reason);

  });



  function markDirty() {

    if (!state.dirty) {

      state.dirty = true;

      setSaveStatus('Unsaved changes.', null);

    }

    updateTokenCounters();

    updatePreview();

  }



  function getGenerationNotes() {

    const notesEl = document.getElementById('ai_notes');

    const base = notesEl ? notesEl.value.trim() : '';

    const quality = qualityModeEl && qualityModeEl.checked ? QUALITY_HINT : '';

    const emojiHint = allowEmojisEl && allowEmojisEl.checked ? 'Emojis allowed.' : 'No emojis.';

    return [base, quality, emojiHint].filter(Boolean).join('\\n\\n');

  }



  function initEmojiToggle() {

    if (!allowEmojisEl) return;

    const saved = localStorage.getItem('allow_emojis');

    if (saved !== null) {

      allowEmojisEl.checked = saved === 'true';

    }

    allowEmojisEl.addEventListener('change', () => {

      localStorage.setItem('allow_emojis', allowEmojisEl.checked ? 'true' : 'false');

    });

  }



  const TOKEN_PRESET_VALUES = {

    low: '250',

    medium: '500',

    high: '750',

    'very high': '1000',

    very_high: '1000',

    extreme: '1500',

  };



  function normalizeTokenValue(raw) {

    const text = String(raw || '').trim().toLowerCase();

    if (!text || text === 'auto' || text === 'default' || text === 'none') return 'auto';

    if (TOKEN_PRESET_VALUES[text]) return TOKEN_PRESET_VALUES[text];

    const normalized = text.replace('_', ' ');

    if (TOKEN_PRESET_VALUES[normalized]) return TOKEN_PRESET_VALUES[normalized];

    const numeric = parseInt(text, 10);

    if (!Number.isNaN(numeric) && numeric > 0) return String(numeric);

    return 'auto';

  }



  function refreshTokenTargets() {

    document.querySelectorAll('[data-token-target]').forEach((toggleEl) => {

      const targetId = toggleEl.dataset.tokenTarget;

      const input = document.getElementById(targetId);

      if (!input) return;

      const normalized = normalizeTokenValue(input.value);

      let matched = false;

      toggleEl.querySelectorAll('button[data-value]').forEach((btn) => {

        const isActive = btn.dataset.value === normalized;

        btn.classList.toggle('active', isActive);

        if (isActive) matched = true;

      });

      if (!matched) {

        const autoBtn = toggleEl.querySelector('button[data-value="auto"]');

        if (autoBtn) autoBtn.classList.add('active');

        input.value = 'auto';

      } else {

        input.value = normalized;

      }

    });

  }



  function initTokenTargets() {

    document.querySelectorAll('[data-token-target]').forEach((toggleEl) => {

      const targetId = toggleEl.dataset.tokenTarget;

      const input = document.getElementById(targetId);

      if (!input) return;

      toggleEl.querySelectorAll('button[data-value]').forEach((btn) => {

        btn.addEventListener('click', () => {

          const value = normalizeTokenValue(btn.dataset.value);

          input.value = value;

          refreshTokenTargets();

          markDirty();

        });

      });

    });

    refreshTokenTargets();

  }



  function syncSimpleField(simpleEl, fieldKey) {

    if (!simpleEl) return;

    const target = document.querySelector(`[data-field="${fieldKey}"]`);

    if (!target) return;

    if (!simpleEl.value.trim() && target.value) {

      simpleEl.value = target.value;

    }

    simpleEl.addEventListener('input', () => {

      target.value = simpleEl.value;

      markDirty();

    });

  }



  function initSimpleFieldSync() {

    syncSimpleField(simpleNameEl, 'name');

    syncSimpleField(simpleAgeEl, 'age');

    syncSimpleField(simpleSpeciesEl, 'species');

  }



  function formatDialoguesOutput(dialogues) {

    if (!Array.isArray(dialogues)) return '';

    return dialogues.map((pair) => {

      const user = pair.user ? `{{user}}: ${pair.user}` : '';

      const bot = pair.bot ? `{{char}}: ${pair.bot}` : '';

      return [user, bot].filter(Boolean).join('\\n');

    }).filter(Boolean).join('\\n\\n');

  }



  function renderCompileOutputs(compiled) {

    state.compileOutputs = compiled || null;

    if (compileDescriptionEl) compileDescriptionEl.value = compiled?.description || '';

    if (compileScenarioEl) compileScenarioEl.value = compiled?.scenario || '';

    if (compileFirstMessagesEl) {

      const list = Array.isArray(compiled?.first_messages) ? compiled.first_messages : [];

      compileFirstMessagesEl.value = list.join('\\n\\n');

    }

    if (compileDialoguesEl) {

      compileDialoguesEl.value = formatDialoguesOutput(compiled?.dialogues || []);

    }

  }



  function copyText(text) {

    if (!text) return;

    if (navigator.clipboard && navigator.clipboard.writeText) {

      navigator.clipboard.writeText(text).then(() => {

        setStatus('Copied to clipboard.', 'ok');

      }).catch(() => {

        setStatus('Copy failed.', 'error');

      });

      return;

    }

    const temp = document.createElement('textarea');

    temp.value = text;

    document.body.appendChild(temp);

    temp.select();

    try {

      document.execCommand('copy');

      setStatus('Copied to clipboard.', 'ok');

    } catch (err) {

      setStatus('Copy failed.', 'error');

    }

    document.body.removeChild(temp);

  }



  function copyCompileOutputs() {

    if (!state.compileOutputs) {

      setStatus('No compiled outputs to copy.', 'error');

      return;

    }

    const compiled = state.compileOutputs;

    const text = [

      `Description:\\n${compiled.description || ''}`,

      `First Messages:\\n${(compiled.first_messages || []).join('\\n\\n')}`,

      `Scenario:\\n${compiled.scenario || ''}`,

      `Example Dialogues:\\n${formatDialoguesOutput(compiled.dialogues || [])}`,

    ].join('\\n\\n');

    copyText(text);

  }



  function copyCompileJson() {

    if (!state.compileOutputs) {

      setStatus('No compiled outputs to copy.', 'error');

      return;

    }

    const compiled = state.compileOutputs;

    const payload = {

      description: compiled.description || '',

      first_messages: compiled.first_messages || [],

      scenario: compiled.scenario || '',

      dialogues: compiled.dialogues || [],

    };

    copyText(JSON.stringify(payload, null, 2));

  }



  function applyCompileOutputs() {

    if (!state.compileOutputs) {

      setStatus('No compiled outputs to apply.', 'error');

      return;

    }

    if (!confirm('Apply compiled outputs to the bot fields and lists?')) {

      return;

    }

    const compiled = state.compileOutputs;

    applyFields({

      description: compiled.description || '',

      current_scenario: compiled.scenario || '',

    });

    if (firstMessagesEl) {

      firstMessagesEl.innerHTML = '';

      (compiled.first_messages || []).forEach((msg) => addListItem(firstMessagesEl, msg));

    }

    if (scenarioListEl) {

      scenarioListEl.innerHTML = '';

      if (compiled.scenario) addListItem(scenarioListEl, compiled.scenario);

    }

    if (dialogueListEl) {

      dialogueListEl.innerHTML = '';

      (compiled.dialogues || []).forEach((item) => addDialogueRow(dialogueListEl, item.user, item.bot));

    }

    const primaryInput = document.getElementById('primary_first_message');

    if (primaryInput && !primaryInput.value.trim() && (compiled.first_messages || []).length) {

      primaryInput.value = compiled.first_messages[0];

    }

    updatePrimarySelect();

    setStatus('Compiled outputs applied.', 'ok');

    markDirty();

  }



  function initQualityMode() {

    if (!qualityModeEl) return;

    const saved = localStorage.getItem('quality_mode');

    if (saved !== null) {

      qualityModeEl.checked = saved === 'true';

    }

    qualityModeEl.addEventListener('change', () => {

      localStorage.setItem('quality_mode', qualityModeEl.checked ? 'true' : 'false');

    });

  }



  function toggleHelp(show) {

    if (!helpModalEl) return;

    helpModalEl.classList.toggle('open', show);

    helpModalEl.setAttribute('aria-hidden', show ? 'false' : 'true');

  }



  function clampValue(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function rectOverlapArea(a, b) {
    const xOverlap = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
    const yOverlap = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    return xOverlap * yOverlap;
  }

  function positionOnboardCard(targetRect) {
    if (!onboardCardWrapEl) return;
    if (!targetRect) {
      onboardCardWrapEl.style.transform = '';
      return;
    }
    const cardRect = onboardCardWrapEl.getBoundingClientRect();
    const viewport = { width: window.innerWidth, height: window.innerHeight };
    const margin = 18;
    const maxLeft = Math.max(margin, viewport.width - cardRect.width - margin);
    const maxTop = Math.max(margin, viewport.height - cardRect.height - margin);
    const targetCenterX = (targetRect.left + targetRect.right) / 2;
    const targetCenterY = (targetRect.top + targetRect.bottom) / 2;
    const sideSpace = {
      left: targetRect.left - margin,
      right: viewport.width - targetRect.right - margin,
      top: targetRect.top - margin,
      bottom: viewport.height - targetRect.bottom - margin,
    };
    const positions = [];
    if (sideSpace.right >= cardRect.width) {
      positions.push({
        left: targetRect.right + margin,
        top: clampValue(targetCenterY - cardRect.height / 2, margin, maxTop),
      });
    }
    if (sideSpace.left >= cardRect.width) {
      positions.push({
        left: targetRect.left - margin - cardRect.width,
        top: clampValue(targetCenterY - cardRect.height / 2, margin, maxTop),
      });
    }
    if (sideSpace.bottom >= cardRect.height) {
      positions.push({
        left: clampValue(targetCenterX - cardRect.width / 2, margin, maxLeft),
        top: targetRect.bottom + margin,
      });
    }
    if (sideSpace.top >= cardRect.height) {
      positions.push({
        left: clampValue(targetCenterX - cardRect.width / 2, margin, maxLeft),
        top: targetRect.top - margin - cardRect.height,
      });
    }
    if (!positions.length) {
      positions.push(
        { left: margin, top: margin },
        { left: viewport.width - cardRect.width - margin, top: margin },
        { left: margin, top: viewport.height - cardRect.height - margin },
        { left: viewport.width - cardRect.width - margin, top: viewport.height - cardRect.height - margin },
        { left: (viewport.width - cardRect.width) / 2, top: margin },
        { left: (viewport.width - cardRect.width) / 2, top: viewport.height - cardRect.height - margin },
        { left: margin, top: (viewport.height - cardRect.height) / 2 },
        { left: viewport.width - cardRect.width - margin, top: (viewport.height - cardRect.height) / 2 },
        { left: (viewport.width - cardRect.width) / 2, top: (viewport.height - cardRect.height) / 2 }
      );
    }
    let best = null;
    let bestOverlap = Number.POSITIVE_INFINITY;
    let bestDistance = -Infinity;
    positions.forEach((pos) => {
      const left = clampValue(pos.left, margin, maxLeft);
      const top = clampValue(pos.top, margin, maxTop);
      const rect = {
        left,
        top,
        right: left + cardRect.width,
        bottom: top + cardRect.height,
      };
      const overlap = rectOverlapArea(rect, targetRect);
      const dist = Math.hypot(
        (rect.left + rect.right) / 2 - targetCenterX,
        (rect.top + rect.bottom) / 2 - targetCenterY
      );
      if (overlap < bestOverlap || (overlap === bestOverlap && dist > bestDistance)) {
        bestOverlap = overlap;
        bestDistance = dist;
        best = rect;
      }
    });
    if (!best) return;
    const dx = best.left - cardRect.left;
    const dy = best.top - cardRect.top;
    onboardCardWrapEl.style.transform = `translate3d(${Math.round(dx)}px, ${Math.round(dy)}px, 0)`;
  }

  function updateOnboardModeValue() {
    if (!onboardModeValueEl) return;

    const mode = document.documentElement.classList.contains('mode-advanced') ? 'Advanced' : 'Simple';

    onboardModeValueEl.textContent = mode;

  }



  function updateOnboardTrackValue() {

    if (onboardTrackValueEl) {

      onboardTrackValueEl.textContent = onboardTrack === 'advanced' ? 'Advanced' : 'Simple';

    }

    if (onboardTrackNoteEl) {

      onboardTrackNoteEl.textContent = onboardTrack === 'advanced'

        ? 'Advanced tour switches to Advanced mode.'

        : 'Simple tour focuses on the minimal flow.';

    }

  }



  function updateOnboardFlowValue() {

    if (!onboardFlowValueEl) return;

    const flow = onboardTrack === 'advanced'

      ? 'Profile -> Controls -> Export'

      : 'Input -> Generate -> Compile';

    onboardFlowValueEl.textContent = flow;

  }



  function getOnboardSteps() {

    return onboardTrack === 'advanced' ? ONBOARD_STEPS_ADVANCED : ONBOARD_STEPS_SIMPLE;

  }



  function setOnboardTrack(track, keepStep) {

    const next = track === 'advanced' ? 'advanced' : 'simple';

    onboardTrack = next;

    localStorage.setItem(ONBOARD_TRACK_KEY, next);

    if (onboardTrackToggleEl) {

      onboardTrackToggleEl.querySelectorAll('button[data-track]').forEach((btn) => {

        btn.classList.toggle('active', btn.dataset.track === next);

      });

    }

    if (next === 'advanced' && onboardActive) {

      setMode('advanced');

    }

    updateOnboardTrackValue();

    updateOnboardFlowValue();

    buildOnboardDots();

    const steps = getOnboardSteps();

    const nextIndex = keepStep ? Math.min(onboardIndex, steps.length - 1) : 0;

    renderOnboardStep(nextIndex);

  }



  function initOnboardTrack() {

    const saved = localStorage.getItem(ONBOARD_TRACK_KEY);

    const current = document.documentElement.classList.contains('mode-advanced') ? 'advanced' : 'simple';

    setOnboardTrack(saved || current || 'simple', true);

    if (onboardTrackToggleEl) {

      onboardTrackToggleEl.querySelectorAll('button[data-track]').forEach((btn) => {

        btn.addEventListener('click', () => setOnboardTrack(btn.dataset.track, false));

      });

    }

  }



  function toggleOnboarding(show) {
    if (!onboardModalEl) return;
    onboardActive = show;
    onboardModalEl.classList.toggle('open', show);
    onboardModalEl.setAttribute('aria-hidden', show ? 'false' : 'true');
    if (!show && onboardFocusEl) {
      onboardFocusEl.classList.remove('active');
    }
    if (!show && onboardCardWrapEl) {
      onboardCardWrapEl.style.transform = '';
    }
    if (show) {
      if (onboardTrack === 'advanced') {
        setMode('advanced');
      }
      updateOnboardModeValue();

      updateOnboardTrackValue();

      updateOnboardFlowValue();

      renderOnboardStep(0);

    }

  }



  function buildOnboardDots() {

    if (!onboardDotsEl) return;

    onboardDotsEl.innerHTML = '';

    getOnboardSteps().forEach((step, index) => {

      const dot = document.createElement('button');

      dot.type = 'button';

      dot.className = 'onboard-dot';

      dot.addEventListener('click', () => renderOnboardStep(index));

      onboardDotsEl.appendChild(dot);

    });

  }



  function isElementVisible(el) {

    return !!(el && el.getClientRects().length);

  }



  function focusOnboardTarget(selector, allowScroll = true) {
    if (!onboardFocusEl || !onboardFocusBoxEl) return;
    if (!selector) {
      onboardFocusEl.classList.remove('active');
      positionOnboardCard(null);
      return;
    }
    const target = document.querySelector(selector);
    if (!isElementVisible(target)) {
      onboardFocusEl.classList.remove('active');
      positionOnboardCard(null);
      return;
    }
    if (allowScroll) {

      const rect = target.getBoundingClientRect();

      const margin = 80;

      const inView = rect.top >= margin && rect.bottom <= (window.innerHeight - margin);

      if (!inView) {

        const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        target.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });

        setTimeout(() => focusOnboardTarget(selector, false), 350);

        return;

      }

    }

    const rect = target.getBoundingClientRect();
    const padding = 10;

    const left = Math.max(rect.left - padding, 8);

    const top = Math.max(rect.top - padding, 8);

    const width = Math.min(rect.width + padding * 2, window.innerWidth - left - 8);

    const height = Math.min(rect.height + padding * 2, window.innerHeight - top - 8);

    onboardFocusBoxEl.style.left = `${left}px`;

    onboardFocusBoxEl.style.top = `${top}px`;

    onboardFocusBoxEl.style.width = `${Math.max(width, 48)}px`;

    onboardFocusBoxEl.style.height = `${Math.max(height, 48)}px`;

    onboardFocusEl.style.setProperty('--focus-x', `${left + width / 2}px`);
    onboardFocusEl.style.setProperty('--focus-y', `${top + height / 2}px`);
    onboardFocusEl.style.setProperty('--focus-r', `${Math.max(width, height) * 0.7}px`);
    onboardFocusEl.classList.add('active');
    positionOnboardCard(rect);
  }


  function renderOnboardStep(index) {

    const steps = getOnboardSteps();

    const total = steps.length;

    onboardIndex = Math.min(Math.max(index, 0), total - 1);

    const step = steps[onboardIndex];

    if (onboardStepTitleEl) onboardStepTitleEl.textContent = step.title;

    if (onboardStepBodyEl) onboardStepBodyEl.textContent = step.body;

    if (onboardStepCountEl) {

      onboardStepCountEl.textContent = `Step ${onboardIndex + 1} of ${total}`;

    }

    if (onboardProgressEl) {

      onboardProgressEl.style.width = `${Math.round(((onboardIndex + 1) / total) * 100)}%`;

    }

    if (onboardNextBtnEl) {

      onboardNextBtnEl.textContent = onboardIndex === total - 1 ? 'Finish' : 'Next';

    }

    if (onboardBackBtnEl) {

      onboardBackBtnEl.disabled = onboardIndex === 0;

    }

    if (onboardDotsEl) {

      Array.from(onboardDotsEl.children).forEach((dot, dotIndex) => {

        dot.classList.toggle('active', dotIndex === onboardIndex);

      });

    }

    if (onboardStepCardEl) {

      onboardStepCardEl.classList.remove('animate');

      void onboardStepCardEl.offsetWidth;

      onboardStepCardEl.classList.add('animate');

    }

    if (onboardActive && step?.target) {

      focusOnboardTarget(step.target, true);

    } else if (onboardFocusEl) {

      onboardFocusEl.classList.remove('active');

    }

    updateOnboardModeValue();

    updateOnboardTrackValue();

    updateOnboardFlowValue();

  }



  function nextOnboardStep() {

    if (onboardIndex >= getOnboardSteps().length - 1) {

      finishOnboarding();

      return;

    }

    renderOnboardStep(onboardIndex + 1);

  }



  function prevOnboardStep() {

    renderOnboardStep(onboardIndex - 1);

  }



  function finishOnboarding() {
    localStorage.setItem(ONBOARD_KEY, 'true');
    onboardActive = false;
    if (onboardFocusEl) onboardFocusEl.classList.remove('active');
    if (onboardCardWrapEl) onboardCardWrapEl.style.transform = '';
    toggleOnboarding(false);
  }


  function initOnboarding() {

    if (!onboardModalEl) return;

    initOnboardTrack();

    window.addEventListener('resize', () => {

      if (!onboardActive) return;

      const step = getOnboardSteps()[onboardIndex];

      focusOnboardTarget(step?.target, false);

    });

    window.addEventListener('scroll', () => {

      if (!onboardActive) return;

      const step = getOnboardSteps()[onboardIndex];

      focusOnboardTarget(step?.target, false);

    }, { passive: true });

    const seen = localStorage.getItem(ONBOARD_KEY) === 'true';

    if (!seen) {

      setTimeout(() => toggleOnboarding(true), 450);

    }

  }



  function loadHelpHistory() {

    try {

      return JSON.parse(localStorage.getItem(HELP_HISTORY_KEY) || '[]');

    } catch (err) {

      return [];

    }

  }



  function saveHelpHistory(history) {

    localStorage.setItem(HELP_HISTORY_KEY, JSON.stringify(history));

  }



  function renderHelpChat() {

    if (!helpChatLogEl) return;

    const history = loadHelpHistory();

    helpChatLogEl.innerHTML = '';

    history.forEach((item) => {

      const bubble = document.createElement('div');

      bubble.className = `help-chat-bubble ${item.role}`;

      bubble.textContent = item.content;

      helpChatLogEl.appendChild(bubble);

    });

    helpChatLogEl.scrollTop = helpChatLogEl.scrollHeight;

  }



  function setHelpMode(mode) {

    const next = mode === 'advanced' ? 'advanced' : 'simple';

    if (helpModeToggleEl) {

      helpModeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {

        btn.classList.toggle('active', btn.dataset.mode === next);

      });

    }

    localStorage.setItem(HELP_MODE_KEY, next);

  }



  function initHelpMode() {

    if (!helpModeToggleEl) return;

    const saved = localStorage.getItem(HELP_MODE_KEY);

    const current = document.documentElement.classList.contains('mode-advanced') ? 'advanced' : 'simple';

    setHelpMode(saved || current || 'simple');

    helpModeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {

      btn.addEventListener('click', () => setHelpMode(btn.dataset.mode));

    });

  }



  function initHelpModel() {

    if (!helpModelEl) return;

    const saved = localStorage.getItem(HELP_MODEL_KEY);

    if (saved && HELP_MODELS.includes(saved)) {

      helpModelEl.value = saved;

    }

    helpModelEl.addEventListener('change', () => {

      localStorage.setItem(HELP_MODEL_KEY, helpModelEl.value);

    });

  }



  function buildHelpSystemPrompt() {

    const mode = localStorage.getItem(HELP_MODE_KEY) || 'simple';

    const scope = mode === 'advanced'

      ? 'Only answer questions about Advanced mode features and controls.'

      : 'Only answer questions about Simple mode and Compile basics.';

    return `${HELP_SYSTEM_BASE}\n\n${scope}`;

  }



  function formatHelpError(err) {

    if (!err) return 'Help chat failed.';

    if (typeof err === 'string') return err;

    if (err.message) return err.message;

    if (err.error) return err.error;

    if (err.code) return String(err.code);

    try {

      return JSON.stringify(err);

    } catch (jsonErr) {

      return 'Help chat failed.';

    }

  }



  async function callPuterChat(messages, model) {

    const puterClient = getPuterClient();

    if (!puterClient || !puterClient.ai || typeof puterClient.ai.chat !== 'function') {

      throw new Error('Puter.js is not available. Make sure you are logged in to Puter and the script loaded.');

    }

    let result = null;

    try {

      result = await puterClient.ai.chat(messages, false, { model });

    } catch (err) {

      result = await puterClient.ai.chat(messages, { model });

    }

    if (typeof result === 'string') return result;

    if (result?.error) {

      throw new Error(result.error.message || result.error);

    }

    if (result?.text) return result.text;

    if (result?.message?.content) return result.message.content;

    if (result?.content) return result.content;

    if (result?.choices?.[0]?.message?.content) return result.choices[0].message.content;

    return '';

  }



  async function updateHelpAuthStatus() {

    if (!helpStatusHintEl) return;

    const puterClient = getPuterClient();

    if (!puterClient || !puterClient.auth) {

      if (puterClient && puterClient.ui && typeof puterClient.ui.authenticateWithPuter === 'function') {

        helpStatusHintEl.textContent = 'Sign in to use the help chat.';

      } else {

        helpStatusHintEl.textContent = 'Puter.js not loaded yet.';

      }

      return;

    }

    if (typeof puterClient.auth.isSignedIn !== 'function') {

      helpStatusHintEl.textContent = 'Puter auth not ready. Try reload.';

      return;

    }

    try {

      const signedIn = await Promise.resolve(puterClient.auth.isSignedIn());

      if (signedIn) {

        helpStatusHintEl.textContent = 'Signed in. Ask a question.';

      } else {

        helpStatusHintEl.textContent = 'Sign in to use the help chat.';

      }

    } catch (err) {

      helpStatusHintEl.textContent = 'Sign in to use the help chat.';

    }

  }



  async function signInHelpChat() {

    try {

      const puterClient = getPuterClient();

      if (!puterClient) {

        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Puter.js not loaded yet.';

        return false;

      }

      if (puterClient.auth && typeof puterClient.auth.signIn === 'function') {

        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Opening Puter sign-in...';

        await puterClient.auth.signIn({ attempt_temp_user_creation: true });

      } else if (puterClient.ui && typeof puterClient.ui.authenticateWithPuter === 'function') {

        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Requesting Puter auth...';

        await puterClient.ui.authenticateWithPuter();

      } else {

        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Puter.js auth is not available.';

        return false;

      }

      if (helpStatusHintEl) helpStatusHintEl.textContent = 'Signed in. Ask a question.';

      return true;

    } catch (err) {

      if (helpStatusHintEl) helpStatusHintEl.textContent = err.message || 'Sign-in failed.';

      return false;

    }

  }



  async function ensureHelpAuth() {

    const puterClient = getPuterClient();

    if (!puterClient || !puterClient.auth) return false;

    if (typeof puterClient.auth.isSignedIn === 'function') {

      const signedIn = await Promise.resolve(puterClient.auth.isSignedIn());

      if (signedIn) return true;

    }

    return await signInHelpChat();

  }



  async function sendHelpChat() {

    if (!helpChatInputEl || !helpChatLogEl) return;

    const text = helpChatInputEl.value.trim();

    if (!text) return;

    helpChatInputEl.value = '';

    const authed = await ensureHelpAuth();

    if (!authed) {

      if (helpStatusHintEl) helpStatusHintEl.textContent = 'Sign in to use the help chat.';

      return;

    }

    const history = loadHelpHistory();

    history.push({ role: 'user', content: text });

    saveHelpHistory(history);

    renderHelpChat();

    if (helpStatusHintEl) helpStatusHintEl.textContent = 'Thinking...';

    const model = helpModelEl ? helpModelEl.value : HELP_MODELS[0];

    const systemPrompt = buildHelpSystemPrompt();

    const messages = [{ role: 'system', content: systemPrompt }].concat(

      history.map((item) => ({ role: item.role, content: item.content }))

    );

    try {

      const reply = await callPuterChat(messages, model);

      if (!reply) {

        throw new Error('No response from model.');

      }

      history.push({ role: 'bot', content: reply });

      saveHelpHistory(history);

      renderHelpChat();

      if (helpStatusHintEl) helpStatusHintEl.textContent = 'Ready.';

    } catch (err) {

      if (helpStatusHintEl) helpStatusHintEl.textContent = formatHelpError(err);

    }

  }



  function clearHelpChat() {

    localStorage.removeItem(HELP_HISTORY_KEY);

    renderHelpChat();

    if (helpStatusHintEl) helpStatusHintEl.textContent = 'Cleared.';

  }



  function initHelpChat() {

    renderHelpChat();

    initHelpMode();

    initHelpModel();

    updateHelpAuthStatus();

    if (helpChatInputEl) {

      helpChatInputEl.addEventListener('keydown', (event) => {

        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {

          event.preventDefault();

          sendHelpChat();

        }

      });

    }

  }



  function setChatMode(mode) {

    const next = mode === 'character' ? 'character' : 'assistant';

    if (chatModeToggleEl) {

      chatModeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {

        btn.classList.toggle('active', btn.dataset.mode === next);

      });

    }

    localStorage.setItem(CHAT_MODE_KEY, next);

    return next;

  }



  function getChatMode() {

    return localStorage.getItem(CHAT_MODE_KEY) || 'assistant';

  }



  function initChatMode() {

    if (!chatModeToggleEl) return;

    const saved = getChatMode();

    setChatMode(saved);

    chatModeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {

      btn.addEventListener('click', () => setChatMode(btn.dataset.mode));

    });

  }



  function initChatVision() {

    if (!chatUseImagesEl) return;

    const saved = localStorage.getItem(CHAT_IMAGES_KEY);

    if (saved !== null) {

      chatUseImagesEl.checked = saved === 'true';

    }

    chatUseImagesEl.addEventListener('change', () => {

      localStorage.setItem(CHAT_IMAGES_KEY, chatUseImagesEl.checked ? 'true' : 'false');

    });

  }



  function setMode(mode) {

    const next = mode === 'advanced' ? 'advanced' : 'simple';

    const root = document.documentElement;

    root.classList.toggle('mode-simple', next === 'simple');

    root.classList.toggle('mode-advanced', next === 'advanced');

    if (modeToggleEl) {

      modeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {

        btn.classList.toggle('active', btn.dataset.mode === next);

      });

    }

    localStorage.setItem('ui_mode', next);

    updateOnboardModeValue();

    if (onboardActive) {

      const step = getOnboardSteps()[onboardIndex];

      focusOnboardTarget(step?.target, false);

    }

  }



  function initModeToggle() {

    if (!modeToggleEl) return;

    const saved = localStorage.getItem('ui_mode') || 'simple';

    setMode(saved);

    modeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {

      btn.addEventListener('click', () => setMode(btn.dataset.mode));

    });

  }



  function applyTheme(mode) {

    if (mode === 'system') {

      document.documentElement.removeAttribute('data-theme');

    } else {

      document.documentElement.setAttribute('data-theme', mode);

    }

    localStorage.setItem('theme', mode);

  }



  function initTheme() {

    const saved = localStorage.getItem('theme') || 'system';

    themeSelect.value = saved;

    applyTheme(saved);

    themeSelect.addEventListener('change', (event) => {

      applyTheme(event.target.value);

    });

  }



  function initSearch() {

    if (!botSearch || !botList) {

      return;

    }

    botSearch.addEventListener('input', (event) => {

      const query = event.target.value.trim().toLowerCase();

      const cards = botList.querySelectorAll('.bot-card');

      cards.forEach((card) => {

        const hay = (card.dataset.search || '').toLowerCase();

        card.style.display = hay.includes(query) ? '' : 'none';

      });

    });

  }



  function initLibraryImageViewer() {

    if (!botList) return;

    botList.addEventListener('click', (event) => {

      const target = event.target;

      if (!(target instanceof Element)) return;

      const carousel = target.closest('.bot-card-images');

      if (!carousel) return;

      const card = target.closest('.bot-card');

      if (!card) return;

      event.preventDefault();

      event.stopPropagation();

      let images = [];

      try {

        images = JSON.parse(card.dataset.images || '[]');

      } catch (err) {

        images = [];

      }

      if (!images.length) return;

      const title = card.dataset.title || 'Bot images';

      openImageViewer(images, title, 0);

    });

  }



  function initPromptTemplate() {

    if (!promptTemplateEl || !exportPromptLink) return;

    promptTemplateEl.addEventListener('change', () => {

      const value = promptTemplateEl.value;

      const base = exportPromptLink.getAttribute('href').split('?')[0];

      exportPromptLink.setAttribute('href', `${base}?template=${value}`);

    });

  }



  function updatePngExportLink() {

    if (!exportPngLink) return;

    const embed = pngEmbedEl && pngEmbedEl.checked ? '1' : '0';

    const useImage = pngUseImageEl && pngUseImageEl.checked ? '1' : '0';

    const base = exportPngLink.getAttribute('href').split('?')[0];

    exportPngLink.setAttribute('href', `${base}?embed=${embed}&use_image=${useImage}`);

  }



  function initPngExportToggle() {

    if (!exportPngLink) return;

    if (pngEmbedEl) pngEmbedEl.addEventListener('change', updatePngExportLink);

    if (pngUseImageEl) pngUseImageEl.addEventListener('change', updatePngExportLink);

    updatePngExportLink();

  }



  function loadPresets() {

    try {

      return JSON.parse(localStorage.getItem('botmaker_presets') || '[]');

    } catch (err) {

      return [];

    }

  }



  function savePresets(presets) {

    localStorage.setItem('botmaker_presets', JSON.stringify(presets));

  }



  function renderPresetSelect() {

    if (!presetSelectEl) return;

    const presets = loadPresets();

    presetSelectEl.innerHTML = '';

    presets.forEach((preset, idx) => {

      const opt = document.createElement('option');

      opt.value = String(idx);

      opt.textContent = preset.name;

      presetSelectEl.appendChild(opt);

    });

  }



  function savePreset() {

    if (!presetNameEl || !presetSelectEl) return;

    const name = presetNameEl.value.trim();

    if (!name) {

      setStatus('Preset name required.', 'error');

      return;

    }

    const preset = {

      name,

      toggles: collectToggles(),

      fields: {

        response_length: document.querySelector('[data-field=\"response_length\"]')?.value || '',

        pov: document.querySelector('[data-field=\"pov\"]')?.value || '',

        narration_style: document.querySelector('[data-field=\"narration_style\"]')?.value || '',

        emoji_use: document.querySelector('[data-field=\"emoji_use\"]')?.value || '',

        formatting: document.querySelector('[data-field=\"formatting\"]')?.value || '',

        style_rules: document.querySelector('[data-field=\"style_rules\"]')?.value || '',

        consent_rules: document.querySelector('[data-field=\"consent_rules\"]')?.value || '',

        boundaries: document.querySelector('[data-field=\"boundaries\"]')?.value || '',

        limits: document.querySelector('[data-field=\"limits\"]')?.value || '',

      },

    };

    const presets = loadPresets().filter((p) => p.name !== name);

    presets.push(preset);

    savePresets(presets);

    renderPresetSelect();

    setStatus('Preset saved.', 'ok');

  }



  function loadPreset() {

    const presets = loadPresets();

    const idx = parseInt(presetSelectEl?.value || '-1', 10);

    if (idx < 0 || !presets[idx]) {

      setStatus('Select a preset.', 'error');

      return;

    }

    const preset = presets[idx];

    document.querySelectorAll('input[name=\"toggle\"]').forEach((el) => {

      el.checked = preset.toggles.includes(el.value);

    });

    Object.entries(preset.fields || {}).forEach(([key, value]) => {

      const el = document.querySelector(`[data-field=\"${key}\"]`);

      if (el) el.value = value;

    });

    markDirty();

    setStatus(`Preset loaded: ${preset.name}`, 'ok');

  }



  function deletePreset() {

    const presets = loadPresets();

    const idx = parseInt(presetSelectEl?.value || '-1', 10);

    if (idx < 0 || !presets[idx]) {

      setStatus('Select a preset.', 'error');

      return;

    }

    const name = presets[idx].name;

    presets.splice(idx, 1);

    savePresets(presets);

    renderPresetSelect();

    setStatus(`Preset deleted: ${name}`, 'ok');

  }



  const imageViewerState = {

    images: [],

    index: 0,

    title: '',

  };



  function renderImageViewer() {

    if (!imageViewerModalEl || !imageViewerImgEl) return;

    const total = imageViewerState.images.length;

    if (!total) {

      closeImageViewer();

      return;

    }

    const clampedIndex = Math.min(Math.max(imageViewerState.index, 0), total - 1);

    imageViewerState.index = clampedIndex;

    const filename = imageViewerState.images[clampedIndex];

    imageViewerImgEl.src = `/images/${filename}`;

    if (imageViewerMetaEl) {

      const title = imageViewerState.title ? `${imageViewerState.title} • ` : '';

      imageViewerMetaEl.textContent = `${title}${filename}`;

    }

    if (imageViewerCounterEl) {

      imageViewerCounterEl.textContent = `${clampedIndex + 1} / ${total}`;

    }

    if (imageViewerPrevEl) {

      imageViewerPrevEl.disabled = total <= 1;

    }

    if (imageViewerNextEl) {

      imageViewerNextEl.disabled = total <= 1;

    }

    imageViewerModalEl.classList.add('open');

    imageViewerModalEl.setAttribute('aria-hidden', 'false');

  }



  function openImageViewer(srcOrList, label, startIndex = 0) {

    if (Array.isArray(srcOrList)) {

      imageViewerState.images = srcOrList.slice();

      imageViewerState.index = startIndex;

      imageViewerState.title = label || '';

      renderImageViewer();

      return;

    }

    const src = String(srcOrList || '');

    const filename = src.split('/').pop() || '';

    imageViewerState.images = filename ? [filename] : [];

    imageViewerState.index = 0;

    imageViewerState.title = label || '';

    renderImageViewer();

  }



  function stepImageViewer(direction) {

    if (!imageViewerState.images.length) return;

    imageViewerState.index = (imageViewerState.index + direction + imageViewerState.images.length) % imageViewerState.images.length;

    renderImageViewer();

  }



  function closeImageViewer() {

    if (!imageViewerModalEl) return;

    imageViewerModalEl.classList.remove('open');

    imageViewerModalEl.setAttribute('aria-hidden', 'true');

    if (imageViewerImgEl) imageViewerImgEl.removeAttribute('src');

    if (imageViewerMetaEl) imageViewerMetaEl.textContent = '';

    if (imageViewerCounterEl) imageViewerCounterEl.textContent = '0 / 0';

  }



  function renderImages() {

    imageGrid.innerHTML = '';

    state.images.forEach((filename) => {

      const card = document.createElement('div');

      card.className = 'image-card';

      const img = document.createElement('img');

      img.src = `/images/${filename}`;

      img.alt = filename;

      img.addEventListener('click', () => openImageViewer(state.images, 'Current bot images', state.images.indexOf(filename)));

      const btn = document.createElement('button');

      btn.type = 'button';

      btn.textContent = 'x';

      btn.onclick = (event) => {

        event.stopPropagation();

        removeImage(filename);

      };

      card.appendChild(img);

      card.appendChild(btn);

      imageGrid.appendChild(card);

    });

  }



  function renderTags() {

    if (!tagListEl) {

      return;

    }

    tagListEl.innerHTML = '';

    state.tags.forEach((tag) => {

      const chip = document.createElement('span');

      chip.className = 'tag-chip';

      chip.textContent = tag;

      const remove = document.createElement('button');

      remove.type = 'button';

      remove.textContent = 'x';

      remove.onclick = () => removeTag(tag);

      chip.appendChild(remove);

      tagListEl.appendChild(chip);

    });

  }



  function addTag(tag) {

    const cleaned = String(tag || '').trim();

    if (!cleaned) {

      return;

    }

    if (!state.tags.includes(cleaned)) {

      state.tags.push(cleaned);

      renderTags();

      markDirty();

    }

  }



  function removeTag(tag) {

    state.tags = state.tags.filter((item) => item !== tag);

    renderTags();

    markDirty();

  }



  function addTagFromInput() {

    if (!tagInputEl) {

      return;

    }

    addTag(tagInputEl.value);

    tagInputEl.value = '';

  }



  function initTagSuggestions() {

    const list = document.getElementById('tag_suggestions');

    if (!list) {

      return;

    }

    list.innerHTML = '';

    const suggestions = [...new Set([...COMMON_TAGS, ...Array.from(document.querySelectorAll('input[name=\"toggle\"]')).map((el) => el.value)])];

    suggestions.forEach((tag) => {

      const opt = document.createElement('option');

      opt.value = tag;

      list.appendChild(opt);

    });

    if (tagQuickEl) {

      tagQuickEl.innerHTML = '';

      suggestions.slice(0, 18).forEach((tag) => {

        const btn = document.createElement('button');

        btn.type = 'button';

        btn.className = 'btn ghost';

        btn.textContent = tag;

        btn.onclick = () => addTag(tag);

        tagQuickEl.appendChild(btn);

      });

    }

    if (tagInputEl) {

      tagInputEl.addEventListener('keydown', (event) => {

        if (event.key === 'Enter') {

          event.preventDefault();

          addTagFromInput();

        }

      });

    }

  }



  function initDragLists() {

    enableDrag(firstMessagesEl);

    enableDrag(scenarioListEl);

    enableDrag(dialogueListEl);

    enableDrag(pairListEl);

    enableDrag(memoryListEl);

    enableDrag(lorebookListEl);

  }



  async function removeImage(filename) {

    const res = await fetch(`/api/bot/${BOT.id}/remove_image`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify({ filename }),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Failed to remove image', 'error');

      return;

    }

    state.images = data.images || [];

    renderImages();

    setStatus('Image removed.', 'ok');

    markDirty();

  }



  imageUpload.addEventListener('change', async () => {

    if (!imageUpload.files.length) {

      return;

    }

    if (state.images.length + imageUpload.files.length > MAX_IMAGES) {

      setStatus(`Max ${MAX_IMAGES} images allowed.`, 'error');

      return;

    }

    const form = new FormData();

    Array.from(imageUpload.files).forEach((file) => form.append('images', file));

    const res = await fetch(`/api/bot/${BOT.id}/upload`, { method: 'POST', body: form });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Upload failed', 'error');

      return;

    }

    state.images = data.images || [];

    imageUpload.value = '';

    renderImages();

    setStatus('Images uploaded.', 'ok');

    markDirty();

  });



  function addListItem(container, value = '') {

    const row = document.createElement('div');

    row.className = 'list-item';

    row.draggable = true;

    row.addEventListener('dragstart', () => {

      row.classList.add('dragging');

    });

    row.addEventListener('dragend', () => {

      row.classList.remove('dragging');

      markDirty();

    });

    const checkbox = document.createElement('input');

    checkbox.type = 'checkbox';

    checkbox.className = 'row-select';

    const handle = document.createElement('span');

    handle.className = 'drag-handle';

    handle.textContent = '::';

    const textarea = document.createElement('textarea');

    textarea.value = value;

    textarea.addEventListener('input', markDirty);

    const remove = document.createElement('button');

    remove.type = 'button';

    remove.className = 'btn ghost';

    remove.textContent = 'Remove';

    remove.onclick = () => {

      row.remove();

      markDirty();

    };

    row.appendChild(checkbox);

    row.appendChild(handle);

    row.appendChild(textarea);

    row.appendChild(remove);

    container.appendChild(row);

  }



  function addDialogueRow(container, userValue = '', botValue = '') {

    const row = document.createElement('div');

    row.className = 'dialogue-row';

    row.draggable = true;

    row.addEventListener('dragstart', () => {

      row.classList.add('dragging');

    });

    row.addEventListener('dragend', () => {

      row.classList.remove('dragging');

      markDirty();

    });

    const checkbox = document.createElement('input');

    checkbox.type = 'checkbox';

    checkbox.className = 'row-select';

    const handle = document.createElement('span');

    handle.className = 'drag-handle';

    handle.textContent = '::';

    const user = document.createElement('textarea');

    user.placeholder = 'User';

    user.value = userValue;

    user.addEventListener('input', markDirty);

    const bot = document.createElement('textarea');

    bot.placeholder = 'Bot';

    bot.value = botValue;

    bot.addEventListener('input', markDirty);

    const remove = document.createElement('button');

    remove.type = 'button';

    remove.className = 'btn ghost';

    remove.textContent = 'Remove';

    remove.onclick = () => {

      row.remove();

      markDirty();

    };

    row.appendChild(checkbox);

    row.appendChild(handle);

    row.appendChild(user);

    row.appendChild(bot);

    row.appendChild(remove);

    container.appendChild(row);

  }



  function addPairRow(container, promptValue = '', responseValue = '') {

    const row = document.createElement('div');

    row.className = 'lore-row';

    row.draggable = true;

    row.addEventListener('dragstart', () => {

      row.classList.add('dragging');

    });

    row.addEventListener('dragend', () => {

      row.classList.remove('dragging');

      markDirty();

    });

    const checkbox = document.createElement('input');

    checkbox.type = 'checkbox';

    checkbox.className = 'row-select';

    const handle = document.createElement('span');

    handle.className = 'drag-handle';

    handle.textContent = '::';

    const prompt = document.createElement('textarea');

    prompt.placeholder = 'Input text';

    prompt.value = promptValue;

    prompt.addEventListener('input', markDirty);

    const response = document.createElement('textarea');

    response.placeholder = 'Response';

    response.value = responseValue;

    response.addEventListener('input', markDirty);

    const remove = document.createElement('button');

    remove.type = 'button';

    remove.className = 'btn ghost';

    remove.textContent = 'Remove';

    remove.onclick = () => {

      row.remove();

      markDirty();

    };

    row.appendChild(checkbox);

    row.appendChild(handle);

    row.appendChild(prompt);

    row.appendChild(response);

    row.appendChild(remove);

    container.appendChild(row);

  }



  function addLorebookEntry(container, entry = {}) {

    const row = document.createElement('div');

    row.className = 'pair-row';

    row.draggable = true;

    row.addEventListener('dragstart', () => {

      row.classList.add('dragging');

    });

    row.addEventListener('dragend', () => {

      row.classList.remove('dragging');

      markDirty();

    });

    const checkbox = document.createElement('input');

    checkbox.type = 'checkbox';

    checkbox.className = 'row-select';

    const handle = document.createElement('span');

    handle.className = 'drag-handle';

    handle.textContent = '::';

    const key = document.createElement('textarea');

    key.placeholder = 'Key';

    key.value = entry.key || '';

    key.addEventListener('input', markDirty);

    const content = document.createElement('textarea');

    content.placeholder = 'Content';

    content.value = entry.content || '';

    content.addEventListener('input', markDirty);

    const enabled = document.createElement('input');

    enabled.type = 'checkbox';

    enabled.checked = entry.enabled !== false;

    enabled.className = 'lore-enabled';

    enabled.addEventListener('change', markDirty);

    const remove = document.createElement('button');

    remove.type = 'button';

    remove.className = 'btn ghost';

    remove.textContent = 'Remove';

    remove.onclick = () => {

      row.remove();

      markDirty();

    };

    row.appendChild(checkbox);

    row.appendChild(handle);

    row.appendChild(key);

    row.appendChild(content);

    row.appendChild(enabled);

    row.appendChild(remove);

    container.appendChild(row);

  }



  function addLorebook() {

    addLorebookEntry(lorebookListEl || document.getElementById('lorebook_list'), {});

    markDirty();

  }



  function removeSelected(containerId) {

    const container = document.getElementById(containerId);

    if (!container) return;

    container.querySelectorAll('.row-select:checked').forEach((checkbox) => {

      const row = checkbox.closest('.list-item, .dialogue-row, .pair-row, .lore-row');

      if (row) row.remove();

    });

    markDirty();

  }



  function convertMemoryToLorebook() {

    const memories = collectList(memoryListEl);

    if (!memories.length) {

      setStatus('No memory anchors to convert.', 'error');

      return;

    }

    memories.forEach((item) => {

      addLorebookEntry(lorebookListEl, { key: item.slice(0, 40), content: item, enabled: true });

    });

    setStatus('Memory converted to lorebook entries.', 'ok');

    markDirty();

  }



  function enableDrag(container) {

    if (!container) return;

    container.addEventListener('dragover', (event) => {

      event.preventDefault();

      const dragging = container.querySelector('.dragging');

      if (!dragging) return;

      const after = getDragAfterElement(container, event.clientY);

      if (!after) {

        container.appendChild(dragging);

      } else {

        container.insertBefore(dragging, after);

      }

    });

  }



  function getDragAfterElement(container, y) {

    const elements = [...container.querySelectorAll('.list-item, .dialogue-row, .pair-row, .lore-row')].filter((el) => !el.classList.contains('dragging'));

    return elements.reduce((closest, child) => {

      const box = child.getBoundingClientRect();

      const offset = y - box.top - box.height / 2;

      if (offset < 0 && offset > closest.offset) {

        return { offset, element: child };

      }

      return closest;

    }, { offset: Number.NEGATIVE_INFINITY, element: null }).element;

  }



  function collectList(container) {

    return Array.from(container.querySelectorAll('textarea'))

      .map((el) => el.value.trim())

      .filter(Boolean);

  }



  function useFirstMessageAsPrimary() {

    const messages = collectList(firstMessagesEl);

    if (!messages.length) {

      setStatus('Add a first message first.', 'error');

      return;

    }

    const input = document.getElementById('primary_first_message');

    if (input) {

      input.value = messages[0];

      markDirty();

      setStatus('Primary greeting set.', 'ok');

      updatePrimarySelect();

    }

  }



  function updatePrimarySelect() {

    if (!primarySelectEl) return;

    const messages = collectList(firstMessagesEl);

    const current = (document.getElementById('primary_first_message')?.value || '').trim();

    primarySelectEl.innerHTML = '';

    const blank = document.createElement('option');

    blank.value = '';

    blank.textContent = 'Select from greetings...';

    primarySelectEl.appendChild(blank);

    messages.forEach((msg) => {

      const opt = document.createElement('option');

      opt.value = msg;

      opt.textContent = msg.slice(0, 60);

      if (msg === current) opt.selected = true;

      primarySelectEl.appendChild(opt);

    });

    if (current && !messages.includes(current)) {

      const opt = document.createElement('option');

      opt.value = current;

      opt.textContent = 'Custom greeting';

      opt.selected = true;

      primarySelectEl.appendChild(opt);

    }

    primarySelectEl.onchange = () => {

      const val = primarySelectEl.value;

      if (val) {

        const input = document.getElementById('primary_first_message');

        if (input) {

          input.value = val;

          markDirty();

        }

      }

    };

  }



  function estimateTokens(text) {

    const clean = String(text || '');

    return Math.ceil(clean.length / 4);

  }



  function initTokenCounters() {

    const fields = [

      'description',

      'personality',

      'current_scenario',

      'system_prompt',

      'post_history_instructions',

      'rules',

      'world_lore',

      'author_notes',

      'memory_notes',

    ];

    fields.forEach((field) => {

      const el = document.querySelector(`[data-field="${field}"]`);

      if (!el) return;

      const wrap = el.parentElement;

      if (!wrap) return;

      const counter = document.createElement('div');

      counter.className = 'token-counter';

      counter.dataset.tokenField = field;

      wrap.appendChild(counter);

    });

  }



  function tokenClass(value) {

    if (value >= 128000) return 'token-counter danger';

    if (value >= 32000) return 'token-counter warn';

    if (value >= 8000) return 'token-counter caution';

    return 'token-counter';

  }



  function updateTokenCounters() {

    document.querySelectorAll('.token-counter[data-token-field]').forEach((counter) => {

      const field = counter.dataset.tokenField;

      const el = document.querySelector(`[data-field="${field}"]`);

      if (!el) return;

      const tokens = estimateTokens(el.value);

      counter.className = tokenClass(tokens);

      counter.textContent = `${tokens} tokens`;

    });

    const firstTokens = estimateTokens(collectList(firstMessagesEl).join('\\n'));

    const scenarioTokens = estimateTokens(collectList(scenarioListEl).join('\\n'));

    const dialogueTokens = estimateTokens(collectDialogues(dialogueListEl).map((p) => `${p.user}\\n${p.bot}`).join('\\n'));

    const memoryTokens = estimateTokens(collectList(memoryListEl).join('\\n'));

    const setCounter = (id, value) => {

      const el = document.getElementById(id);

      if (!el) return;

      el.className = tokenClass(value);

      el.textContent = `${value} tokens`;

    };

    setCounter('first_messages_counter', firstTokens);

    setCounter('scenario_counter', scenarioTokens);

    setCounter('dialogue_counter', dialogueTokens);

    setCounter('memory_counter', memoryTokens);



    if (tokenTotalEl) {

      const total = firstTokens + scenarioTokens + dialogueTokens + memoryTokens + Array.from(document.querySelectorAll('.token-counter[data-token-field]')).reduce((acc, counter) => {

        const value = parseInt(counter.textContent, 10) || 0;

        return acc + value;

      }, 0);

      tokenTotalEl.className = tokenClass(total);

      tokenTotalEl.textContent = `Estimated total tokens: ${total}`;

    }

  }



  function ratingToTag(rating) {

    const key = String(rating || '').trim().toLowerCase();

    const map = {

      sfw: 'SFW',

      suggestive: 'Suggestive',

      nsfw: 'NSFW',

      extreme: 'Extreme',

      unrated: 'Unrated',

    };

    return map[key] || (key ? key.toUpperCase() : '');

  }



  function mergeTagsWithRating(tags, rating) {

    const cleaned = (tags || []).map((t) => String(t).trim()).filter(Boolean);

    const tag = ratingToTag(rating);

    if (tag && !cleaned.includes(tag)) cleaned.push(tag);

    return cleaned;

  }



  function buildExportDescription(fields) {

    const parts = [];

    if (fields.description) parts.push(fields.description);

    const basics = [];

    if (fields.age) basics.push(`Age: ${fields.age}`);

    if (fields.species) basics.push(`Species: ${fields.species}`);

    if (fields.gender) basics.push(`Gender: ${fields.gender}`);

    if (fields.pronouns) basics.push(`Pronouns: ${fields.pronouns}`);

    if (fields.occupation) basics.push(`Occupation: ${fields.occupation}`);

    if (basics.length) parts.push(basics.join(', '));

    if (fields.appearance) parts.push(`Appearance: ${fields.appearance}`);

    if (fields.distinguishing_features) parts.push(`Distinguishing features: ${fields.distinguishing_features}`);

    if (fields.backstory) parts.push(`Backstory: ${fields.backstory}`);

    return parts.filter(Boolean).join('\\n\\n');

  }



  function buildExportPersonality(fields) {

    const parts = [];

    if (fields.personality) parts.push(fields.personality);

    if (fields.voice) parts.push(`Voice: ${fields.voice}`);

    if (fields.speech_style) parts.push(`Speech style: ${fields.speech_style}`);

    if (fields.mannerisms) parts.push(`Mannerisms: ${fields.mannerisms}`);

    if (fields.catchphrases) parts.push(`Catchphrases: ${fields.catchphrases}`);

    if (fields.values) parts.push(`Values: ${fields.values}`);

    if (fields.likes) parts.push(`Likes: ${fields.likes}`);

    if (fields.dislikes) parts.push(`Dislikes: ${fields.dislikes}`);

    if (fields.flaws) parts.push(`Flaws: ${fields.flaws}`);

    return parts.filter(Boolean).join('\\n\\n');

  }



  function buildExportScenario(fields) {

    const parts = [];

    if (fields.current_scenario) parts.push(fields.current_scenario);

    if (fields.setting) parts.push(`Setting: ${fields.setting}`);

    if (fields.relationship) parts.push(`Relationship to user: ${fields.relationship}`);

    if (fields.world_lore) parts.push(`World lore: ${fields.world_lore}`);

    if (fields.goals) parts.push(`Goals: ${fields.goals}`);

    if (fields.motivations) parts.push(`Motivations: ${fields.motivations}`);

    return parts.filter(Boolean).join('\\n\\n');

  }



  function buildMesExample(dialogues) {

    const lines = [];

    (dialogues || []).forEach((pair) => {

      if (pair.user) lines.push(`{{user}}: ${pair.user}`);

      if (pair.bot) lines.push(`{{char}}: ${pair.bot}`);

    });

    return lines.join('\\n');

  }



  function buildSystemPrompt(fields) {

    const parts = [];

    if (fields.system_prompt) parts.push(fields.system_prompt);

    if (fields.rules) parts.push(`Rules: ${fields.rules}`);

    if (fields.consent_rules) parts.push(`Consent rules: ${fields.consent_rules}`);

    if (fields.boundaries) parts.push(`Boundaries: ${fields.boundaries}`);

    if (fields.limits) parts.push(`Limits: ${fields.limits}`);

    return parts.filter(Boolean).join('\\n\\n');

  }



  function splitGreetings(fields, greetings) {

    const primary = (fields.primary_first_message || '').trim() || (greetings[0] || '');

    const alternates = greetings.filter((g) => g !== primary);

    return { primary, alternates };

  }



  function updatePreview() {

    if (!previewOutput) return;

    const payload = collectPayload();

    const fields = payload.fields;

    const greetings = payload.first_messages || [];

    const { primary, alternates } = splitGreetings(fields, greetings);

    if (state.previewMode === 'card') {

      const card = {

        spec: 'chara_card_v2',

        spec_version: '2.0',

        data: {

          name: fields.name || '',

          description: buildExportDescription(fields),

          personality: buildExportPersonality(fields),

          scenario: buildExportScenario(fields),

          first_mes: primary,

          mes_example: buildMesExample(payload.example_dialogues),

          creator_notes: fields.author_notes || '',

          system_prompt: buildSystemPrompt(fields),

          post_history_instructions: fields.post_history_instructions || '',

          alternate_greetings: alternates,

          tags: mergeTagsWithRating(payload.tags, fields.rating),

          creator: fields.creator || '',

          character_version: fields.character_version || '',

          extensions: {

            botmaker: {

              fields,

              toggles: payload.toggles,

              scenarios: payload.scenarios,

              memory: payload.memory,

              first_messages: payload.first_messages,

              example_dialogues: payload.example_dialogues,

              prompt_pairs: payload.prompt_pairs,

              lorebook: payload.lorebook,

            },

          },

        },

      };

      previewOutput.textContent = JSON.stringify(card, null, 2);

      return;

    }

    if (state.previewMode === 'janitor') {

      const janitor = {

        name: fields.name || '',

        description: buildExportDescription(fields),

        personality: buildExportPersonality(fields),

        scenario: buildExportScenario(fields),

        first_message: primary,

        alternate_greetings: alternates,

        example_messages: buildMesExample(payload.example_dialogues),

        tags: mergeTagsWithRating(payload.tags, fields.rating),

        creator_notes: fields.author_notes || '',

        system_prompt: buildSystemPrompt(fields),

        post_history_instructions: fields.post_history_instructions || '',

        rules: fields.rules || '',

        world_lore: fields.world_lore || '',

      };

      previewOutput.textContent = JSON.stringify(janitor, null, 2);

      return;

    }

    if (state.previewMode === 'system') {

      previewOutput.textContent = buildSystemPrompt(fields);

    }

  }



  function setPreviewMode(mode) {

    state.previewMode = mode;

    updatePreview();

  }



  function applyImportedData(data) {

    if (!data) return;

    const fields = data.fields || {};

    Object.keys(fields).forEach((key) => {

      const el = document.querySelector(`[data-field="${key}"]`);

      if (el && fields[key] !== undefined && fields[key] !== null) {

        el.value = fields[key];

      }

    });

    if (Array.isArray(data.images)) {

      state.images = data.images.slice(0, MAX_IMAGES);

      renderImages();

    }

    if (Array.isArray(data.tags)) {

      state.tags = data.tags.filter(Boolean);

      renderTags();

    }

    if (Array.isArray(data.toggles)) {

      document.querySelectorAll('input[name="toggle"]').forEach((el) => {

        el.checked = data.toggles.includes(el.value);

      });

    }

    if (data.gen_sections && typeof data.gen_sections === 'object') {

      document.querySelectorAll('input[name="gen_section"]').forEach((el) => {

        if (Object.prototype.hasOwnProperty.call(data.gen_sections, el.value)) {

          el.checked = Boolean(data.gen_sections[el.value]);

        }

      });

    }

    const resetList = (container) => {

      if (container) container.innerHTML = '';

    };

    resetList(firstMessagesEl);

    (data.first_messages || []).forEach((msg) => addListItem(firstMessagesEl, msg));

    resetList(scenarioListEl);

    (data.scenarios || []).forEach((item) => addListItem(scenarioListEl, item));

    resetList(dialogueListEl);

    (data.example_dialogues || []).forEach((item) => addDialogueRow(dialogueListEl, item.user, item.bot));

    resetList(pairListEl);

    (data.prompt_pairs || []).forEach((item) => addPairRow(pairListEl, item.user, item.bot));

    resetList(memoryListEl);

    (data.memory || []).forEach((item) => addListItem(memoryListEl, item));

    resetList(lorebookListEl);

    (data.lorebook || []).forEach((item) => addLorebookEntry(lorebookListEl, item));

    updatePrimarySelect();

    refreshTokenTargets();

    markDirty();

  }



  function parseMesExample(text) {

    const examples = [];

    if (!text) return examples;

    text.split('\\n').forEach((line) => {

      if (line.startsWith('{{user}}:')) {

        examples.push({ user: line.replace('{{user}}:', '').trim(), bot: '' });

      } else if (line.startsWith('{{char}}:')) {

        const last = examples[examples.length - 1];

        if (last) {

          last.bot = line.replace('{{char}}:', '').trim();

        } else {

          examples.push({ user: '', bot: line.replace('{{char}}:', '').trim() });

        }

      }

    });

    return examples.filter((pair) => pair.user || pair.bot);

  }



  function parseCardV2(json) {

    const data = json.data || {};

    const fields = {

      name: data.name || '',

      description: data.description || '',

      personality: data.personality || '',

      current_scenario: data.scenario || '',

      primary_first_message: data.first_mes || '',

      author_notes: data.creator_notes || '',

      system_prompt: data.system_prompt || '',

      post_history_instructions: data.post_history_instructions || '',

      creator: data.creator || '',

      character_version: data.character_version || '',

    };

    const examples = parseMesExample(data.mes_example);

    return {

      fields,

      tags: data.tags || [],

      first_messages: [data.first_mes || ''].filter(Boolean).concat(data.alternate_greetings || []),

      example_dialogues: examples,

      lorebook: (data.extensions && data.extensions.botmaker && data.extensions.botmaker.lorebook) || [],

    };

  }



  function parseJanitor(json) {

    const fields = {

      name: json.name || '',

      description: json.description || '',

      personality: json.personality || '',

      current_scenario: json.scenario || '',

      primary_first_message: json.first_message || '',

      author_notes: json.creator_notes || '',

      system_prompt: json.system_prompt || '',

      post_history_instructions: json.post_history_instructions || '',

      rules: json.rules || '',

      world_lore: json.world_lore || '',

    };

    return {

      fields,

      tags: json.tags || [],

      first_messages: [json.first_message || ''].filter(Boolean).concat(json.alternate_greetings || []),

    };

  }



  function parseRisu(json) {

    const base = json.character || json.data || json;

    const fields = {

      name: base.name || base.char_name || base.character_name || base.display_name || '',

      description: base.description || base.character_info || base.char_description || base.persona || '',

      personality: base.personality || base.character_personality || '',

      current_scenario: base.scenario || base.current_scenario || base.world_scenario || '',

      primary_first_message: base.first_mes || base.first_message || base.greeting || '',

      author_notes: base.creator_notes || base.author_notes || '',

      system_prompt: base.system_prompt || base.prompt || '',

      post_history_instructions: base.post_history_instructions || '',

      creator: base.creator || '',

      character_version: base.character_version || base.version || '',

    };

    const greetings = [];

    const first = base.first_mes || base.first_message || base.greeting || '';

    if (first) greetings.push(first);

    const alternates = base.alternate_greetings || base.greetings || [];

    if (Array.isArray(alternates)) {

      alternates.forEach((item) => {

        if (item) greetings.push(item);

      });

    }

    return {

      fields,

      tags: base.tags || json.tags || [],

      first_messages: greetings.filter(Boolean),

      example_dialogues: parseMesExample(base.mes_example || base.example_dialogue || ''),

    };

  }



  function parsePlaintext(text) {

    const fields = {};

    const lines = text.split(/\\r?\\n/);

    lines.forEach((line) => {

      const match = line.match(/^([A-Za-z ]+):\\s*(.*)$/);

      if (!match) return;

      const key = match[1].trim().toLowerCase();

      const value = match[2].trim();

      const map = {

        name: 'name',

        description: 'description',

        personality: 'personality',

        scenario: 'current_scenario',

        'system prompt': 'system_prompt',

        rules: 'rules',

        'post history instructions': 'post_history_instructions',

        'first message': 'primary_first_message',

      };

      if (map[key]) {

        fields[map[key]] = value;

      }

    });

    if (!fields.system_prompt) {

      fields.system_prompt = text.slice(0, 4000);

    }

    return { fields };

  }



  function decodeTextFromPng(arrayBuffer) {

    const data = new DataView(arrayBuffer);

    const bytes = new Uint8Array(arrayBuffer);

    const signature = [137, 80, 78, 71, 13, 10, 26, 10];

    for (let i = 0; i < signature.length; i += 1) {

      if (bytes[i] !== signature[i]) return null;

    }

    let offset = 8;

    const textChunks = [];

    while (offset < bytes.length) {

      const length = data.getUint32(offset);

      offset += 4;

      const type = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);

      offset += 4;

      const chunkData = bytes.slice(offset, offset + length);

      offset += length + 4; // skip CRC

      if (type === 'tEXt') {

        const text = new TextDecoder('latin1').decode(chunkData);

        textChunks.push(text);

      }

    }

    return textChunks;

  }



  async function handleImportFile(file) {

    const ext = (file.name.split('.').pop() || '').toLowerCase();

    let imported = false;

    try {

      if (ext === 'json') {

        const text = await file.text();

        const json = JSON.parse(text);

        if (json.spec === 'chara_card_v2' || (json.data && (json.data.name || json.data.description))) {

          applyImportedData(parseCardV2(json));

          imported = true;

        } else if (json.type === 'character' || json.character || json.risu || json.risu_version) {

          applyImportedData(parseRisu(json));

          imported = true;

        } else if (json.name || json.description) {

          applyImportedData(parseJanitor(json));

          imported = true;

        } else {

          setStatus('Unknown JSON format.', 'error');

        }

      } else if (ext === 'txt') {

        const text = await file.text();

        applyImportedData(parsePlaintext(text));

        imported = true;

      } else if (ext === 'png') {

        const buf = await file.arrayBuffer();

        const textChunks = decodeTextFromPng(buf) || [];

        let found = null;

        textChunks.forEach((chunk) => {

          const parts = chunk.split('\\0');

          const keyword = parts[0];

          const value = parts.slice(1).join('\\0');

          if (keyword === 'chara' || keyword === 'chara_card_v2') {

            try {

              const decoded = atob(value);

              found = JSON.parse(decoded);

            } catch (err) {

              // ignore

            }

          }

        });

        if (found) {

          applyImportedData(parseCardV2(found));

          imported = true;

        } else {

          setStatus('No character data found in PNG.', 'error');

        }

      } else {

        setStatus('Unsupported file type.', 'error');

      }

    } catch (err) {

      setStatus('Import failed.', 'error');

    }

    if (imported) {

      setStatus('Import complete.', 'ok');

    }

  }



  async function importFile() {

    if (!importFileEl || !importFileEl.files.length) {

      setStatus('Choose a file to import.', 'error');

      return;

    }

    const file = importFileEl.files[0];

    await handleImportFile(file);

    importFileEl.value = '';

  }



  function initImportDrop() {

    if (!importDropEl || !importFileEl) return;

    const highlight = () => importDropEl.classList.add('drag');

    const unhighlight = () => importDropEl.classList.remove('drag');

    ['dragenter', 'dragover'].forEach((event) => {

      importDropEl.addEventListener(event, (ev) => {

        ev.preventDefault();

        highlight();

      });

    });

    ['dragleave', 'drop'].forEach((event) => {

      importDropEl.addEventListener(event, (ev) => {

        ev.preventDefault();

        unhighlight();

      });

    });

    importDropEl.addEventListener('drop', (event) => {

      const file = event.dataTransfer?.files?.[0];

      if (file) {

        handleImportFile(file);

      }

    });

    importDropEl.addEventListener('click', () => importFileEl.click());

  }



  function collectDialogues(container) {

    return Array.from(container.querySelectorAll('.dialogue-row')).map((row) => {

      const fields = row.querySelectorAll('textarea');

      return { user: fields[0].value.trim(), bot: fields[1].value.trim() };

    }).filter((pair) => pair.user || pair.bot);

  }



  function collectPairs(container) {

    return Array.from(container.querySelectorAll('.pair-row')).map((row) => {

      const fields = row.querySelectorAll('textarea');

      return { user: fields[0].value.trim(), bot: fields[1].value.trim() };

    }).filter((pair) => pair.user || pair.bot);

  }



  function collectLorebook(container) {

    if (!container) return [];

    return Array.from(container.querySelectorAll('.lore-row')).map((row) => {

      const fields = row.querySelectorAll('textarea');

      const enabled = row.querySelector('.lore-enabled');

      return {

        key: fields[0].value.trim(),

        content: fields[1].value.trim(),

        enabled: enabled ? enabled.checked : true,

      };

    }).filter((entry) => entry.key || entry.content);

  }



  function collectFields() {

    const fields = {};

    document.querySelectorAll('[data-field]').forEach((el) => {

      const key = el.getAttribute('data-field');

      if (!key) return;

      fields[key] = el.value;

    });

    return fields;

  }



  function collectToggles() {

    return Array.from(document.querySelectorAll('input[name="toggle"]:checked'))

      .map((el) => el.value);

  }



  function collectGenSections() {

    const sections = {};

    document.querySelectorAll('input[name="gen_section"]').forEach((el) => {

      sections[el.value] = el.checked;

    });

    return sections;

  }



  function setAllGenSections(value) {

    document.querySelectorAll('input[name="gen_section"]').forEach((el) => {

      el.checked = value;

    });

    markDirty();

  }



  function hasEnabledSections(sections) {

    return Object.values(sections).some((value) => value);

  }



  function collectPayload() {

    const fields = collectFields();

    return {

      fields,

      tags: state.tags,

      toggles: collectToggles(),

      allow_emojis: allowEmojisEl ? allowEmojisEl.checked : false,

      gen_sections: collectGenSections(),

      first_messages: collectList(firstMessagesEl),

      scenarios: collectList(scenarioListEl),

      example_dialogues: collectDialogues(dialogueListEl),

      prompt_pairs: collectPairs(pairListEl),

      memory: collectList(memoryListEl),

      lorebook: collectLorebook(lorebookListEl),

      images: state.images,

    };

  }



  function historyStorageKey() {

    return `botmaker_history_${BOT.id}`;

  }



  function loadHistory() {

    try {

      return JSON.parse(localStorage.getItem(historyStorageKey()) || '[]');

    } catch (err) {

      return [];

    }

  }



  function saveHistory(history) {

    localStorage.setItem(historyStorageKey(), JSON.stringify(history));

  }



  function formatHistoryLabel(entry) {

    const stamp = new Date(entry.ts);

    const time = stamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const date = stamp.toLocaleDateString();

    const kind = entry.kind ? String(entry.kind).toUpperCase() : '';

    return kind ? `${date} ${time} (${kind})` : `${date} ${time}`;

  }



  function renderHistorySelect() {

    if (!historySelectEl) return;

    const history = loadHistory();

    historySelectEl.innerHTML = '';

    if (!history.length) {

      const opt = document.createElement('option');

      opt.value = '';

      opt.textContent = 'No history yet';

      historySelectEl.appendChild(opt);

      return;

    }

    for (let i = history.length - 1; i >= 0; i -= 1) {

      const entry = history[i];

      const opt = document.createElement('option');

      opt.value = String(i);

      opt.textContent = formatHistoryLabel(entry);

      historySelectEl.appendChild(opt);

    }

  }



  function updateHistoryMeta() {

    if (!historyMetaEl) return;

    const history = loadHistory();

    const last = history[history.length - 1];

    const lastLabel = last ? formatHistoryLabel(last) : 'Never';

    const autosave = autosaveSecondsEl ? autosaveSecondsEl.value : '30';

    historyMetaEl.textContent = `Autosave every ${autosave}s. Versions: ${history.length}. Last saved: ${lastLabel}.`;

  }



  function saveHistorySnapshot(payload, kind) {

    const history = loadHistory();

    const hash = JSON.stringify(payload);

    if (history.length && history[history.length - 1].hash === hash) {

      updateHistoryMeta();

      return;

    }

    history.push({

      ts: new Date().toISOString(),

      kind: kind || 'manual',

      hash,

      payload,

    });

    while (history.length > 30) {

      history.shift();

    }

    saveHistory(history);

    renderHistorySelect();

    updateHistoryMeta();

  }



  function updateSaveStatusFromHistory() {

    if (!saveStatusEl) return;

    const history = loadHistory();

    const last = history[history.length - 1];

    if (!last) {

      setSaveStatus('Autosave ready.', null);

      return;

    }

    const stamp = new Date(last.ts);

    const time = formatTime(stamp);

    const kind = String(last.kind || '').toLowerCase();

    if (kind === 'auto') {

      setSaveStatus(`Autosaved at ${time}.`, 'ok');

    } else if (kind === 'manual') {

      setSaveStatus(`Saved at ${time}.`, 'ok');

    } else {

      setSaveStatus('Autosave ready.', null);

    }

  }



  function seedHistory() {

    const history = loadHistory();

    if (history.length) {

      renderHistorySelect();

      updateHistoryMeta();

      updateSaveStatusFromHistory();

      return;

    }

    saveHistorySnapshot(collectPayload(), 'seed');

    updateSaveStatusFromHistory();

  }



  function restoreHistory() {

    const history = loadHistory();

    const idx = parseInt(historySelectEl?.value || '-1', 10);

    if (idx < 0 || !history[idx]) {

      setStatus('Select a history entry.', 'error');

      return;

    }

    if (!confirm('Restore this snapshot? Current changes will be replaced.')) {

      return;

    }

    applyImportedData(history[idx].payload);

    setStatus('History snapshot restored.', 'ok');

  }



  function deleteHistory() {

    const history = loadHistory();

    const idx = parseInt(historySelectEl?.value || '-1', 10);

    if (idx < 0 || !history[idx]) {

      setStatus('Select a history entry.', 'error');

      return;

    }

    history.splice(idx, 1);

    saveHistory(history);

    renderHistorySelect();

    updateHistoryMeta();

    setStatus('History entry deleted.', 'ok');

  }



  function clearHistory() {

    if (!confirm('Clear all local history entries?')) {

      return;

    }

    localStorage.removeItem(historyStorageKey());

    renderHistorySelect();

    updateHistoryMeta();

    setStatus('History cleared.', 'ok');

  }



  function scheduleAutosave() {

    if (!autosaveSecondsEl) return;

    const raw = parseInt(autosaveSecondsEl.value || '30', 10);

    const seconds = Math.max(10, Math.min(300, Number.isNaN(raw) ? 30 : raw));

    autosaveSecondsEl.value = seconds;

    if (autosaveTimer) {

      clearInterval(autosaveTimer);

    }

    autosaveTimer = setInterval(() => {

      if (state.dirty) {

        saveBot('auto');

      }

    }, seconds * 1000);

    updateHistoryMeta();

  }



  function initAutosave() {

    if (!autosaveSecondsEl) return;

    autosaveSecondsEl.addEventListener('change', scheduleAutosave);

    scheduleAutosave();

  }



  async function saveBot(source = 'manual') {

    if (state.saving) return;

    state.saving = true;

    const payload = collectPayload();

    const payloadHash = JSON.stringify(payload);

    const savingLabel = source === 'auto' ? 'Autosaving...' : 'Saving...';

    setSaveStatus(savingLabel, null);

    try {

      const res = await fetch(`/api/bot/${BOT.id}`, {

        method: 'POST',

        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify(payload),

      });

      const data = await res.json();

      if (!res.ok) {

        setSaveStatus(data.error || 'Save failed', 'error');

        return;

      }

      saveHistorySnapshot(payload, source);

      const currentHash = JSON.stringify(collectPayload());

      state.dirty = payloadHash !== currentHash;

      if (state.dirty) {

        setSaveStatus('Saved. New changes pending.', null);

      } else {

        const label = source === 'auto' ? 'Autosaved' : 'Saved';

        setSaveStatus(`${label} at ${formatTime(new Date())}.`, 'ok');

      }

    } finally {

      state.saving = false;

    }

  }



  async function saveSettings() {

    const payload = {

      provider: document.getElementById('ai_provider').value,

      model: document.getElementById('ai_model').value,

      use_images: document.getElementById('ai_use_images').value === 'true',

      api_key: document.getElementById('ai_key').value,

      creator_name: document.getElementById('creator_name').value,

      base_url: document.getElementById('ai_base_url').value,

      temperature: parseFloat(document.getElementById('ai_temp').value || '0.7'),

      max_tokens: parseInt(document.getElementById('ai_tokens').value || '1200', 10),

      autosave_seconds: parseInt(document.getElementById('autosave_seconds').value || '30', 10),

    };

    const res = await fetch('/api/settings', {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    if (!res.ok) {

      setStatus('Failed to save settings', 'error');

      return;

    }

    setStatus('Settings saved.', 'ok');

    scheduleAutosave();

  }



  function formatTime(date) {

    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  }



  function incrementVersion() {

    const input = document.getElementById('character_version');

    if (!input) return;

    const text = (input.value || '').trim();

    if (!text) {

      input.value = '1';

      markDirty();

      return;

    }

    const match = text.match(/(\\d+)(?!.*\\d)/);

    if (!match) {

      input.value = `${text} 2`;

      markDirty();

      return;

    }

    const num = parseInt(match[1], 10) + 1;

    input.value = `${text.slice(0, match.index)}${num}${text.slice(match.index + match[1].length)}`;

    markDirty();

  }



  function pickRandom(list) {

    if (!list || !list.length) return '';

    return list[Math.floor(Math.random() * list.length)];

  }



  function randomInt(min, max) {

    return Math.floor(Math.random() * (max - min + 1)) + min;

  }



  function shuffleArray(list) {

    const arr = list.slice();

    for (let i = arr.length - 1; i > 0; i -= 1) {

      const j = Math.floor(Math.random() * (i + 1));

      [arr[i], arr[j]] = [arr[j], arr[i]];

    }

    return arr;

  }



  function randomInspiration() {

    const fieldsToOverwrite = ['name', 'species', 'age', 'occupation', 'setting', 'relationship', 'personality', 'appearance', 'voice'];

    const hasExisting = fieldsToOverwrite.some((field) => {

      const el = document.querySelector(`[data-field="${field}"]`);

      return el && el.value.trim();

    });

    if (hasExisting && !confirm('Random inspiration will overwrite key fields. Continue?')) {

      return;

    }

    const toggleInputs = Array.from(document.querySelectorAll('input[name="toggle"]'));

    toggleInputs.forEach((el) => {

      el.checked = false;

    });

    const pickCount = Math.min(toggleInputs.length, randomInt(2, 5));

    shuffleArray(toggleInputs).slice(0, pickCount).forEach((el) => {

      el.checked = true;

    });

    const seeds = INSPIRATION_SEEDS;

    const fields = {

      name: `${pickRandom(seeds.firstNames)} ${pickRandom(seeds.lastNames)}`.trim(),

      species: pickRandom(seeds.species),

      age: String(randomInt(21, 60)),

      occupation: pickRandom(seeds.occupations),

      setting: pickRandom(seeds.settings),

      relationship: pickRandom(seeds.relationships),

      personality: pickRandom(seeds.personality),

      appearance: pickRandom(seeds.appearance),

      voice: pickRandom(seeds.voice),

    };

    Object.entries(fields).forEach(([key, value]) => {

      const el = document.querySelector(`[data-field="${key}"]`);

      if (el) el.value = value;

    });

    setAllGenSections(true);

    const mergeEl = document.getElementById('profile_merge');

    if (mergeEl) mergeEl.value = 'overwrite';

    markDirty();

    generateProfile();

  }



  function applyProviderDefaults(force, previousProvider) {

    const provider = document.getElementById('ai_provider').value;

    const defaults = PROVIDER_DEFAULTS[provider];

    if (!defaults) return;

    const modelInput = document.getElementById('ai_model');

    const baseInput = document.getElementById('ai_base_url');

    const previous = PROVIDER_DEFAULTS[previousProvider] || {};

    const modelValue = modelInput.value.trim();

    const baseValue = baseInput.value.trim();

    const shouldSetModel = force || !modelValue || (previous.model && modelValue === previous.model);

    const shouldSetBase = force || !baseValue || (previous.base_url && baseValue === previous.base_url);

    if (shouldSetModel) {

      modelInput.value = defaults.model;

    }

    if (shouldSetBase) {

      baseInput.value = defaults.base_url;

    }

  }



  document.getElementById('ai_provider').addEventListener('change', (event) => {

    applyProviderDefaults(false, lastProvider);

    lastProvider = event.target.value;

  });



  function setMergePreset(mode) {

    document.getElementById('profile_merge').value = mode === 'fresh' ? 'overwrite' : mode;

    document.getElementById('first_merge').value = mode === 'fresh' ? 'overwrite' : mode;

    document.getElementById('scenario_merge').value = mode === 'fresh' ? 'overwrite' : mode;

    document.getElementById('dialogue_merge').value = mode === 'fresh' ? 'overwrite' : mode;

    markDirty();

  }



  function clearGeneratedFields() {

    const enabled = collectGenSections();

    GEN_SECTIONS.forEach((section) => {

      if (!enabled[section.id]) return;

      section.fields.forEach((field) => {

        const el = document.querySelector(`[data-field="${field}"]`);

        if (el && !['response_length', 'pov', 'narration_style', 'emoji_use', 'rating', 'language'].includes(field)) {

          el.value = '';

        }

      });

    });

    if (firstMessagesEl) firstMessagesEl.innerHTML = '';

    if (scenarioListEl) scenarioListEl.innerHTML = '';

    if (dialogueListEl) dialogueListEl.innerHTML = '';

    const primaryInput = document.getElementById('primary_first_message');

    if (primaryInput) primaryInput.value = '';

  }



  function freshStart() {

    if (!confirm('Clear current fields in enabled sections and lists?')) {

      return;

    }

    clearGeneratedFields();

    markDirty();

    setMergePreset('overwrite');

    setStatus('Fresh start applied. Ready to generate.', 'ok');

  }



  function aiDecideEverything() {

    if (!confirm('AI Decide Everything will overwrite generated fields and lists. Continue?')) {

      return;

    }

    setAllGenSections(true);

    setMergePreset('overwrite');

    const greetingsCountEl = document.getElementById('gen_greetings_count');

    const scenariosCountEl = document.getElementById('gen_scenarios_count');

    const dialoguesCountEl = document.getElementById('gen_dialogues_count');

    if (greetingsCountEl) greetingsCountEl.value = '3';

    if (scenariosCountEl) scenariosCountEl.value = '5';

    if (dialoguesCountEl) dialoguesCountEl.value = '4';

    clearGeneratedFields();

    generateAll();

  }



  function setProgress(value) {

    if (!progressBar) return;

    progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;

  }



  async function generateSimple() {

    const simpleInputEl = document.getElementById('simple_input');

    const simpleInput = simpleInputEl ? simpleInputEl.value.trim() : '';

    if (!simpleInput) {

      setStatus('Add original input first.', 'error');

      return;

    }

    setStatus('Generating simple output...', null);

    const payload = collectPayload();

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_simple`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    applyFields(data.fields || {});

    firstMessagesEl.innerHTML = '';

    (data.first_messages || []).forEach((msg) => addListItem(firstMessagesEl, msg));

    scenarioListEl.innerHTML = '';

    (data.scenarios || []).forEach((item) => addListItem(scenarioListEl, item));

    dialogueListEl.innerHTML = '';

    (data.example_dialogues || []).forEach((item) => addDialogueRow(dialogueListEl, item.user, item.bot));

    const primaryInput = document.getElementById('primary_first_message');

    if (primaryInput && !primaryInput.value.trim() && (data.first_messages || []).length) {

      primaryInput.value = data.first_messages[0];

    }

    updatePrimarySelect();

    setStatus('Simple generation complete.', 'ok');

    markDirty();

  }



  async function compileBot() {

    setStatus('Compiling...', null);

    const payload = collectPayload();

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/compile`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Compile failed', 'error');

      return;

    }

    renderCompileOutputs(data.compiled || null);

    setStatus('Compile complete.', 'ok');

    document.getElementById('compile-output')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  }



  async function generateEmptyOnly() {

    setStatus('Generating empty fields...', null);

    const payload = collectPayload();

    payload.merge_mode = 'fill';

    payload.empty_only = true;

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_profile`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    applyFields(data.fields || {});

    setStatus('Empty fields filled.', 'ok');

    markDirty();

  }



  async function generateAll() {

    const payloadBase = collectPayload();

    if (!hasEnabledSections(payloadBase.gen_sections || {})) {

      setStatus('Enable at least one generation section.', 'error');

      return;

    }

    const steps = [];

    steps.push({ name: 'Profile', type: 'profile' });

    const greetingsCount = parseInt(document.getElementById('gen_greetings_count').value || '0', 10);

    const scenariosCount = parseInt(document.getElementById('gen_scenarios_count').value || '0', 10);

    const dialoguesCount = parseInt(document.getElementById('gen_dialogues_count').value || '0', 10);

    if (greetingsCount > 0) steps.push({ name: 'Greetings', type: 'greetings', count: greetingsCount });

    if (scenariosCount > 0) steps.push({ name: 'Scenarios', type: 'scenarios', count: scenariosCount });

    if (dialoguesCount > 0) steps.push({ name: 'Dialogues', type: 'dialogues', count: dialoguesCount });

    setProgress(0);

    for (let i = 0; i < steps.length; i += 1) {

      const step = steps[i];

      setStatus(`Generating ${step.name}...`, null);

      const payload = collectPayload();

      payload.notes = getGenerationNotes();

      if (step.type === 'profile') {

        payload.merge_mode = document.getElementById('profile_merge').value;

        const res = await fetch(`/api/bot/${BOT.id}/generate_profile`, {

          method: 'POST',

          headers: { 'Content-Type': 'application/json' },

          body: JSON.stringify(payload),

        });

        const data = await res.json();

        if (!res.ok) {

          setStatus(data.error || 'Generation failed', 'error');

          return;

        }

        applyFields(data.fields || {});

      } else if (step.type === 'greetings') {

        payload.merge_mode = document.getElementById('first_merge').value;

        payload.count = step.count;

        const res = await fetch(`/api/bot/${BOT.id}/generate_first_messages`, {

          method: 'POST',

          headers: { 'Content-Type': 'application/json' },

          body: JSON.stringify(payload),

        });

        const data = await res.json();

        if (!res.ok) {

          setStatus(data.error || 'Generation failed', 'error');

          return;

        }

        firstMessagesEl.innerHTML = '';

        (data.first_messages || []).forEach((msg) => addListItem(firstMessagesEl, msg));

      } else if (step.type === 'scenarios') {

        payload.merge_mode = document.getElementById('scenario_merge').value;

        payload.count = step.count;

        const res = await fetch(`/api/bot/${BOT.id}/generate_scenarios`, {

          method: 'POST',

          headers: { 'Content-Type': 'application/json' },

          body: JSON.stringify(payload),

        });

        const data = await res.json();

        if (!res.ok) {

          setStatus(data.error || 'Generation failed', 'error');

          return;

        }

        scenarioListEl.innerHTML = '';

        (data.scenarios || []).forEach((item) => addListItem(scenarioListEl, item));

      } else if (step.type === 'dialogues') {

        payload.merge_mode = document.getElementById('dialogue_merge').value;

        payload.count = step.count;

        const res = await fetch(`/api/bot/${BOT.id}/generate_dialogues`, {

          method: 'POST',

          headers: { 'Content-Type': 'application/json' },

          body: JSON.stringify(payload),

        });

        const data = await res.json();

        if (!res.ok) {

          setStatus(data.error || 'Generation failed', 'error');

          return;

        }

        dialogueListEl.innerHTML = '';

        (data.example_dialogues || []).forEach((item) => addDialogueRow(dialogueListEl, item.user, item.bot));

      }

      setProgress(((i + 1) / steps.length) * 100);

    }

    setStatus('Generate All complete.', 'ok');

    markDirty();

  }



  async function generateProfile() {

    setStatus('Generating profile...', null);

    const payload = collectPayload();

    if (!hasEnabledSections(payload.gen_sections || {})) {

      setStatus('Enable at least one generation section.', 'error');

      return;

    }

    payload.merge_mode = document.getElementById('profile_merge').value;

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_profile`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    applyFields(data.fields || {});

    setStatus('Profile generated.', 'ok');

    markDirty();

  }



  async function generateFirstMessages() {

    setStatus('Generating first messages...', null);

    const payload = collectPayload();

    payload.merge_mode = document.getElementById('first_merge').value;

    payload.count = parseInt(document.getElementById('first_count').value || '3', 10);

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_first_messages`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    firstMessagesEl.innerHTML = '';

    (data.first_messages || []).forEach((msg) => addListItem(firstMessagesEl, msg));

    const primaryInput = document.getElementById('primary_first_message');

    if (primaryInput && !primaryInput.value.trim() && (data.first_messages || []).length) {

      primaryInput.value = data.first_messages[0];

    }

    updatePrimarySelect();

    setStatus('First messages generated.', 'ok');

    markDirty();

  }



  async function generateScenarios() {

    setStatus('Generating scenarios...', null);

    const payload = collectPayload();

    payload.merge_mode = document.getElementById('scenario_merge').value;

    payload.count = parseInt(document.getElementById('scenario_count').value || '5', 10);

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_scenarios`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    scenarioListEl.innerHTML = '';

    (data.scenarios || []).forEach((item) => addListItem(scenarioListEl, item));

    setStatus('Scenarios generated.', 'ok');

    markDirty();

  }



  async function generateDialogues() {

    setStatus('Generating dialogues...', null);

    const payload = collectPayload();

    payload.merge_mode = document.getElementById('dialogue_merge').value;

    payload.count = parseInt(document.getElementById('dialogue_count').value || '4', 10);

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_dialogues`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    dialogueListEl.innerHTML = '';

    (data.example_dialogues || []).forEach((item) => addDialogueRow(dialogueListEl, item.user, item.bot));

    setStatus('Dialogues generated.', 'ok');

    markDirty();

  }



  function applyFields(fields) {

    Object.keys(fields).forEach((key) => {

      const el = document.querySelector(`[data-field="${key}"]`);

      if (el) {

        el.value = fields[key];

      }

    });

  }



  async function generateField(field) {

    setStatus(`Generating ${field}...`, null);

    const payload = collectPayload();

    payload.field = field;

    payload.merge_mode = document.getElementById('profile_merge').value;

    payload.notes = getGenerationNotes();

    const res = await fetch(`/api/bot/${BOT.id}/generate_field`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Generation failed', 'error');

      return;

    }

    const el = document.querySelector(`[data-field="${data.field}"]`);

    if (el) {

      el.value = data.value;

    }

    setStatus(`Generated ${data.field}.`, 'ok');

    markDirty();

  }



  function attachGenerateButtons() {

    document.querySelectorAll('[data-field]').forEach((input) => {

      const field = input.getAttribute('data-field');

      if (!field || !GENERATABLE_FIELDS.includes(field)) {

        return;

      }

      const wrapper = input.closest('div');

      if (!wrapper) return;

      const label = wrapper.querySelector('label');

      if (!label || label.dataset.hasGen === 'true') return;

      const btn = document.createElement('button');

      btn.type = 'button';

      btn.className = 'gen-btn';

      btn.textContent = 'gen';

      btn.onclick = () => generateField(field);

      label.dataset.hasGen = 'true';

      label.appendChild(btn);

    });

  }



  function bindDirtyEvents() {

    document.querySelectorAll('[data-field]').forEach((el) => {

      el.addEventListener('input', markDirty);

      el.addEventListener('change', markDirty);

    });

    document.querySelectorAll('input[name="toggle"]').forEach((el) => {

      el.addEventListener('change', markDirty);

    });

    document.querySelectorAll('input[name="gen_section"]').forEach((el) => {

      el.addEventListener('change', markDirty);

    });

  }



  async function sendChat() {

    const input = document.getElementById('chat_input');

    const message = input.value.trim();

    if (!message) return;

    input.value = '';

    addChatBubble('user', message);

    const payload = collectPayload();

    payload.message = message;

    payload.chat_mode = getChatMode();

    payload.chat_use_images = chatUseImagesEl ? chatUseImagesEl.checked : true;

    const res = await fetch(`/api/bot/${BOT.id}/chat`, {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify(payload),

    });

    const data = await res.json();

    if (!res.ok) {

      setStatus(data.error || 'Chat failed', 'error');

      return;

    }

    addChatBubble('bot', data.response || '');

  }



  function addChatBubble(role, text) {

    const bubble = document.createElement('div');

    bubble.className = `chat-bubble ${role}`;

    bubble.textContent = text;

    document.getElementById('chat_log').appendChild(bubble);

  }



  function addFirstMessage() { addListItem(firstMessagesEl); updatePrimarySelect(); markDirty(); }

  function addScenario() { addListItem(scenarioListEl); markDirty(); }

  function addDialogue() { addDialogueRow(dialogueListEl); markDirty(); }

  function addPair() { addPairRow(pairListEl); markDirty(); }

  function addMemory() { addListItem(memoryListEl); markDirty(); }



  function hydrate() {

    renderImages();

    renderTags();

    (BOT.first_messages || []).forEach((msg) => addListItem(firstMessagesEl, msg));

    (BOT.scenarios || []).forEach((item) => addListItem(scenarioListEl, item));

    (BOT.example_dialogues || []).forEach((item) => addDialogueRow(dialogueListEl, item.user, item.bot));

    (BOT.prompt_pairs || []).forEach((item) => addPairRow(pairListEl, item.user, item.bot));

    (BOT.memory || []).forEach((item) => addListItem(memoryListEl, item));

    (BOT.lorebook || []).forEach((item) => addLorebookEntry(lorebookListEl, item));

    updatePrimarySelect();

  }



  hydrate();

  bindDirtyEvents();

  initTheme();

  initModeToggle();

  initSearch();

  initLibraryImageViewer();

  initTagSuggestions();

  renderPresetSelect();

  attachGenerateButtons();

  initDragLists();

  initPromptTemplate();

  initPngExportToggle();

  initImportDrop();

  initAutosave();

  initQualityMode();

  initEmojiToggle();

  initTokenTargets();

  initSimpleFieldSync();

  initHelpChat();

  initOnboarding();

  initChatMode();

  initChatVision();

  initTokenCounters();

  updateTokenCounters();

  updatePreview();

  seedHistory();



  document.addEventListener('keydown', (event) => {

    if (event.key === 'Escape') {

      toggleHelp(false);

      finishOnboarding();

      closeImageViewer();

    }

    if (imageViewerModalEl && imageViewerModalEl.classList.contains('open')) {

      if (event.key === 'ArrowRight') {

        stepImageViewer(1);

      } else if (event.key === 'ArrowLeft') {

        stepImageViewer(-1);

      }

    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {

      event.preventDefault();

      saveBot();

    }

  });



  window.addEventListener('beforeunload', (event) => {

    if (!state.dirty) {

      return;

    }

    event.preventDefault();

    event.returnValue = '';

  });

</script>

</body>

</html>

"""





def main() -> None:

    ensure_dirs()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)





if __name__ == "__main__":

    main()

