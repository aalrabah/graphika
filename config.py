# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

# outputs / chunking
OUT_DIR = os.getenv("OUT_DIR", "out")
# [JR] restored to original 8191 -- the 6000 value (see findings.txt,
# 2026-07-07/2026-07-08 entries) didn't actually buy real headroom: docling's
# HybridChunker counts MAX_TOKENS using its own default tokenizer
# (sentence-transformers/all-MiniLM-L6-v2), not the Qwen tokenizer that
# governs VLLM_MAX_MODEL_LEN at LLM-call time -- so the two numbers were
# never directly comparable. The actual fix for context overflow is
# VLLM_MAX_MODEL_LEN (adapters.py), not this chunk-size budget.
MAX_TOKENS = env_int("MAX_TOKENS", 8191)
CONCURRENCY = env_int("CONCURRENCY", 5)

# UIUC Chat (optional)
UIUC_CHAT_API_KEY = os.getenv("UIUC_CHAT_API_KEY", "")
UIUC_CHAT_URL     = os.getenv("UIUC_CHAT_URL", "https://uiuc.chat/api/chat-api/chat")
UIUC_CHAT_MODEL   = os.getenv("UIUC_CHAT_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

# OpenAI
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
OPENAI_CONCEPTS_MODEL = os.getenv("OPENAI_CONCEPTS_MODEL", "gpt-5-mini-2025-08-07")

# common convenience alias used by some scripts
LLM_MODEL = os.getenv("LLM_MODEL", OPENAI_CONCEPTS_MODEL)
