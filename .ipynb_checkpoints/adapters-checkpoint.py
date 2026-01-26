# adapters.py
import os
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


SENTINEL_MODEL = "concepts-default"

# Optional short-name map (same idea as your other project)
HF_MODELS_MAP: Dict[str, str] = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    # add more if you want
}


def _extract_user_text(input_payload: List[Dict[str, Any]]) -> str:
    """
    Your llm.py sends:
      input=[{"role":"user","content":[{"type":"input_text","text": "..."}]}]
    This pulls the text out in a provider-agnostic way.
    """
    try:
        content = input_payload[0].get("content", [])
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()
    except Exception:
        return str(input_payload).strip()


def _resolve_model(provider: str, requested_model: str) -> str:
    """
    If llm.py passes the sentinel model, swap it for a provider-specific model from env.
    Otherwise, keep requested_model as-is.
    """
    if (requested_model or "").strip() and requested_model != SENTINEL_MODEL:
        return requested_model

    provider = (provider or "").lower().strip()

    if provider == "openai":
        return os.getenv("OPENAI_CONCEPTS_MODEL", "gpt-5-mini-2025-08-07")
    if provider in ("anthropic", "claude"):
        return os.getenv("ANTHROPIC_CONCEPTS_MODEL", "claude-3-5-sonnet-latest")
    if provider == "gemini":
        return os.getenv("GEMINI_CONCEPTS_MODEL", "gemini-1.5-pro")

    # HF: allow LLM_MODEL or HF_CONCEPTS_MODEL
    if provider in ("hf", "huggingface", "local"):
        return os.getenv("HF_CONCEPTS_MODEL", os.getenv("LLM_MODEL", "qwen7b"))

    return requested_model or SENTINEL_MODEL


# -----------------------
# OpenAI (compatible)
# -----------------------
class _OpenAIResponses:
    def __init__(self, client, provider_name: str):
        self._client = client
        self._provider = provider_name

    async def create(self, *, model: str, instructions: str, input: List[Dict[str, Any]], **kwargs):
        model = _resolve_model(self._provider, model)
        resp = await self._client.responses.create(
            model=model,
            instructions=instructions,
            input=input,
            **kwargs,
        )
        return resp


class OpenAICompatClient:
    def __init__(self):
        from openai import AsyncOpenAI
        self._provider = "openai"
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.responses = _OpenAIResponses(self._client, self._provider)


# -----------------------
# Anthropic/Claude (compatible)
# -----------------------
class _AnthropicResponses:
    def __init__(self, client, provider_name: str):
        self._client = client
        self._provider = provider_name

    async def create(self, *, model: str, instructions: str, input: List[Dict[str, Any]], **kwargs):
        model = _resolve_model(self._provider, model)
        user_text = _extract_user_text(input)
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "800"))

        msg = await self._client.messages.create(
            model=model,
            system=instructions,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=max_tokens,
        )

        # Convert Anthropic blocks -> plain text
        text = ""
        try:
            for block in msg.content:
                t = getattr(block, "text", None)
                if t:
                    text += t
        except Exception:
            text = str(msg)

        return SimpleNamespace(output_text=text.strip())


class AnthropicCompatClient:
    def __init__(self):
        from anthropic import AsyncAnthropic
        self._provider = "anthropic"
        self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.responses = _AnthropicResponses(self._client, self._provider)


# -----------------------
# Hugging Face Local (compatible)
# -----------------------
class _HFLocalEngine:
    """
    Lazy-load a transformers text-generation pipeline once.
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipe = None
        self.tok = None

    def load(self):
        if self.pipe is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        hf_token = os.getenv("HF_TOKEN")  # optional (needed for gated models)
        self.tok = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            token=hf_token,
        )
        force_cpu = os.getenv("HF_FORCE_CPU", "0") == "1"

        if force_cpu:
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                token=hf_token,
            ).to("cpu")

            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=self.tok,
                device=-1,  # CPU
                pad_token_id=self.tok.eos_token_id,
            )
        else:
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",  # CUDA/MPS if available
                torch_dtype="auto",
                token=hf_token,
            )
            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=self.tok,
                pad_token_id=self.tok.eos_token_id,
            )


    def format_prompt(self, system: str, user: str) -> str:
        # Simple instruct template. Works for many instruct models.
        return (
            f"<|system|>\n{system}\n<|end|>\n"
            f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
        )

    def generate(self, system: str, user: str) -> str:
        self.load()
        prompt = self.format_prompt(system, user)

        max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("HF_TEMPERATURE", "0.1"))
        top_p = float(os.getenv("HF_TOP_P", "1.0"))

        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
        )[0]["generated_text"]

        # Return only assistant portion if present
        if "<|assistant|>" in out:
            return out.split("<|assistant|>", 1)[-1].strip()
        return out.strip()


class _HFResponses:
    def __init__(self, provider_name: str):
        self._provider = provider_name
        self._engine: Optional[_HFLocalEngine] = None
        self._model_id: Optional[str] = None

    def _ensure_engine(self, requested_model: str):
        resolved = _resolve_model(self._provider, requested_model)
        # Map short names -> full HF ids
        model_id = HF_MODELS_MAP.get(resolved, resolved)

        # If model changed, rebuild engine
        if self._engine is None or self._model_id != model_id:
            self._model_id = model_id
            self._engine = _HFLocalEngine(model_id=model_id)

    async def create(self, *, model: str, instructions: str, input: List[Dict[str, Any]], **kwargs):
        self._ensure_engine(model)
        user_text = _extract_user_text(input)

        # Run generation in a thread so async code doesn't block the event loop
        def _run():
            assert self._engine is not None
            return self._engine.generate(instructions, user_text)

        text = await asyncio.to_thread(_run)
        return SimpleNamespace(output_text=text.strip())


class HFCompatClient:
    def __init__(self):
        self._provider = "hf"
        self.responses = _HFResponses(self._provider)


# -----------------------
# Gemini (placeholder)
# -----------------------
# Once you pick the exact Gemini SDK you installed, we can implement this similarly.
# class GeminiCompatClient:
#     ...


def get_llm_client():
    provider = (os.getenv("LLM_PROVIDER", "openai") or "openai").lower()

    if provider == "openai":
        return OpenAICompatClient()
    if provider in ("anthropic", "claude"):
        return AnthropicCompatClient()
    if provider in ("hf", "huggingface", "local"):
        return HFCompatClient()

    raise ValueError(f"Unsupported LLM_PROVIDER={provider}")
