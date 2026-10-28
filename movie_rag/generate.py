"""Grounded answer generation with a pluggable LLM provider (groq, local, fake).

The prompt, context format, year guard and answer post-processing are copied from the notebook
(Section 5); only the model call differs between providers.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import Protocol

import httpx

from movie_rag.config import Settings
from movie_rag.preprocess import list_to_text

logger = logging.getLogger(__name__)

REFUSAL_TEXT = "I don't have information about that in the provided context."
UNAVAILABLE_TEXT = "Sorry, the answer service is temporarily unavailable. Please try again in a minute."

STRICT_SYSTEM_PROMPT = """You are a grounded movie question-answering assistant.

Rules:
1. Answer ONLY using the provided movie context.
2. If the context does not contain enough evidence, answer exactly: "I don't have information about that in the provided context."
3. Do not use outside knowledge, even if you know the movie.
4. Cite every movie title you used in square brackets, for example [Toy Story].
5. If the user asks about a movie released after July 2017, refuse unless that exact movie appears in the context.

Refusal example:
Question: Who directed Oppenheimer (2023)?
Answer: I don't have information about that in the provided context.
"""  # noqa: E501

REFUSAL_MARKERS = [
    "don't have information",
    "do not have information",
    "i don't know",
    "insufficient",
    "not enough information",
    "cannot answer",
]

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


class GenerationUnavailable(RuntimeError):
    """The LLM could not produce an answer (provider down, rate-limited, misconfigured)."""


class Generator(Protocol):
    name: str

    def complete(self, messages: list[dict]) -> str: ...


# ---------------------------------------------------------------- prompt and post-processing


def concise_doc(record, max_overview_chars=850):
    year = record.get("release_year") or "Unknown"
    genres = list_to_text(record.get("genres", []), empty="Unknown")
    cast = list_to_text(record.get("cast_names", []), empty="Unknown")
    overview = str(record.get("overview", ""))[:max_overview_chars]
    return (
        f"Title: {record.get('title', 'Unknown')}\n"
        f"Year: {year}\n"
        f"Genres: {genres}\n"
        f"Director: {record.get('director', 'Unknown')}\n"
        f"Cast: {cast}\n"
        f"Overview: {overview}"
    )


def format_context(records, max_docs=5):
    chunks = []
    for i, record in enumerate(records[:max_docs], start=1):
        chunks.append(f"[Document {i}]\n{concise_doc(record)}")
    return "\n\n".join(chunks)


def mentions_post_cutoff_year(query: str) -> bool:
    years = [int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", str(query))]
    return any(year > 2017 for year in years)


def build_messages(query: str, contexts, system_prompt: str = STRICT_SYSTEM_PROMPT) -> list[dict]:
    user_prompt = f"Movie context:\n{format_context(contexts)}\n\nQuestion: {query}\nAnswer:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def postprocess_answer(answer: str, contexts) -> str:
    answer = (answer or "").strip()
    if not answer:
        return REFUSAL_TEXT
    if REFUSAL_TEXT.lower() in answer.lower():
        return REFUSAL_TEXT
    cited = any(f"[{record['title']}]" in answer for record in contexts)
    if contexts and not cited:
        answer = f"{answer} [{contexts[0]['title']}]"
    return answer


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning so titles mentioned while reasoning are not taken as citations.

    Also handles a block cut off by the token limit, and output that has only the closing tag because
    the chat template opened the block in the prompt.
    """
    text = _THINK_RE.sub("", text or "")
    return _THINK_CLOSE_RE.split(text)[-1].strip()


def is_refusal(answer: str) -> bool:
    lower = str(answer).lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def parse_citations(answer: str, allowed_titles=None) -> list[str]:
    """Bracketed citations in order of appearance, without duplicates.

    With ``allowed_titles``, only citations that exactly match one of those titles are kept, which
    drops brackets the model uses for other things such as ``[1999]`` or ``[Document 1]``.
    """
    allowed = set(allowed_titles) if allowed_titles is not None else None
    citations: list[str] = []
    for match in _CITATION_RE.findall(str(answer or "")):
        title = match.strip()
        if not title or title in citations:
            continue
        if allowed is not None and title not in allowed:
            continue
        citations.append(title)
    return citations


def generate_answer(generator: Generator, query: str, contexts) -> str:
    if not contexts:
        return REFUSAL_TEXT
    return postprocess_answer(generator.complete(build_messages(query, contexts)), contexts)


# ---------------------------------------------------------------- providers


class FakeGenerator:
    """Deterministic stand-in for tests and smoke checks: cites the first document in the context."""

    name = "fake"

    def complete(self, messages: list[dict]) -> str:
        match = re.search(r"^Title: (.+)$", messages[-1]["content"], flags=re.MULTILINE)
        if not match:
            return REFUSAL_TEXT
        title = match.group(1).strip()
        return f"The best match in the provided context is {title}. [{title}]"


class GroqGenerator:
    """Groq's OpenAI-compatible chat completions API, temperature 0, one retry on 429 and 5xx.

    Only ``message.content`` is used. Reasoning models return their reasoning in a separate field
    (``message.reasoning`` for gpt-oss), which is ignored, or inline in <think> tags, which are stripped.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 20.0,
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout_s)
        self._sleep = sleep

    def complete(self, messages: list[dict]) -> str:
        if not self._api_key:
            raise GenerationUnavailable("GROQ_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in (1, 2):
            try:
                response = self._client.post(self.url, json=payload, headers=headers, timeout=self.timeout_s)
            except httpx.TimeoutException as exc:
                raise GenerationUnavailable("Groq request timed out") from exc
            except httpx.HTTPError as exc:
                raise GenerationUnavailable(f"Groq request failed: {type(exc).__name__}") from exc

            status = response.status_code
            if status == 429 or status >= 500:
                if attempt == 1:
                    logger.warning("Groq returned %s, retrying once", status)
                    self._sleep(_retry_delay(response))
                    continue
                raise GenerationUnavailable(f"Groq returned {status} after one retry")
            if status >= 400:
                raise GenerationUnavailable(f"Groq returned {status}")
            try:
                choice = response.json()["choices"][0]
                content = choice["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                raise GenerationUnavailable("Groq returned an unexpected response body") from exc
            answer = strip_think(content)
            if not answer:
                # Usually the token budget ran out while the model was still reasoning. Fail loudly:
                # post-processing would otherwise turn the empty answer into an ordinary-looking refusal.
                finish_reason = choice.get("finish_reason")
                raise GenerationUnavailable(f"Groq returned no answer text (finish_reason={finish_reason})")
            return answer
        raise GenerationUnavailable("Groq request failed")  # unreachable; keeps type checkers happy


def _retry_delay(response: httpx.Response, default: float = 1.0, maximum: float = 5.0) -> float:
    try:
        return min(float(response.headers.get("retry-after", default)), maximum)
    except ValueError:
        return default


class LocalGenerator:
    """Qwen2.5-3B-Instruct with transformers (4-bit on GPU), as in the notebook. Needs a GPU to be useful."""

    name = "local"

    def __init__(self, model_name: str, cpu_fallback_model: str, max_new_tokens: int = 180):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name if torch.cuda.is_available() else cpu_fallback_model
        logger.info("Loading local generator %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.model.to("cpu")
        self.model.eval()

    def complete(self, messages: list[dict]) -> str:
        tokenizer, torch = self.tokenizer, self._torch
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3900)
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


def make_generator(settings: Settings) -> Generator:
    if settings.llm_provider == "fake":
        return FakeGenerator()
    if settings.llm_provider == "local":
        return LocalGenerator(settings.local_model, settings.local_cpu_fallback_model, settings.max_new_tokens)
    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    if not api_key:
        logger.warning("LLM_PROVIDER is groq but GROQ_API_KEY is not set; /ask will return 503")
    return GroqGenerator(
        api_key=api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
        timeout_s=settings.groq_timeout_s,
        max_tokens=settings.groq_max_tokens,
    )
