import httpx
import pytest

from movie_rag.generate import (
    REFUSAL_TEXT,
    STRICT_SYSTEM_PROMPT,
    GenerationUnavailable,
    GroqGenerator,
    build_messages,
    clean_answer,
    is_refusal,
    make_generator,
    mentions_post_cutoff_year,
    parse_citations,
    postprocess_answer,
)

CONTEXTS = [{"title": "Toy Story", "release_year": 1995}, {"title": "Toy Story 2", "release_year": 1999}]


def test_refusal_string_is_exactly_the_notebooks():
    assert REFUSAL_TEXT == "I don't have information about that in the provided context."
    assert f'answer exactly: "{REFUSAL_TEXT}"' in STRICT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Who directed Oppenheimer (2023)?", True),
        ("What is the plot of Dune (2021)?", True),
        ("Movies from 2018", True),
        ("Who directed Toy Story (1995)?", False),
        ("Best films of 2017", False),
        ("Who directed Inception?", False),
        ("Order 66 in 1234 steps", False),
    ],
)
def test_year_guard(query, expected):
    assert mentions_post_cutoff_year(query) is expected


def test_postprocess_normalizes_any_refusal_to_the_exact_string():
    assert postprocess_answer("Sorry. I don't have information about that in the provided context.", CONTEXTS) == (
        REFUSAL_TEXT
    )
    assert postprocess_answer("   ", CONTEXTS) == REFUSAL_TEXT


def test_postprocess_appends_a_citation_when_missing():
    assert postprocess_answer("John Lasseter directed it.", CONTEXTS) == "John Lasseter directed it. [Toy Story]"
    assert postprocess_answer("John Lasseter [Toy Story]", CONTEXTS) == "John Lasseter [Toy Story]"


def test_parse_citations():
    answer = "Finding Nemo [2003] [Finding Nemo] and [Document 1] [Finding Nemo] [Toy Story]"
    assert parse_citations(answer) == ["2003", "Finding Nemo", "Document 1", "Toy Story"]
    assert parse_citations(answer, allowed_titles=["Finding Nemo", "Toy Story"]) == ["Finding Nemo", "Toy Story"]
    assert parse_citations(REFUSAL_TEXT) == []
    assert parse_citations("") == []


def test_is_refusal_uses_the_notebook_markers():
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("I cannot answer that.")
    assert not is_refusal("The director of Toy Story is John Lasseter. [Toy Story]")


def test_messages_use_the_strict_prompt_and_context_format():
    messages = build_messages("Who directed Toy Story?", [{"title": "Toy Story", "director": "John Lasseter"}])
    assert messages[0] == {"role": "system", "content": STRICT_SYSTEM_PROMPT}
    assert messages[1]["content"].startswith("Movie context:\n[Document 1]\nTitle: Toy Story\n")
    assert messages[1]["content"].endswith("\n\nQuestion: Who directed Toy Story?\nAnswer:")


# ---------------------------------------------------------------- Groq client


def groq_with(responses, sleeps=None):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.Client(transport=httpx.MockTransport(handler))
    generator = GroqGenerator(
        api_key="gsk_test_secret",
        model="openai/gpt-oss-120b",
        client=client,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )
    return generator, calls


def ok(content="John Lasseter [Toy Story]", finish_reason="stop", **message):
    choice = {"message": {"content": content, **message}, "finish_reason": finish_reason}
    return httpx.Response(200, json={"choices": [choice]})


def test_groq_success_sends_temperature_zero_and_bearer_key():
    generator, calls = groq_with([ok()])
    assert generator.complete([{"role": "user", "content": "hi"}]) == "John Lasseter [Toy Story]"
    request = calls[0]
    body = httpx.Response(200, content=request.content).json()
    assert request.url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer gsk_test_secret"
    assert body["temperature"] == 0 and body["model"] == "openai/gpt-oss-120b"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_groq_retries_once_on_429_and_5xx(status):
    sleeps = []
    generator, calls = groq_with([httpx.Response(status, headers={"retry-after": "2"}), ok()], sleeps)
    assert generator.complete([{"role": "user", "content": "hi"}]) == "John Lasseter [Toy Story]"
    assert len(calls) == 2 and sleeps == [2.0]


def test_groq_gives_up_after_one_retry():
    generator, calls = groq_with([httpx.Response(429), httpx.Response(429)])
    with pytest.raises(GenerationUnavailable, match="after one retry"):
        generator.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 2


def test_groq_does_not_retry_client_errors():
    generator, calls = groq_with([httpx.Response(401)])
    with pytest.raises(GenerationUnavailable, match="401"):
        generator.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_groq_timeout_is_a_friendly_failure_without_the_key():
    generator, _ = groq_with([httpx.ReadTimeout("timed out")])
    with pytest.raises(GenerationUnavailable) as excinfo:
        generator.complete([{"role": "user", "content": "hi"}])
    assert "timed out" in str(excinfo.value)
    assert "gsk_test_secret" not in str(excinfo.value)


def test_groq_without_key_fails_without_calling_the_api():
    generator = GroqGenerator(api_key=None, model="m", client=httpx.Client(transport=httpx.MockTransport(None)))
    with pytest.raises(GenerationUnavailable, match="GROQ_API_KEY"):
        generator.complete([{"role": "user", "content": "hi"}])


def test_groq_uses_only_the_message_content_not_the_reasoning():
    reasoning = "Document 2 is the sequel [Toy Story 2]; the question asks about [Toy Story]."
    generator, _ = groq_with([ok("John Lasseter directed it. [Toy Story]", reasoning=reasoning)])
    assert generator.complete([{"role": "user", "content": "hi"}]) == "John Lasseter directed it. [Toy Story]"


@pytest.mark.parametrize(
    "content",
    [
        "<think>Document 2 is the sequel [Toy Story 2].</think>\nJohn Lasseter directed it. [Toy Story]",
        "<THINK>\nmaybe [Toy Story 2]?\n</THINK>John Lasseter directed it. [Toy Story]",
        "Document 2 is the sequel [Toy Story 2].</think>\n\nJohn Lasseter directed it. [Toy Story]",
    ],
)
def test_think_blocks_are_stripped_before_citations_are_parsed(content):
    generator, _ = groq_with([ok(content)])
    answer = generator.complete([{"role": "user", "content": "hi"}])
    assert answer == "John Lasseter directed it. [Toy Story]"
    assert parse_citations(answer, allowed_titles=["Toy Story", "Toy Story 2"]) == ["Toy Story"]


@pytest.mark.parametrize("content", ["", None, "<think>The answer is in Document 1 [Toy Story], so"])
def test_groq_without_answer_text_is_unavailable_not_a_silent_refusal(content):
    generator, _ = groq_with([ok(content, finish_reason="length")])
    with pytest.raises(GenerationUnavailable, match="finish_reason=length"):
        generator.complete([{"role": "user", "content": "hi"}])


def test_groq_defaults_leave_room_for_reasoning():
    from movie_rag.config import Settings

    generator = make_generator(Settings(llm_provider="groq", groq_api_key="gsk_test_secret"))
    assert generator.model == "openai/gpt-oss-120b"
    assert generator.max_tokens == 1024


# ---------------------------------------------------------------- answer clean-up


def test_fullwidth_citations_become_square_brackets_without_adjacent_duplicates():
    assert postprocess_answer("John Lasseter【Toy Story】 [Toy Story]", CONTEXTS) == "John Lasseter [Toy Story]"
    assert clean_answer("John Lasseter [Toy Story][Toy Story] [Toy Story]") == "John Lasseter [Toy Story]"
    assert parse_citations(postprocess_answer("John Lasseter【Toy Story】", CONTEXTS)) == ["Toy Story"]


def test_only_adjacent_duplicates_of_the_same_title_are_dropped():
    assert clean_answer("A [Toy Story] [Toy Story 2]") == "A [Toy Story] [Toy Story 2]"
    assert clean_answer("A [Toy Story]. B [Toy Story].") == "A [Toy Story]. B [Toy Story]."
    assert clean_answer("- A [Toy Story]\n- B [Toy Story]") == "- A [Toy Story]\n- B [Toy Story]"


def test_odd_spaces_are_normalized_and_collapsed():
    answer = "John Lasseter directed it.   [Toy Story]"
    assert clean_answer(answer) == "John Lasseter directed it. [Toy Story]"


def test_markdown_bold_is_stripped_from_list_answers():
    contexts = [{"title": "The Empire Strikes Back"}, {"title": "Return of the Jedi"}]
    answer = (
        "- **The Empire Strikes Back** – Darth Vader reveals he is Luke's father. [The Empire Strikes Back]\n"
        "- __Return of the Jedi__ – Luke confronts the Emperor. [Return of the Jedi]"
    )
    cleaned = postprocess_answer(answer, contexts)
    assert cleaned == (
        "- The Empire Strikes Back – Darth Vader reveals he is Luke's father. [The Empire Strikes Back]\n"
        "- Return of the Jedi – Luke confronts the Emperor. [Return of the Jedi]"
    )
    assert parse_citations(cleaned, [c["title"] for c in contexts]) == ["The Empire Strikes Back", "Return of the Jedi"]


def test_single_asterisks_in_titles_survive():
    assert clean_answer("Robert Altman directed it. [M*A*S*H]") == "Robert Altman directed it. [M*A*S*H]"


def test_prompt_asks_for_plain_text_bracket_citations_and_fact_title_items():
    assert "Use only plain square brackets for citations." in STRICT_SYSTEM_PROMPT
    assert "plain text, not markdown" in STRICT_SYSTEM_PROMPT
    assert 'Use "- " only to start list items' in STRICT_SYSTEM_PROMPT
    assert '"<fact> [Title]"' in STRICT_SYSTEM_PROMPT
