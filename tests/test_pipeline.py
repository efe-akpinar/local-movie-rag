from movie_rag.generate import REFUSAL_TEXT, UNAVAILABLE_TEXT, GenerationUnavailable
from movie_rag.pipeline import Pipeline
from movie_rag.testset import TEST_SET, answerable, unanswerable


class DownGenerator:
    name = "down"

    def complete(self, messages):
        raise GenerationUnavailable("provider down")


class BrokenRetriever:
    def retrieve(self, query, k=None):
        raise RuntimeError("index exploded")


def test_answers_with_citation_and_top5(pipeline):
    result = pipeline.answer("Who directed Toy Story?")
    assert result.retrieved_titles[0] == "Toy Story"
    assert len(result.retrieved) == 5
    assert result.citations == ["Toy Story"]
    assert not result.refused and result.error is None
    assert set(result.timings_ms) == {"retrieve", "rerank", "generate", "total"}


def test_year_guard_refuses_oppenheimer_but_still_returns_titles(pipeline):
    result = pipeline.answer("Who directed Oppenheimer (2023)?")
    assert result.answer == REFUSAL_TEXT
    assert result.refused
    assert len(result.retrieved_titles) == 5


def test_empty_query_is_a_refusal_with_no_titles(pipeline):
    for query in ("", "   ", None):
        assert pipeline.run_query(query) == {"answer": REFUSAL_TEXT, "retrieved_titles": []}


def test_run_query_contract(pipeline):
    for query in ["Who directed Inception?", "asdfgh nonsense query", "", "Bir Zamanlar Anadolu'da hakkında?"]:
        result = pipeline.run_query(query)
        assert set(result) == {"answer", "retrieved_titles"}
        assert isinstance(result["answer"], str)
        assert all(isinstance(title, str) for title in result["retrieved_titles"])


def test_run_query_never_raises(pipeline):
    broken = Pipeline(BrokenRetriever(), pipeline.generator)
    assert broken.run_query("Who directed Jaws?") == {
        "answer": f"{REFUSAL_TEXT} Internal error: RuntimeError",
        "retrieved_titles": [],
    }


def test_generation_outage_gives_a_friendly_answer(retriever):
    result = Pipeline(retriever, DownGenerator()).answer("Who directed Jaws?")
    assert result.answer == UNAVAILABLE_TEXT
    assert result.error == "generation_unavailable"
    assert not result.refused
    assert result.retrieved_titles  # retrieval still worked


def test_test_set_matches_the_notebook():
    categories = [item["category"] for item in TEST_SET]
    assert len(TEST_SET) == 20 and len(answerable()) == 16 and len(unanswerable()) == 4
    assert {c: categories.count(c) for c in set(categories)} == {
        "standard": 6,
        "adversarial": 7,
        "sequel_confusion": 3,
        "unanswerable": 4,
    }


class NanScorer:
    def predict(self, pairs, **kwargs):
        return [float("nan")] * len(pairs)


def test_nan_reranker_scores_fail_loudly_instead_of_keeping_the_bi_encoder_order(retriever, pipeline):
    import pytest

    from movie_rag.retrieve import RerankerError, Retriever

    broken = Retriever(retriever.movie_index, retriever.bi_encoder, NanScorer())
    with pytest.raises(RerankerError):
        broken.search("Who directed Toy Story?")
    with pytest.raises(RerankerError):
        broken.self_check()
    result = Pipeline(broken, pipeline.generator).run_query("Who directed Toy Story?")
    assert result == {"answer": f"{REFUSAL_TEXT} Internal error: RerankerError", "retrieved_titles": []}


def test_align_parameters_moves_only_misaligned_tensors_and_keeps_values():
    import pytest

    torch = pytest.importorskip("torch")
    from movie_rag.retrieve import align_parameters

    module = torch.nn.Linear(4, 4)
    offset = torch.arange(17, dtype=torch.float32)[1:]  # starts 4 bytes into its storage
    module.weight = torch.nn.Parameter(offset.view(4, 4))
    assert module.weight.data_ptr() % 64 != 0
    before = module.weight.detach().clone()
    moved = align_parameters(module)
    assert moved >= 1 and module.weight.data_ptr() % 64 == 0
    assert torch.equal(module.weight.detach(), before)
