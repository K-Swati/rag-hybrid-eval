import math

from rag_eval import metrics as M


def test_precision_hit_mrr():
    rel = [False, True, False]
    assert M.precision_at_k(rel, 3) == pytest_approx(1 / 3)
    assert M.hit_rate_at_k(rel, 1) == 0.0
    assert M.hit_rate_at_k(rel, 2) == 1.0
    assert M.reciprocal_rank(rel) == 0.5
    assert M.reciprocal_rank([False, False]) == 0.0


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_ndcg_perfect_and_degraded():
    assert M.ndcg_at_k([True, False, False], 3) == 1.0
    assert M.ndcg_at_k([False, True, False], 3) == pytest_approx(1 / math.log2(3))
    assert M.ndcg_at_k([False, False, False], 3) == 0.0


def test_relevance_ignores_case_whitespace_backticks():
    assert M.is_relevant("Keys start with `nk_live_`  today", "NK_LIVE_")
    assert not M.is_relevant("nothing here", "nk_live_")


def test_groundedness():
    ctx = "tokens expire after 3600 seconds"
    assert M.groundedness("tokens expire after 3600 seconds", ctx) == 1.0
    assert M.groundedness("quantum teleportation engine", ctx) == 0.0
