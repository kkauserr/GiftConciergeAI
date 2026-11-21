import numpy as np

from CODE.ranking.filters import check_product
from CODE.ranking.fusion import rrf
from CODE.ranking.ranker import bayesian_rating, budget_fit, mmr

ROW = {"price_num": 25.0, "age_min": 8.0, "age_max": None, "domain": "Toys & Games",
       "title": "Wooden Puzzle Box", "categories": "Toys & Games > Puzzles", "specific_category": "Puzzles"}


def test_budget():
    assert check_product(ROW, {"budget_max": 30}) == []
    assert check_product(ROW, {"budget_max": 20}) == ["over_budget"]
    assert check_product(ROW, {"budget_min": 30, "budget_max": 60}) == ["under_budget"]
    assert check_product({**ROW, "price_num": None}, {"budget_max": 30}) == ["no_price"]


def test_age():
    assert check_product(ROW, {"recipient_age": 10}) == []
    assert check_product(ROW, {"recipient_age": 5}) == ["too_young"]
    assert check_product({**ROW, "age_max": 12.0}, {"recipient_age": 30}) == ["too_old"]
    assert check_product({**ROW, "age_min": None}, {"recipient_age": 2}) == []


def test_exclusions():
    assert check_product(ROW, {"exclusions": ["toys"]}) == ["excluded:toys"]
    assert check_product(ROW, {"exclusions": ["puzzles"]}) == ["excluded:puzzles"]
    assert check_product(ROW, {"exclusions": ["puzzle"]}) == ["excluded:puzzle"]
    assert check_product(ROW, {"exclusions": ["books"]}) == []
    assert check_product(ROW, {"exclusions": ["box"]}) == ["excluded:box"]
    assert check_product(ROW, {"exclusions": ["ox"]}) == []


def test_rrf_rewards_agreement():
    fused, ranks = rrf({"bm25": [("a", 9), ("b", 5), ("c", 1)], "semantic": [("d", 0.9), ("b", 0.8), ("e", 0.1)]})
    order = [a for a, _ in fused]
    assert order[0] == "b"
    assert ranks["b"] == {"bm25_rank": 2, "semantic_rank": 2}
    assert set(order) == {"a", "b", "c", "d", "e"}


def test_bayesian_rating():
    few, many = bayesian_rating([5.0, 4.7], [2, 5000], prior_mean=4.4, prior_count=20)
    assert many > few


def test_budget_fit():
    fit = budget_fit([5, 56, 100], {"budget_max": 80})
    assert fit[1] > fit[0] and fit[1] > fit[2]
    assert (budget_fit([5, 50], {}) == 1).all()


def test_mmr_skips_near_duplicate():
    scores = np.array([1.0, 0.99, 0.8])
    sim = np.array([[1, 0.95, 0.1], [0.95, 1, 0.1], [0.1, 0.1, 1]])
    assert mmr(scores, sim, k=2, lam=0.7) == [0, 2]
