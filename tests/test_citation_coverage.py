from app.generation.citation import citation_check_count, citations_cover_every_claim


def test_every_claim_must_cite_an_existing_passage():
    assert citations_cover_every_claim("Dense search ranks passages [1].\nBM25 finds exact terms [2].", 2)
    assert not citations_cover_every_claim("Dense search ranks passages [1].\nBM25 finds exact terms.", 2)
    assert not citations_cover_every_claim("The answer is 42 [3].", 2)
    assert not citations_cover_every_claim("The answer is 42 [Source 1].", 2)
    assert not citations_cover_every_claim("The documents do not say.", 2)


def test_citation_check_count_can_bound_judge_provider_calls():
    assert citation_check_count("Fact one [1].\nFact two [2][3].") == 3
