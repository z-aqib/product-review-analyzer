from prompts.summarize import generate_summary


def test_summary_not_empty():
    out = generate_summary("This is a test review.")
    assert isinstance(out, str)
    assert len(out) > 0
