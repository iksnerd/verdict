from verdict import inputs


def test_fields_named_reads_structured_and_missing_instructions():
    questions = {
        "a": {"type": "noul", "instructions": "What is in `message`?"},
        "b": {"type": "noul", "instructions": {"task": "Read `subject` and `body`"}},
        "c": {"type": "noul", "instructions": None},
        "d": {"type": "noul"},
    }
    assert inputs.fields_named(questions) == ["message", "subject", "body"]
