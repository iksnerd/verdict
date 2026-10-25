"""The GitHub issue domain: synthetic issues to fine-tune the bug / feature / question question.

It exists to test one thing (FINDINGS §41, §42): whether targeted synthetic data fixes Laya reading
GitHub questions as bugs. The real test set is NLBSE'24 (issues from react, tensorflow, vscode,
bitcoin and opencv), so generation must not use those projects, or a gain could come from their
vocabulary rather than from reading the issue. No network and no model here."""
from __future__ import annotations

import random

from verdict.domains import get_domain
from verdict.gen.generate import _state, build_prompt, plan_seeds

#: The question the real test set is scored with; training must ask exactly this.
EVAL_QUESTION = {
    "type": "choice", "instructions": "What does `issue` report or ask for?",
    "criteria": {"bug": "something is broken, crashes or behaves wrongly",
                 "feature": "a new feature or an improvement to how something works",
                 "question": "how to do something, or help understanding something"}}


def test_the_domain_is_registered_with_the_evaluated_question():
    spec = get_domain("github_issue_typing")
    assert spec.bank == {"type": EVAL_QUESTION}
    assert set(spec.domain.types) == {"bug", "feature", "question"}


def test_states_are_issues():
    spec = get_domain("github_issue_typing")
    seed = plan_seeds(spec.domain, 1, random.Random(0))[0]
    assert _state(spec.domain, seed, "Title\nbody") == {"issue": "Title\nbody"}


def test_no_test_set_project_is_a_generation_topic():
    topics = " ".join(get_domain("github_issue_typing").domain.room_topics).lower()
    for project in ("react", "tensorflow", "vscode", "vs code", "bitcoin", "opencv"):
        assert project not in topics, project


def test_the_prompt_asks_for_an_unlabelled_issue():
    spec = get_domain("github_issue_typing")
    seed = plan_seeds(spec.domain, 1, random.Random(0))[0]
    prompt = build_prompt(spec.domain, seed)
    assert "GitHub issue" in prompt and "title" in prompt.lower()


def test_the_shapes_domain_targets_questions_that_look_like_bugs():
    # FINDINGS §44: synthetic questions read as clean how-tos; real ones paste an error and ask why
    spec = get_domain("github_issue_shapes")
    assert spec.bank == {"type": EVAL_QUESTION}
    types = spec.domain.types
    assert "error" in types["question"] and "why" in types["question"]
    assert "?" in types["bug"] or "question" in types["bug"]
    topics = " ".join(spec.domain.room_topics).lower()
    for project in ("react", "tensorflow", "vscode", "vs code", "bitcoin", "opencv"):
        assert project not in topics, project
