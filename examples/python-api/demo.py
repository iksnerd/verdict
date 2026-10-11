"""verdict as a Python library: typed questions, answers grouped by type, one error type.

    python examples/python-api/demo.py        # asks the server `verdict serve` started

`main` takes a client, so a test can run it with no model (`Verdict(backend=UniformBackend())`).
"""
import json
from pathlib import Path

from verdict import Choice, Noul, Score, Verdict, VerdictError

TICKET = {"message": "I was charged twice this month. Please refund the duplicate today or I'm cancelling."}
TREE = Path(__file__).resolve().parent.parent / "banking-tree" / "tree.json"


def main(v: Verdict) -> None:
    # Three question types about one state; every question runs against the same text.
    result = v.decide(TICKET, {
        "refund": Noul(instructions="Does `message` ask for a refund?"),
        "dept": Choice(instructions="Which team should handle `message`?",
                       criteria={"billing": "invoices, payments, refunds",
                                 "tech": "bugs and outages", "other": "everything else"}),
        "urgent": Score(instructions="How urgent does `message` sound?",
                        criteria=["can wait", "soon", "needs action today"]),
    })
    print(f"refund   {result.nouls['refund'].noul:.2f}")
    dept = result.choices["dept"]
    print(f"dept     {dept.choice} {dept.probabilities[dept.choice]:.2f}")
    print(f"urgent   {result.scores['urgent'].score:.2f} of 2")

    # A question about a consequence measured at chance, so it is refused: a status and a code.
    try:
        v.decide(TICKET, {"risk": Noul(instructions="How dangerous is `message`?")})
    except VerdictError as err:
        print(f"refused  status {err.status}, code {err.code}")

    # More than 20 options: a two-level tree (FINDINGS §50).
    best = v.classify("My card still has not arrived and it has been two weeks",
                      json.loads(TREE.read_text()))
    print(f"classify {best.label} {best.probability:.2f} via {' > '.join(best.path)} ({best.passes} passes)")

    print(f"backend  {v.capabilities().family}, {len(v.models())} model name(s) offered")


if __name__ == "__main__":
    with Verdict() as client:
        main(client)
