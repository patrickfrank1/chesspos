"""Minimal BDD step helpers for structuring tests as Given/When/Then.

The context managers are deliberate no-ops: they exist purely so test bodies
read as structured scenarios, in the style required by ``AGENTS.md``.
"""


class _Step:
    """No-op context manager representing a single BDD step."""

    def __init__(self, step: str, description: str) -> None:
        self._step = step
        self._description = description

    def __enter__(self) -> "_Step":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self._step} {self._description!r}"


def given(description: str) -> _Step:
    """Arrange: establish the context under test."""
    return _Step("given", description)


def when(description: str) -> _Step:
    """Act: perform the action under test."""
    return _Step("when", description)


def then(description: str) -> _Step:
    """Assert: verify the expected outcome."""
    return _Step("then", description)
