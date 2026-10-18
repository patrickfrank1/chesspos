"""Archive planning: turn a source config into a list of work items.

A *work item* is one PGN archive (a local file or a remote URL). The
planner knows nothing about processing; it only decides which archives
need work, based on the ``archives/<archive_id>/_COMPLETE.json``
completion markers found in the dataset repo (or the dry-run output
directory).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.dataset.config import SourceConfig

COMPLETE_SUFFIX = "_COMPLETE.json"
COMPLETE_PREFIX = "archives/"


@dataclass
class WorkItem:
    """One PGN archive to process.

    ``archive_id`` is the URL/file stem without ``.pgn``/``.pgn.zst`` and
    is used as the directory name for the archive's batches and its
    completion marker.
    """

    archive_id: str
    provider: str
    path: str | None = None
    url: str | None = None


def completed_archives(files: list[str]) -> set[str]:
    """Archive ids that have a completion marker in a repo file listing."""
    return {
        f.split("/")[1]
        for f in files
        if f.startswith(COMPLETE_PREFIX) and f.endswith("/" + COMPLETE_SUFFIX)
    }


def archive_id_from_name(name: str) -> str:
    """Strip ``.pgn.zst`` / ``.pgn`` suffixes from an archive file name."""
    for suffix in (".pgn.zst", ".pgn"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class ArchivePlanner:
    """Build the list of archives to process from a SourceConfig."""

    def __init__(self, config: SourceConfig):
        self.config = config
        return

    def plan(self, completed: set[str]) -> list[WorkItem]:
        if self.config.archive_urls is not None:
            items = [
                WorkItem(
                    archive_id=archive_id_from_name(url.rsplit("/", 1)[-1]),
                    provider="lichess",
                    url=url,
                )
                for url in self.config.archive_urls
            ]
        else:
            root = Path(self.config.data_path)
            items = [
                WorkItem(
                    archive_id=archive_id_from_name(p.name),
                    provider="local",
                    path=str(p),
                )
                for p in sorted(root.iterdir())
                if p.is_file() and p.suffix in (".pgn", ".zst")
            ]
        return [item for item in items if item.archive_id not in completed]
