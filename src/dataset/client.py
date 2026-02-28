from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import HfApi, Repository, create_repo, repo_exists
from huggingface_hub.utils import RepositoryNotFoundError

import numpy as np


@dataclass
class HuggingFaceClient:
    repo_name: str
    local_path: str = "./hf_dataset_cache"
    token: str | None = None

    def __post_init__(self):
        self._api = HfApi()
        self._check_authentication()

    def _check_authentication(self) -> None:
        try:
            self._api.whoami()
        except Exception as e:
            raise RuntimeError(
                "Not authenticated with HuggingFace Hub. "
                "Run 'huggingface-cli login' first."
            ) from e

    def get_repository(self) -> Repository:
        Path(self.local_path).mkdir(parents=True, exist_ok=True)
        repo_local_path = os.path.join(self.local_path, self.repo_name.split("/")[-1])

        if not repo_exists(self.repo_name):
            create_repo(self.repo_name, repo_type="dataset", exist_ok=True)

        if os.path.exists(repo_local_path):
            return Repository(repo_local_path, clone_from=self.repo_name)
        else:
            return Repository(
                repo_local_path,
                clone_from=self.repo_name,
                repo_type="dataset",
            )

    def push_batch(
        self,
        data: dict[str, np.ndarray],
        split: str,
        batch_number: int,
        commit_message: str | None = None,
    ) -> str:
        dataset = Dataset.from_dict(data)
        path_in_repo = f"{split}/batch_{batch_number:04d}.parquet"

        if commit_message is None:
            commit_message = f"Add {split} batch {batch_number}"

        url = dataset.push_to_hub(
            self.repo_name,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            repo_type="dataset",
            token=self.token,
        )
        return url

    def create_version_tag(self, tag: str, tag_message: str | None = None) -> str:
        if tag_message is None:
            tag_message = f"Dataset version {tag}"

        self._api.create_tag(
            repo_id=self.repo_name,
            tag=tag,
            repo_type="dataset",
            tag_message=tag_message,
            exist_ok=True,
        )
        return f"https://huggingface.co/datasets/{self.repo_name}/tree/{tag}"

    def get_existing_batches(self, split: str = "train") -> list[int]:
        try:
            files = self._api.list_repo_files(
                repo_id=self.repo_name,
                repo_type="dataset",
            )
            batch_numbers = []
            for f in files:
                if f.startswith(f"{split}/batch_") and f.endswith(".parquet"):
                    try:
                        batch_str = f.split("batch_")[1].split(".")[0]
                        batch_numbers.append(int(batch_str))
                    except (IndexError, ValueError):
                        continue
            return sorted(batch_numbers)
        except RepositoryNotFoundError:
            return []

    def get_next_batch_number(self, split: str = "train") -> int:
        existing = self.get_existing_batches(split)
        if not existing:
            return 1
        return max(existing) + 1

    def create_dataset_card(
        self,
        description: str,
        features: dict[str, Any],
        usage_example: str | None = None,
    ) -> str:
        card_content = f"""---
license: mit
---

# {self.repo_name.split("/")[-1]}

{description}

## Dataset Schema

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
"""
        for name, info in features.items():
            shape = info.get("shape", "scalar")
            dtype = info.get("dtype", "unknown")
            desc = info.get("description", "")
            card_content += f"| {name} | {shape} | {dtype} | {desc} |\n"

        if usage_example:
            card_content += f"\n## Usage\n\n```python\n{usage_example}\n```\n"

        return card_content

    def push_dataset_card(self, content: str) -> None:
        self._api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="README.md",
            repo_id=self.repo_name,
            repo_type="dataset",
            commit_message="Update dataset card",
        )
