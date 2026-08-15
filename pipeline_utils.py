from __future__ import annotations

import fnmatch
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


def setup_logging(common: Dict[str, Any], stage_name: str) -> logging.Logger:
    log_dir = Path(common["logging"]["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(stage_name)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        debug = logging.FileHandler(log_dir / common["logging"]["debug_log"], encoding="utf-8")
        debug.setLevel(logging.DEBUG)
        debug.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(debug)
    return logger


def retry_kwargs(common: Dict[str, Any], op: str) -> Dict[str, Any]:
    return dict(common["retry"][op])


def retry_call(fn, *args, op_name: str, **kwargs):
    max_attempts = int(kwargs.pop("max_attempts"))
    base_delay = float(kwargs.pop("base_delay_seconds"))
    max_delay = float(kwargs.pop("max_delay_seconds"))
    jitter = float(kwargs.pop("jitter_seconds"))
    last: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            logging.getLogger("retry").warning("Attempt %d/%d for %s failed: %s", attempt, max_attempts, op_name, exc)
            if attempt == max_attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, jitter)
            time.sleep(delay)
    raise RuntimeError(f"{op_name} failed after {max_attempts} attempts") from last


def hf_token(common: Dict[str, Any]) -> str:
    env = common["huggingface"]["token_env_var"]
    token = os.environ.get(env)
    if not token:
        raise ValueError(f"Missing Hugging Face token in environment variable {env!r}")
    return token


def ensure_repo(api: HfApi, repo_id: str, token: str, common: Dict[str, Any], must_exist: bool = False) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        return
    except RepositoryNotFoundError:
        if must_exist:
            raise
    if not common["huggingface"].get("create_missing_output_repos", True):
        raise RepositoryNotFoundError(f"Dataset repository does not exist: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=bool(common["huggingface"].get("private", True)),
        exist_ok=True,
        token=token,
    )


def upload_file(api: HfApi, repo_id: str, local_path: Path, remote_path: str, token: str, common: Dict[str, Any]) -> None:
    retry_call(
        api.upload_file,
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        op_name=f"upload:{repo_id}:{remote_path}",
        **retry_kwargs(common, "upload"),
    )


def download_file(
    repo_id: str,
    filename: str,
    revision: str,
    token: str,
    common: Dict[str, Any],
    local_dir: str | Path | None = None,
) -> Path:
    return Path(
        retry_call(
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            token=token,
            local_dir=str(local_dir) if local_dir else None,
            op_name=f"download:{repo_id}:{filename}",
            **retry_kwargs(common, "download"),
        )
    )


def list_repo_files(api: HfApi, repo_id: str, token: str, common: Dict[str, Any]) -> List[str]:
    return list(
        retry_call(
            api.list_repo_files,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            op_name=f"list:{repo_id}",
            **retry_kwargs(common, "download"),
        )
    )


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    if not value:
        raise ValueError("Value cannot be converted to a non-empty slug")
    return value


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_remote_json(repo_id: str, path: str, token: str, common: Dict[str, Any]) -> Dict[str, Any]:
    local = download_file(repo_id, path, "main", token, common)
    return json.loads(local.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
