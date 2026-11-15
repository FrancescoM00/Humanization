"""Verifica la ripresa del notebook con archivi o cartelle, senza usare API."""
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "kaggle" / "Humanization_Naturalness_Panel.ipynb"
SETTINGS = {
    "mode": "study",
    "model": "qwen/qwen3.8-27b",
    "systems": ["qwen_humanize", "qwen14b_humanize", "dipper", "dipper_context3"],
    "repetitions": 3,
    "seed": 42,
    "max_sources": 126,
}


def checkpoint(directory, *, settings=None, requested_ref="main", n_replies=2):
    directory.mkdir(parents=True, exist_ok=True)
    runtime = {
        "settings": SETTINGS if settings is None else settings,
        "code_commit": "291912766541a3a2a144f6bc0dec9d0585d006b2",
        "requested_ref": requested_ref,
    }
    files = {
        "runtime_info.json": json.dumps(runtime).encode(),
        "judge_plan.json": b'{"protocol": "test_offline", "seed": 42}\n',
        "judgments.jsonl": b"".join(
            json.dumps({"request_id": f"reply-{i}", "rating": 4}).encode() + b"\n"
            for i in range(n_replies)
        ),
        "api_attempts.jsonl": b"".join(
            json.dumps({"request_id": f"attempt-{i}"}).encode() + b"\n"
            for i in range(n_replies + 1)
        ),
        "summary.json": '{"status": "partial", "note": "già salvato"}\n'.encode(),
        "inputs/rewrites.jsonl": b'{"text": "The original passage.", "status": "ok"}\n',
    }
    for name, content in files.items():
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return files


def as_zip(source, destination):
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return destination


def snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): (
            path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


def execute_resume(tmp_path, resume, *, run=None):
    """Esegue la cella reale, spostando soltanto i percorsi Kaggle in tmp_path."""
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    input_root.mkdir(exist_ok=True)
    working_root.mkdir(exist_ok=True)
    run = working_root / "run" if run is None else run
    run.mkdir(parents=True, exist_ok=True)

    def local_path(*parts):
        path = Path(*parts)
        for anchor, replacement in (
            (Path("/kaggle/input"), input_root),
            (Path("/kaggle/working"), working_root),
        ):
            if path == anchor or anchor in path.parents:
                return replacement / path.relative_to(anchor)
        return path

    namespace = {
        "MODE": SETTINGS["mode"],
        "JUDGE_MODEL": SETTINGS["model"],
        "HUMANIZERS": SETTINGS["systems"],
        "REPETITIONS": SETTINGS["repetitions"],
        "SEED": SETTINGS["seed"],
        "MAX_SOURCES": SETTINGS["max_sources"],
        "REPO_REF": "main",
        "RESUME_ZIP": str(resume),
        "RUN": run,
        "Path": local_path,
        "json": json,
        "shutil": shutil,
        "stat": stat,
        "tempfile": tempfile,
        "zipfile": zipfile,
    }
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][4]["source"])
    exec(compile(source, f"{NOTEBOOK}:cell4", "exec"), namespace)
    return run, namespace


def assert_files(directory, expected):
    assert {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    } == expected


def test_resume_explicit_zip_preserves_checkpoint_bytes(tmp_path, capsys):
    source = tmp_path / "checkpoint"
    files = checkpoint(source)
    archive = as_zip(source, tmp_path / "checkpoint.zip")

    run, namespace = execute_resume(tmp_path, archive)

    assert_files(run, files)
    assert namespace["saved_runtime"]["code_commit"] == json.loads(files["runtime_info.json"])["code_commit"]
    output = capsys.readouterr().out
    assert "Risposte già salvate: 2" in output
    assert "Tentativi già registrati: 3" in output


@pytest.mark.parametrize("use_parent", [False, True])
def test_resume_extracted_directory_does_not_change_input(tmp_path, use_parent):
    source = tmp_path / "input" / "uploaded-dataset" / "checkpoint"
    files = checkpoint(source)
    before = snapshot(tmp_path / "input")

    run, _ = execute_resume(tmp_path, source.parent if use_parent else source)

    assert_files(run, files)
    assert snapshot(tmp_path / "input") == before
    assert (run / "inputs" / "rewrites.jsonl").read_bytes() == files["inputs/rewrites.jsonl"]


def test_auto_ignores_checkpoints_with_different_settings(tmp_path):
    root = tmp_path / "input"
    checkpoint(root / "smoke", settings={**SETTINGS, "mode": "smoke", "max_sources": 3})
    checkpoint(root / "different-seed", settings={**SETTINGS, "seed": 99})
    checkpoint(root / "incomplete")
    (root / "incomplete" / "judge_plan.json").unlink()
    files = checkpoint(root / "matching" / "nested")
    before = snapshot(root)

    run, _ = execute_resume(tmp_path, "auto")

    assert_files(run, files)
    assert snapshot(root) == before


def test_auto_rejects_multiple_matching_checkpoints(tmp_path):
    checkpoint(tmp_path / "input" / "first")
    checkpoint(tmp_path / "input" / "second")

    with pytest.raises((AssertionError, ValueError)):
        execute_resume(tmp_path, "auto")

    assert_files(tmp_path / "working" / "run", {})


def test_auto_keeps_existing_run_instead_of_older_input(tmp_path):
    checkpoint(tmp_path / "input" / "older", n_replies=1)
    run = tmp_path / "working" / "run"
    files = checkpoint(run, n_replies=4)
    before = snapshot(run)

    restored, namespace = execute_resume(tmp_path, "auto", run=run)

    assert restored == run
    assert_files(run, files)
    assert snapshot(run) == before
    assert namespace["saved_runtime"]["settings"] == SETTINGS


@pytest.mark.parametrize("archive_kind", ["directory", "zip"])
@pytest.mark.parametrize("mismatch", ["settings", "requested_ref"])
def test_resume_rejects_scientific_mismatch_before_copying(tmp_path, archive_kind, mismatch):
    source = tmp_path / "input" / "checkpoint"
    options = (
        {"settings": {**SETTINGS, "repetitions": 1}}
        if mismatch == "settings" else {"requested_ref": "another-branch"}
    )
    checkpoint(source, **options)
    resume = source if archive_kind == "directory" else as_zip(source, tmp_path / "checkpoint.zip")
    before = snapshot(tmp_path / "input")

    with pytest.raises((AssertionError, ValueError)):
        execute_resume(tmp_path, resume)

    assert_files(tmp_path / "working" / "run", {})
    assert snapshot(tmp_path / "input") == before


@pytest.mark.parametrize("archive_kind", ["directory", "zip"])
def test_resume_never_overwrites_conflicting_existing_file(tmp_path, archive_kind):
    source = tmp_path / "input" / "checkpoint"
    checkpoint(source)
    resume = source if archive_kind == "directory" else as_zip(source, tmp_path / "checkpoint.zip")
    run = tmp_path / "working" / "run"
    run.mkdir(parents=True)
    conflict = run / "summary.json"
    conflict.write_bytes(b'{"status": "local-newer"}\n')
    saved_bytes = conflict.read_bytes()

    with pytest.raises((AssertionError, ValueError)):
        execute_resume(tmp_path, resume, run=run)

    assert conflict.read_bytes() == saved_bytes


def test_resume_rejects_zip_path_traversal(tmp_path):
    source = tmp_path / "checkpoint"
    checkpoint(source)
    archive = as_zip(source, tmp_path / "checkpoint.zip")
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr("../outside.json", b'{"unexpected": true}')

    with pytest.raises((AssertionError, ValueError)):
        execute_resume(tmp_path, archive)

    assert not (tmp_path / "working" / "outside.json").exists()
