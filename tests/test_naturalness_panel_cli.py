"""Controlla la CLI del pannello senza collegarsi a un provider."""
import json
import sys

import pytest

from scripts import judge_naturalness_panel as cli
import humanizers.naturalness_panel_runner as runner


def inputs(path, *, all_failed=False):
    rows = []
    for source in ["source-1", "source-2"]:
        for index, system in enumerate(cli.DEFAULT_SYSTEMS):
            failed = index == 3 and (all_failed or source == "source-2")
            rows.append(dict(source_id=source, source_group_id=source,
                             domain="Essay", generator="Claude", humanizer=system,
                             original="The original English passage.",
                             modified=f"An alternative English passage, version {index}.",
                             status="truncated" if failed else "ok"))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def arguments(monkeypatch, path, output, *extra):
    monkeypatch.setattr(sys, "argv", ["judge_naturalness_panel.py", "--rewrites", str(path),
        "--output", str(output), "--max-sources", "2", "--bootstrap", "10", *extra])


def no_client():
    raise AssertionError("La preparazione non deve creare un client API")


def test_cli_offline_plan_and_empty_ratings(monkeypatch, tmp_path):
    source, output = tmp_path / "rewrites.jsonl", tmp_path / "panel"
    inputs(source)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(runner, "GroqClient", no_client)
    arguments(monkeypatch, source, output)
    cli.main()
    plan = json.loads((output / "judge_plan.json").read_text())
    eligibility = json.loads((output / "judge_eligibility.json").read_text())
    manifest = json.loads((output / "judge_run.json").read_text())
    assert eligibility["n_selected_sources"] == 2
    assert eligibility["n_eligible_panels"] == 1
    assert plan["expected_completion_requests"] == 12
    assert manifest["generation_post_attempts"] == 0
    assert not (output / "judgments.jsonl").exists()
    assert (output / "system_summary.csv").is_file()
    assert (output / "summary.json").is_file()
    assert not list(output.glob("*.md")) and not list(output.glob("*.pdf"))


def test_cli_all_failed_never_calls_api_even_with_execute(monkeypatch, tmp_path):
    source, output = tmp_path / "rewrites.jsonl", tmp_path / "panel"
    inputs(source, all_failed=True)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(runner, "GroqClient", no_client)
    arguments(monkeypatch, source, output, "--execute")
    cli.main()
    manifest = json.loads((output / "judge_run.json").read_text())
    assert manifest["generation_post_attempts"] == 0
    assert manifest["expected_completion_requests"] == 0
    assert manifest["n_primary_sources"] == 0


def test_cli_refuses_changed_input_before_api(monkeypatch, tmp_path):
    source, output = tmp_path / "rewrites.jsonl", tmp_path / "panel"
    inputs(source)
    monkeypatch.setattr(runner, "GroqClient", no_client)
    arguments(monkeypatch, source, output)
    cli.main()
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        if row["status"] == "truncated":
            row["status"] = "generation_error"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    arguments(monkeypatch, source, output, "--execute")
    with pytest.raises(ValueError, match="judge_eligibility.json"):
        cli.main()
