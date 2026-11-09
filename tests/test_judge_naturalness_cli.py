"""Naturalness coverage keeps failed generations visible without API calls."""
import json
import sys

import pytest

from scripts import judge_naturalness as cli


def inputs(path, *, all_failed=False):
    rows = []
    for source in ['good', 'failed']:
        for system in ['a', 'b']:
            failed = system == 'b' and (all_failed or source == 'failed')
            rows.append(dict(id=source+system, source_id=source, source_group_id=source,
                domain='Essay', generator='Claude', humanizer=system,
                original='The original passage.',
                modified='' if failed else 'A rewritten passage.',
                status='generation_error' if failed else 'ok'))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def arguments(monkeypatch, path, output, *extra):
    monkeypatch.setattr(sys, 'argv', ['judge_naturalness.py', '--rewrites', str(path),
        '--output', str(output), '--humanizer-a', 'a', '--humanizer-b', 'b',
        '--max-pairs', '2', '--failure-policy', 'report', *extra])


def test_dry_cli_covers_failed_source_and_guards_changed_failures_on_resume(tmp_path, monkeypatch):
    path, output = tmp_path/'rewrites.jsonl', tmp_path/'judge'
    inputs(path)
    arguments(monkeypatch, path, output)
    cli.main()
    manifest = json.loads((output/'judge_run.json').read_text())
    assert manifest['n_selected_sources'] == 2
    assert manifest['n_eligible_pairs'] == 1
    assert manifest['n_generation_failure_pairs'] == 1
    assert manifest['expected_orders'] == 2
    assert manifest['eligible_pair_coverage'] == .5
    summary = {r['source_id']: r for r in json.loads((output/'pairwise_summary.json').read_text())}
    assert summary['failed']['judgment_status'] == 'not_judged_generation_failure'
    assert summary['failed']['preference'] is None
    assert summary['good']['judgment_status'] == 'not_completed'
    original_plan = (output/'judge_plan.json').read_bytes()
    cli.main()
    assert (output/'judge_plan.json').read_bytes() == original_plan
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row['status'] == 'generation_error':
            row['status'] = 'input_too_long'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='eligibility differs'):
        cli.main()


def test_no_successful_pair_makes_no_api_request_even_when_execute_requested(tmp_path, monkeypatch):
    path, output = tmp_path/'rewrites.jsonl', tmp_path/'judge'
    inputs(path, all_failed=True)
    arguments(monkeypatch, path, output, '--execute')
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    cli.main()
    manifest = json.loads((output/'judge_run.json').read_text())
    assert manifest['status'] == 'not_judged_no_eligible_pairs'
    assert manifest['api_execution_requested'] is True
    assert manifest['generation_api_executed'] is False
    assert manifest['generation_post_attempts'] == manifest['expected_orders'] == 0
    assert manifest['n_generation_failure_pairs'] == 2
    assert manifest['n_selected_sources'] == 2
    assert manifest['stable_preference_coverage'] == 0
    assert len(json.loads((output/'pairwise_summary.json').read_text())) == 2
