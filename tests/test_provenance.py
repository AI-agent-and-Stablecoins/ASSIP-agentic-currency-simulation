from pathlib import Path
import subprocess

import pytest

from src.simulation.provenance import compute_config_hash, compute_git_commit_hash, model_roster_summary_for


def test_compute_git_commit_hash_matches_git_rev_parse():
    result = compute_git_commit_hash()
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert result == expected


def test_compute_config_hash_is_deterministic_and_order_independent(tmp_path):
    file_a = tmp_path / "a.yaml"
    file_b = tmp_path / "b.yaml"
    file_a.write_text("key: value\n")
    file_b.write_text("other: 1\n")

    hash_1 = compute_config_hash([file_a, file_b])
    hash_2 = compute_config_hash([file_b, file_a])  # reversed order

    assert hash_1 == hash_2  # sorted internally, order of the input list shouldn't matter
    assert len(hash_1) == 64  # hex-encoded SHA-256


def test_compute_config_hash_changes_when_file_content_changes(tmp_path):
    file_a = tmp_path / "a.yaml"
    file_a.write_text("key: value\n")
    hash_before = compute_config_hash([file_a])

    file_a.write_text("key: different\n")
    hash_after = compute_config_hash([file_a])

    assert hash_before != hash_after


def test_model_roster_summary_for_describes_agent_count_and_model_diversity():
    from src.agents.population import generate_agent_population

    population = generate_agent_population(seed=0, model_candidates=[f"vendor/model-{i}" for i in range(10)])

    summary = model_roster_summary_for(population)

    assert "100" in summary
    assert "10" in summary  # 10 distinct models used
