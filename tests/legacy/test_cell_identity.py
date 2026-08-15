import pytest

from src.legacy.econometrics.cell_identity import cell_key_from_run_id


def test_cell_key_from_run_id_recovers_master():
    assert cell_key_from_run_id("matrix-abc123-master-seed0") == "master"


def test_cell_key_from_run_id_recovers_sandbox_domestic_and_cross_border():
    assert (
        cell_key_from_run_id("matrix-abc123-liquidity_vs_governance_domestic-seed3")
        == "liquidity_vs_governance_domestic"
    )
    assert (
        cell_key_from_run_id("matrix-abc123-liquidity_vs_governance_cross_border-seed3")
        == "liquidity_vs_governance_cross_border"
    )


def test_cell_key_from_run_id_handles_a_matrix_run_id_containing_hyphens():
    assert (
        cell_key_from_run_id("pilot-run-2026-08-02-asset_backing_vs_liquidity_cross_border-seed12")
        == "asset_backing_vs_liquidity_cross_border"
    )


def test_cell_key_from_run_id_raises_for_unrecognized_run_id():
    with pytest.raises(ValueError):
        cell_key_from_run_id("not-a-matrix-run-id-at-all")
