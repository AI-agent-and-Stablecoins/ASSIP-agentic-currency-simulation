from src.utils.helpers import generate_id


def test_generate_id_uses_full_uuid4_entropy_to_avoid_collisions_at_matrix_run_scale():
    identifier = generate_id("tx")
    prefix, _, suffix = identifier.partition("-")
    assert prefix == "tx"
    assert len(suffix) == 32
    int(suffix, 16)  # is valid hex


def test_generate_id_produces_unique_values_across_many_calls():
    ids = {generate_id("tx") for _ in range(100_000)}
    assert len(ids) == 100_000
