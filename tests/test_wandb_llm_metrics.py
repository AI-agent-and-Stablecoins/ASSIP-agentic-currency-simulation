from metrics.wandb_logger import WandbRunLogger
from src.llm.hallucination_detector import HallucinationDirection, HallucinationResult


class _FakeWandb:
    def __init__(self):
        self.logged: list[tuple[dict, int | None]] = []

    def log(self, data: dict, step: int | None = None) -> None:
        self.logged.append((data, step))


def _logger_with_fake_wandb() -> WandbRunLogger:
    # Bypasses __init__ (which calls the real wandb.init) since wandb is an
    # optional dependency this test suite must not require installed.
    logger = WandbRunLogger.__new__(WandbRunLogger)
    logger._wandb = _FakeWandb()
    return logger


def _hallucination(direction: HallucinationDirection) -> HallucinationResult:
    paid = 150.0 if direction == HallucinationDirection.OVERPAYMENT else 100.0
    return HallucinationResult(
        expected_value=100.0, paid_value=paid, absolute_error=abs(paid - 100.0), percentage_error=0.0, direction=direction
    )


def test_log_llm_metrics_computes_hallucination_and_fallback_rates():
    logger = _logger_with_fake_wandb()
    results = [_hallucination(HallucinationDirection.ACCURATE), _hallucination(HallucinationDirection.OVERPAYMENT)]
    attempts = [["model-a"], ["model-a", "model-b"]]

    logger.log_llm_metrics(results, attempts, step=5)

    data, step = logger._wandb.logged[0]
    assert step == 5
    assert data["llm_hallucination_rate"] == 0.5
    assert data["llm_fallback_rate"] == 0.5


def test_log_llm_metrics_handles_empty_inputs_without_dividing_by_zero():
    logger = _logger_with_fake_wandb()

    logger.log_llm_metrics([], [], step=0)

    data, _ = logger._wandb.logged[0]
    assert data["llm_hallucination_rate"] == 0.0
    assert data["llm_fallback_rate"] == 0.0
