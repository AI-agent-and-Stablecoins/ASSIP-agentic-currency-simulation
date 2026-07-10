"""Reserve disclosure/transparency rating."""

from enum import Enum


class TransparencyRating(str, Enum):
    AUDITED_MONTHLY = "audited_monthly"
    QUARTERLY_ATTESTATION = "quarterly_attestation"
    NO_DISCLOSURE = "no_disclosure"


_SCORES: dict[TransparencyRating, float] = {
    TransparencyRating.AUDITED_MONTHLY: 1.0,
    TransparencyRating.QUARTERLY_ATTESTATION: 0.6,
    TransparencyRating.NO_DISCLOSURE: 0.1,
}


def transparency_score(rating: TransparencyRating) -> float:
    return _SCORES[rating]
