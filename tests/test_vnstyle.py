"""Tests for the deterministic Vietnamese word-order refinement (M3-C2)."""

from __future__ import annotations

from vidub.utils.vnstyle import ADVERBIALS, SUBJECTS, apply_vnstyle


# ------------------------------------------------------------------- fronting
def test_fronts_time_adverbial_user_example() -> None:
    """The exact user-reported example is reordered and the subject lowered."""
    out, n = apply_vnstyle("Bạn có thể sau đó hoàn thành công việc này.")

    assert out == "Sau đó, bạn có thể hoàn thành công việc này."
    assert n == 1


def test_fronts_without_optional_can() -> None:
    out, n = apply_vnstyle("Chúng ta hiện nay đang làm việc.")

    assert out == "Hiện nay, chúng ta đang làm việc."
    assert n == 1


def test_fronts_when_can_follows_adverbial() -> None:
    """The optional "có thể" may sit *after* the adverbial; both orders work."""
    out, n = apply_vnstyle("Họ sau đó có thể rời đi.")

    assert out == "Sau đó, họ có thể rời đi."
    assert n == 1


def test_fronting_consumes_existing_comma() -> None:
    out, _ = apply_vnstyle("Bạn sau đó, hoàn thành.")

    assert out == "Sau đó, bạn hoàn thành."


def test_fronting_is_idempotent() -> None:
    text = "Sau đó, bạn có thể hoàn thành."

    assert apply_vnstyle(text) == (text, 0)


def test_fronting_handles_every_subject_and_adverbial() -> None:
    for subject in SUBJECTS:
        for adverb in ADVERBIALS:
            source = f"{subject} {adverb} làm việc."
            out, n = apply_vnstyle(source)
            assert n == 1
            assert out.startswith(f"{adverb[:1].upper()}{adverb[1:]}, ")
            assert out.endswith(" làm việc.")


# -------------------------------------------------------- repeated adverbials
def test_collapses_repeated_leading_adverbial() -> None:
    out, n = apply_vnstyle("Sau đó, sau đó, bạn hoàn thành.")

    assert out == "Sau đó, bạn hoàn thành."
    assert n == 1


def test_collapses_triple_repeated_adverbial() -> None:
    out, n = apply_vnstyle("Sau đó, sau đó, sau đó, bạn hoàn thành.")

    assert out == "Sau đó, bạn hoàn thành."
    assert n == 2


def test_does_not_collapse_different_adverbials() -> None:
    text = "Sau đó, bây giờ bạn làm."

    assert apply_vnstyle(text) == (text, 0)


# ----------------------------------------------------------------- no-match
def test_no_match_is_unchanged() -> None:
    text = "Đây là một câu bình thường."

    assert apply_vnstyle(text) == (text, 0)


def test_empty_and_none_like_inputs() -> None:
    assert apply_vnstyle("") == ("", 0)


def test_một_de_dupe_is_delegated_to_clean_hesitation() -> None:
    """vnstyle must not touch immediate word repeats (clean_hesitation owns them)."""
    text = "một một"

    assert apply_vnstyle(text) == (text, 0)
