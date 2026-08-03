"""Canonical trained-LoRA selfie prompt and content-free binding tests."""

from dataclasses import replace
from decimal import Decimal

import pytest

from kestrel_feature_visual.selfie_spec import (
    ResolvedSelfiePrompt,
    bind_lora_selfie_spec,
    resolve_lora_selfie_spec,
    resolve_selfie_prompt,
)


def _lora_spec(**changes):
    values = {
        "scene": "casual",
        "style": "photorealistic",
        "custom_prompt": None,
        "trigger_word": "TOKluna",
        "lora_version_id": "00000000-0000-0000-0000-000000000001",
        "lora_encrypted_sha256": "a" * 64,
        "lora_plaintext_sha256": "b" * 64,
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_version": "sdxl-1.0",
        "trainer_version": "kohya-1.2.3",
        "flux_version": "flux1",
    }
    values.update(changes)
    return resolve_lora_selfie_spec(**values)


def test_default_prompt_matches_exact_generation_prompt() -> None:
    prompt, spec = _lora_spec()

    assert prompt.prompt == (
        "A photo of TOKluna, casual selfie at home, comfortable clothes, "
        "natural lighting, relaxed smile. High quality, photorealistic, 8k."
    )
    assert spec.prompt_sha256 == prompt.prompt_sha256
    assert spec.spec_sha256 == _lora_spec()[1].spec_sha256
    assert "TOKluna" not in str(spec.evidence_dict())


def test_custom_prompt_is_normalized_and_triggered_once() -> None:
    prompt = resolve_selfie_prompt(
        scene="beach",
        style="anime",
        custom_prompt="  TRIGGER_WORD   beside  the sea ",
        trigger_word="TOKluna",
    )

    assert prompt.prompt == "TOKluna beside the sea"
    assert prompt.prompt.count("TOKluna") == 1


def test_custom_prompt_with_explicit_trigger_is_not_prefixed_again() -> None:
    prompt = resolve_selfie_prompt(
        scene="beach",
        style="photorealistic",
        custom_prompt="portrait of TOKluna beside the sea",
        trigger_word="TOKluna",
    )

    assert prompt.prompt == "portrait of TOKluna beside the sea"
    assert prompt.prompt.count("TOKluna") == 1


@pytest.mark.parametrize(
    "custom_prompt",
    (
        "TRIGGER_WORD beside TRIGGER_WORD",
        "TOKluna beside TOKluna",
        "TRIGGER_WORD beside TOKluna",
    ),
)
def test_custom_prompt_rejects_multiple_trigger_bindings(custom_prompt) -> None:
    with pytest.raises(ValueError, match="bind the trigger once"):
        resolve_selfie_prompt(
            scene="beach",
            style="photorealistic",
            custom_prompt=custom_prompt,
            trigger_word="TOKluna",
        )


@pytest.mark.parametrize(
    ("changes", "field", "error_type"),
    [
        ({"style": "cinematic"}, "style", ValueError),
        ({"trigger_word": "bad trigger"}, "trigger", ValueError),
        ({"lora_encrypted_sha256": "bad"}, "digest", ValueError),
        ({"lora_plaintext_sha256": "bad"}, "digest", ValueError),
        ({"seed": -1}, "seed", ValueError),
        ({"seed": True}, "seed", TypeError),
        ({"width": 1024.0}, "width", TypeError),
        ({"num_outputs": 5}, "num_outputs", ValueError),
        ({"width": 1025}, "width", ValueError),
        ({"guidance_scale": Decimal(31)}, "guidance", ValueError),
    ],
)
def test_invalid_or_unbounded_spec_fails_closed(changes, field, error_type) -> None:
    with pytest.raises(error_type, match=field):
        _lora_spec(**changes)


def test_digest_changes_for_every_realized_dimension() -> None:
    baseline = _lora_spec()[1].spec_sha256
    variants = (
        {"scene": "portrait"},
        {"style": "anime"},
        {"custom_prompt": "in a library"},
        {"trigger_word": "TOKnova"},
        {"lora_version_id": "00000000-0000-0000-0000-000000000002"},
        {"lora_encrypted_sha256": "c" * 64},
        {"lora_plaintext_sha256": "d" * 64},
        {"base_model": "black-forest-labs/FLUX.1-dev"},
        {"model_version": "flux-dev-v1"},
        {"trainer_version": "trainer-2"},
        {"flux_version": "flux2"},
        {"seed": 42},
        {"num_outputs": 2},
        {"width": 768},
        {"height": 768},
        {"num_inference_steps": 29},
        {"guidance_scale": "4.5"},
    )

    assert all(_lora_spec(**change)[1].spec_sha256 != baseline for change in variants)


def test_provider_can_rebind_the_exact_resolved_prompt() -> None:
    prompt, expected = _lora_spec()

    actual = bind_lora_selfie_spec(
        resolved_prompt=prompt,
        lora_version_id=expected.lora_version_id,
        lora_encrypted_sha256=expected.lora_encrypted_sha256,
        lora_plaintext_sha256=expected.lora_plaintext_sha256,
        base_model=expected.base_model,
        model_version=expected.model_version,
        trainer_version=expected.trainer_version,
        flux_version=expected.flux_version,
    )

    assert actual == expected


def test_resolved_prompt_rejects_a_tampered_digest() -> None:
    prompt, _expected = _lora_spec()

    with pytest.raises(ValueError, match="prompt digest is inconsistent"):
        ResolvedSelfiePrompt(
            scene=prompt.scene,
            style=prompt.style,
            prompt=prompt.prompt,
            prompt_sha256="0" * 64,
            trigger_word_sha256=prompt.trigger_word_sha256,
            seed=prompt.seed,
            num_outputs=prompt.num_outputs,
            width=prompt.width,
            height=prompt.height,
            num_inference_steps=prompt.num_inference_steps,
            guidance_scale=prompt.guidance_scale,
        )


def test_content_free_spec_rejects_a_forged_digest() -> None:
    _prompt, expected = _lora_spec()

    with pytest.raises(ValueError, match="spec digest is inconsistent"):
        replace(expected, spec_sha256="0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema_version", 2, "schema version"),
        ("scene", "unknown", "scene"),
        ("lora_version_id", "bad value", "LoRA version id"),
        ("lora_encrypted_sha256", "bad", "encrypted LoRA digest"),
        ("width", 1025, "width"),
    ),
)
def test_content_free_spec_revalidates_every_public_construction(
    field, value, error
) -> None:
    _prompt, expected = _lora_spec()

    with pytest.raises((TypeError, ValueError), match=error):
        replace(expected, **{field: value})
