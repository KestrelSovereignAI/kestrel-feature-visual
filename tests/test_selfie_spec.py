"""Canonical trained-LoRA selfie prompt and content-free binding tests."""

import hashlib
from dataclasses import replace
from decimal import Decimal

import pytest

from kestrel_feature_visual.selfie_spec import (
    SELFIE_SCENE_PROMPTS,
    ResolvedSelfiePrompt,
    normalize_trigger_word,
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
        ({"trigger_word": "-TOKluna"}, "trigger", ValueError),
        ({"trigger_word": "   "}, "trigger", ValueError),
        ({"trigger_word": ""}, "trigger", ValueError),
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
            trigger_word=prompt.trigger_word,
            trigger_word_sha256=prompt.trigger_word_sha256,
            seed=prompt.seed,
            num_outputs=prompt.num_outputs,
            width=prompt.width,
            height=prompt.height,
            num_inference_steps=prompt.num_inference_steps,
            guidance_scale=prompt.guidance_scale,
        )


def _direct_prompt(base: ResolvedSelfiePrompt, **overrides) -> ResolvedSelfiePrompt:
    """Construct directly, bypassing ``resolve_selfie_prompt``'s validation."""
    fields = {
        "scene": base.scene,
        "style": base.style,
        "prompt": base.prompt,
        "prompt_sha256": base.prompt_sha256,
        "trigger_word": base.trigger_word,
        "trigger_word_sha256": base.trigger_word_sha256,
        "seed": base.seed,
        "num_outputs": base.num_outputs,
        "width": base.width,
        "height": base.height,
        "num_inference_steps": base.num_inference_steps,
        "guidance_scale": base.guidance_scale,
    }
    fields.update(overrides)
    return ResolvedSelfiePrompt(**fields)


def test_resolved_prompt_rejects_a_trigger_the_prompt_never_binds() -> None:
    """The type is the trust boundary, not just the factory.

    ``bind_lora_selfie_spec`` accepts any ``ResolvedSelfiePrompt`` as proof
    that the prompt binds the attested trigger exactly once. Without this
    check, a directly constructed instance could attest one trigger while the
    prompt bound a different one, and still yield a valid ``spec_sha256``.
    """
    prompt, _expected = _lora_spec()
    foreign = "EVILTRIGGER"
    assert foreign not in prompt.prompt

    with pytest.raises(ValueError, match="trigger digest is inconsistent"):
        _direct_prompt(prompt, trigger_word=foreign)

    with pytest.raises(ValueError, match="must bind the trigger once"):
        _direct_prompt(
            prompt,
            trigger_word=foreign,
            trigger_word_sha256=hashlib.sha256(foreign.encode()).hexdigest(),
        )


def test_resolved_prompt_rejects_a_prompt_binding_the_trigger_zero_or_twice() -> None:
    prompt, _expected = _lora_spec()
    trigger = prompt.trigger_word

    unbound = "a nice picture with no token at all"
    with pytest.raises(ValueError, match="must bind the trigger once"):
        _direct_prompt(
            prompt,
            prompt=unbound,
            prompt_sha256=hashlib.sha256(unbound.encode()).hexdigest(),
        )

    doubled = f"{trigger} standing next to {trigger}"
    with pytest.raises(ValueError, match="must bind the trigger once"):
        _direct_prompt(
            prompt,
            prompt=doubled,
            prompt_sha256=hashlib.sha256(doubled.encode()).hexdigest(),
        )


def test_content_free_spec_rejects_a_forged_digest() -> None:
    _prompt, expected = _lora_spec()

    with pytest.raises(ValueError, match="spec digest is inconsistent"):
        replace(expected, spec_sha256="0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema_version", 2, "schema version"),
        ("scene", "Unknown Scene", "scene"),
        ("scene", "-leading-dash", "scene"),
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


@pytest.mark.parametrize(
    "companion_name",
    [
        "Anna Marie",
        "Émilie",
        "O'Brien",
        "李小明",
        "Mary-Jane",
        "Luna",
        # The slice lands ON whitespace for these: TOK{name[:8]} yields a
        # trailing space / embedded tab. Ordinary name shapes, and the exact
        # case the first pass of this fix still rejected.
        "Maria J Lopez",
        "Jo Ann Smith",
        "Mary\tJane",
        "Li  Wei",
    ],
)
def test_triggers_this_package_mints_are_accepted(companion_name) -> None:
    """Triggers are minted as ``TOK{companion_name[:8]}`` and persisted.

    A LoRA is trained on that exact token, so the stored value cannot be
    rewritten without invalidating the weights. Rejecting a name containing a
    space, apostrophe, or non-ASCII letter would permanently break every
    selfie for the affected companion.
    """
    trigger = f"TOK{companion_name[:8]}"
    resolved = resolve_selfie_prompt(
        scene="casual",
        style="photorealistic",
        custom_prompt=None,
        trigger_word=trigger,
    )
    # Stored triggers are canonicalized the same way prompts are, so a slice
    # that landed on whitespace still resolves instead of failing closed.
    assert resolved.trigger_word == normalize_trigger_word(trigger)
    assert resolved.trigger_word in resolved.prompt


def test_trigger_binding_still_exact_for_a_trigger_containing_a_space() -> None:
    """Whole-token matching must survive the permissive character set."""
    trigger = "TOKAnna Mar"
    resolved = resolve_selfie_prompt(
        scene="casual",
        style="photorealistic",
        custom_prompt=f"{trigger} reading a book",
        trigger_word=trigger,
    )
    assert resolved.prompt.count(trigger) == 1

    with pytest.raises(ValueError, match="bind the trigger once"):
        resolve_selfie_prompt(
            scene="casual",
            style="photorealistic",
            custom_prompt=f"{trigger} beside {trigger}",
            trigger_word=trigger,
        )


@pytest.mark.parametrize("raw", [" TOKluna", "TOKluna ", "TOK  luna", "TOKluna\n"])
def test_factory_canonicalizes_whitespace_but_the_type_demands_canonical_form(
    raw,
) -> None:
    """Normalization belongs at the boundary; the type stays strict.

    ``resolve_selfie_prompt`` accepts a stored trigger whose whitespace is
    incidental and canonicalizes it. A directly constructed
    ``ResolvedSelfiePrompt`` gets no such courtesy — it must already hold the
    canonical form, so a digest can never attest an uncanonical variant.
    """
    resolved = resolve_selfie_prompt(
        scene="casual",
        style="photorealistic",
        custom_prompt=None,
        trigger_word=raw,
    )
    assert resolved.trigger_word == normalize_trigger_word(raw)

    with pytest.raises(ValueError, match="trigger word is invalid"):
        _direct_prompt(
            resolved,
            trigger_word=raw,
            trigger_word_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        )


def test_empty_custom_prompt_means_not_supplied() -> None:
    """origin/main gated on ``if custom_prompt:``; "" must still make a selfie.

    ``custom_prompt`` is an optional, undocumented parameter on an LLM-invoked
    tool, and emitting "" for an unused optional string is common model
    behavior. Treating it as a hard error turned the primary "send me a
    selfie" path into a failure.
    """
    scene_only = resolve_selfie_prompt(
        scene="casual",
        style="photorealistic",
        custom_prompt=None,
        trigger_word="TOKluna",
    )
    for blank in ("", "   ", "\t\n "):
        assert (
            resolve_selfie_prompt(
                scene="casual",
                style="photorealistic",
                custom_prompt=blank,
                trigger_word="TOKluna",
            ).prompt
            == scene_only.prompt
        )


@pytest.mark.parametrize("scene", ["shower", "bedroom", "spread_eagle"])
def test_unknown_scene_is_preserved_and_distinguishes_the_digest(scene) -> None:
    """A scene this package cannot describe is still the scene that was asked for.

    frinz tier-gates these as paid sovereign content, routes them to a
    different render engine, coalesces queued work on (companion_did, scene),
    and names the stored asset by scene. Coercing them to "casual" made the
    prompt, the config, and spec_sha256 describe three different things — and
    collapsed four distinct paid quotes onto a single digest.
    """
    assert scene not in SELFIE_SCENE_PROMPTS

    prompt, spec = _lora_spec(scene=scene)
    assert prompt.scene == scene
    assert spec.scene == scene

    # No invented description: the prompt does not claim to be a casual selfie.
    assert SELFIE_SCENE_PROMPTS["casual"] not in prompt.prompt
    assert prompt.prompt.count(prompt.trigger_word) == 1

    # And the digest actually distinguishes it from the coerced-to value.
    assert spec.spec_sha256 != _lora_spec(scene="casual")[1].spec_sha256


def test_scene_is_normalized_before_it_reaches_any_consumer() -> None:
    """Downstream lookups are exact-match, so whitespace/case must not leak."""
    prompt, spec = _lora_spec(scene="  BeAcH  ")
    assert prompt.scene == "beach"
    assert spec.scene == "beach"
    assert spec.spec_sha256 == _lora_spec(scene="beach")[1].spec_sha256
