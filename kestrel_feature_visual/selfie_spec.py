"""Canonical content-free specification for trained-LoRA selfie inference."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

SELFIE_SPEC_SCHEMA_VERSION = 1

SELFIE_SCENE_PROMPTS: Mapping[str, str] = MappingProxyType(
    {
        "portrait": "professional headshot, studio lighting, neutral background, business attire",
        "casual": "casual selfie at home, comfortable clothes, natural lighting, relaxed smile",
        "glamour": "glamorous evening setting, elegant black dress, sophisticated pose, studio lighting",
        "flirty": "playful smile, flirtatious expression, slight head tilt, soft lighting",
        "cozy": "cozy home setting, comfortable sweater, warm atmosphere, soft natural light",
        "adventure": "outdoor hiking setting, athletic wear, dynamic pose, bright daylight",
        "mysterious": "dramatic shadows, dark elegant attire, enigmatic expression, moody lighting",
        "romantic": "soft romantic candlelit setting, elegant dress, intimate atmosphere, warm colors",
        "playful": "fun playful expression, colorful casual outfit, bright colors, dynamic pose",
        "dreamy": "dreamy soft focus, flowing white dress, ethereal lighting, pastel colors",
        "confident": "confident powerful pose, professional attire, strong lighting, bold composition",
        "beach": "at the beach, bikini swimsuit, golden hour sunset lighting, ocean waves in background, beautiful smile, selfie angle",
        "swimsuit": "poolside setting, stylish bikini, bright sunny day, relaxed pose, tropical vibes",
        "tropical": "tropical beach paradise, colorful bikini, palm trees, crystal clear water, vacation selfie",
        "pool": "luxury pool setting, designer swimwear, sunglasses, lounge chair, summer vibes",
        "fitness": "gym or yoga studio, athletic sports bra and leggings, energetic pose, natural lighting",
        "nightout": "nightclub or bar setting, sexy cocktail dress, glamorous makeup, neon lights",
        "lingerie": "elegant bedroom setting, tasteful lingerie, soft boudoir lighting, confident pose",
        "summer": "sunny outdoor cafe, sundress, bright daylight, happy relaxed expression",
        "nurse": "healthcare setting, nurse scrubs with stethoscope, hospital or clinic background, professional caring expression, soft clinical lighting",
        "topless": "artistic portrait, bare breasts visible, tasteful nude photography, studio lighting, sensual pose",
        "nude": "full nude portrait, artistic nude photography, studio setting, tasteful pose, natural lighting",
    }
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,254}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
DESCRIPTOR_MAX_LENGTH = 128


def normalize_descriptor(value: str) -> str:
    """Canonicalize a caller-supplied scene or style label.

    Scene and style are owned by the CALLER, not by this package. frinz
    forwards both unvalidated from an HTTP body and from LLM tool arguments,
    and deliberately supports free-form prose scenes - its own comment records
    that fail-closed handling wrongly 403'd "stargazing at night with aurora
    borealis". So these are normalized for exact-match lookups downstream, and
    never rejected for failing to appear in this package's tables.
    """
    return " ".join(value.split()).lower()


def is_valid_descriptor(value: object) -> bool:
    """Bound a descriptor without constraining its vocabulary."""
    return (
        isinstance(value, str)
        and 1 <= len(value) <= DESCRIPTOR_MAX_LENGTH
        and value == normalize_descriptor(value)
        and _CONTROL_RE.search(value) is None
    )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STYLES = frozenset({"photorealistic", "anime", "artistic"})

TRIGGER_WORD_MAX_LENGTH = 128


def normalize_trigger_word(value: str) -> str:
    """Collapse whitespace runs exactly as ``_normalize_prompt`` does.

    Triggers are minted as ``TOK{companion_name[:8]}``, and that slice lands
    wherever it lands: "Maria J Lopez" yields ``'TOKMaria J '`` with a trailing
    space, and "Mary\\tJane" yields ``'TOKMary\\tJan'`` with an embedded tab.
    Those values are already persisted next to LoRAs trained on them, so they
    must keep working.

    Normalizing the trigger with the same rule applied to the prompt keeps the
    two sides comparable — a trigger written into a custom prompt normalizes
    identically, so the exactly-once binding stays well defined. It is also
    tokenizer-inert: collapsing a trailing or doubled space does not change
    what the model sees.
    """
    return " ".join(value.split())


def is_valid_trigger_word(value: object) -> bool:
    """Return whether ``value`` is a usable, canonical LoRA trigger.

    Deliberately permissive about the character set. A companion named
    "Anna Marie", "Émilie", or "O'Brien" has a stored trigger containing a
    space, a non-ASCII letter, or an apostrophe, and the LoRA was *trained* on
    that exact token — it cannot be rewritten without invalidating the weights.
    Rejecting those would permanently break every selfie for that companion.

    What is enforced is what the prompt contract actually depends on: a
    non-empty, bounded, single-line token in canonical (whitespace-collapsed)
    form, with no control characters, beginning with an alphanumeric. Regex
    safety does not depend on this — ``_trigger_pattern`` escapes the value.
    """
    return (
        isinstance(value, str)
        and 1 <= len(value) <= TRIGGER_WORD_MAX_LENGTH
        and value == normalize_trigger_word(value)
        and value[0].isalnum()
        and _CONTROL_RE.search(value) is None
    )


def _trigger_pattern(trigger_word: str) -> re.Pattern[str]:
    """Match ``trigger_word`` as a whole token, never as a substring."""
    return re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(trigger_word)}(?![A-Za-z0-9_-])")


@dataclass(frozen=True, slots=True)
class ResolvedSelfiePrompt:
    """Exact final prompt and generation knobs sent to an image provider."""

    scene: str
    style: str
    prompt: str
    prompt_sha256: str
    trigger_word: str
    trigger_word_sha256: str
    seed: int
    num_outputs: int
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: Decimal

    def __post_init__(self) -> None:
        if not is_valid_descriptor(self.scene):
            raise ValueError("resolved selfie prompt scene is invalid")
        if not is_valid_descriptor(self.style):
            raise ValueError("resolved selfie prompt style is invalid")
        if (
            not isinstance(self.prompt, str)
            or not self.prompt
            or len(self.prompt) > 8_000
        ):
            raise ValueError("resolved selfie prompt text is invalid")
        _validate_sha256(self.prompt_sha256, "prompt")
        if self.prompt_sha256 != _sha256_text(self.prompt):
            raise ValueError("resolved selfie prompt digest is inconsistent")
        # The trigger binding is this type's whole point, so it is re-checked
        # here rather than only inside ``resolve_selfie_prompt``.  Consumers
        # (``bind_lora_selfie_spec``, and providers reconstructing this object
        # per the README) treat the type itself as the trust boundary: without
        # this, a directly constructed instance could attest a
        # ``trigger_word_sha256`` for a token the prompt never binds — or binds
        # twice — and still produce a valid downstream ``spec_sha256``.  The
        # plaintext trigger is already contained in ``prompt``, so carrying it
        # discloses nothing the object did not already hold.
        if not is_valid_trigger_word(self.trigger_word):
            raise ValueError("resolved selfie prompt trigger word is invalid")
        _validate_sha256(self.trigger_word_sha256, "trigger word")
        if self.trigger_word_sha256 != _sha256_text(self.trigger_word):
            raise ValueError("resolved selfie prompt trigger digest is inconsistent")
        if len(_trigger_pattern(self.trigger_word).findall(self.prompt)) != 1:
            raise ValueError("resolved selfie prompt must bind the trigger once")
        _validate_generation_parameters(
            seed=self.seed,
            num_outputs=self.num_outputs,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )


@dataclass(frozen=True, slots=True)
class ResolvedLoraSelfieSpec:
    """Content-free accepted LoRA identity plus exact generation parameters."""

    schema_version: int
    scene: str
    style: str
    prompt_sha256: str
    trigger_word_sha256: str
    lora_version_id: str
    lora_encrypted_sha256: str
    lora_plaintext_sha256: str
    base_model: str
    model_version: str
    trainer_version: str
    flux_version: str | None
    seed: int
    num_outputs: int
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: Decimal
    spec_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SELFIE_SPEC_SCHEMA_VERSION:
            raise ValueError("selfie spec schema version is unsupported")
        if not is_valid_descriptor(self.scene):
            raise ValueError("selfie spec scene is invalid")
        if not is_valid_descriptor(self.style):
            raise ValueError("selfie spec style is invalid")
        for value, label in (
            (self.lora_version_id, "LoRA version id"),
            (self.base_model, "base model"),
            (self.model_version, "model version"),
            (self.trainer_version, "trainer version"),
        ):
            _validate_safe_id(value, label)
        if self.flux_version is not None:
            _validate_safe_id(self.flux_version, "FLUX version")
        for value, label in (
            (self.prompt_sha256, "prompt"),
            (self.trigger_word_sha256, "trigger word"),
            (self.lora_encrypted_sha256, "encrypted LoRA"),
            (self.lora_plaintext_sha256, "plaintext LoRA"),
        ):
            _validate_sha256(value, label)
        _validate_generation_parameters(
            seed=self.seed,
            num_outputs=self.num_outputs,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        _validate_sha256(self.spec_sha256, "spec")
        if self.spec_sha256 != _sha256_json(self.evidence_dict()):
            raise ValueError("selfie spec digest is inconsistent")

    def evidence_dict(self) -> dict[str, object]:
        """Return the canonical content-free evidence covered by the digest."""

        return {
            "schema_version": self.schema_version,
            "scene": self.scene,
            "style": self.style,
            "prompt_sha256": self.prompt_sha256,
            "trigger_word_sha256": self.trigger_word_sha256,
            "lora_version_id": self.lora_version_id,
            "lora_encrypted_sha256": self.lora_encrypted_sha256,
            "lora_plaintext_sha256": self.lora_plaintext_sha256,
            "base_model": self.base_model,
            "model_version": self.model_version,
            "trainer_version": self.trainer_version,
            "flux_version": self.flux_version,
            "seed": self.seed,
            "num_outputs": self.num_outputs,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": _decimal_text(self.guidance_scale),
        }


def resolve_selfie_prompt(
    *,
    scene: str,
    style: str,
    custom_prompt: str | None,
    trigger_word: str,
    seed: int = 0,
    num_outputs: int = 1,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: Decimal | float | str = Decimal("4.0"),
) -> ResolvedSelfiePrompt:
    """Resolve one final prompt using the same rules for quote and execution."""

    if not isinstance(scene, str) or not isinstance(style, str):
        raise TypeError("selfie scene and style must be strings")
    if custom_prompt is not None and not isinstance(custom_prompt, str):
        raise TypeError("selfie custom prompt must be a string")
    # An empty or whitespace-only custom prompt means "not supplied", as the
    # `if custom_prompt:` gate on origin/main did. An LLM emitting "" for an
    # unused optional argument must still get a scene selfie, not a hard error.
    if custom_prompt is not None and not custom_prompt.strip():
        custom_prompt = None
    if not isinstance(trigger_word, str):
        raise TypeError("LoRA trigger word is invalid")
    # Canonicalize before validating. TOK{companion_name[:8]} slices wherever
    # the name happens to fall, so persisted triggers legitimately carry a
    # trailing space ("Maria J Lopez" -> 'TOKMaria J ') or an embedded tab
    # ("Mary\tJane" -> 'TOKMary\tJan'). Those LoRAs are already trained; the
    # same normalization is applied to prompts, so binding stays comparable.
    trigger_word = normalize_trigger_word(trigger_word)
    # The caller's scene is preserved, never silently swapped for another.
    # SELFIE_SCENE_PROMPTS is a subset of the vocabulary downstream consumers
    # use (frinz tier-gates "shower"/"bedroom"/"spread_eagle", routes them to a
    # different engine, coalesces on it, and names the stored asset by it), so
    # coercing it here made config.scene, config.prompt and spec_sha256
    # describe three different things and collapsed four distinct paid quotes
    # onto one digest. Only the descriptive prompt TEXT depends on the map:
    # an unknown scene contributes none rather than claiming to be "casual".
    normalized_scene = normalize_descriptor(scene)
    if not is_valid_descriptor(normalized_scene):
        raise ValueError("selfie scene is invalid")
    normalized_style = normalize_descriptor(style)
    if not is_valid_descriptor(normalized_style):
        raise ValueError("selfie style is invalid")
    if not is_valid_trigger_word(trigger_word):
        raise ValueError("LoRA trigger word is invalid")
    guidance = _decimal(guidance_scale)
    _validate_generation_parameters(
        seed=seed,
        num_outputs=num_outputs,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
    )

    normalized_custom = _normalize_prompt(custom_prompt)
    trigger_pattern = _trigger_pattern(trigger_word)
    if normalized_custom:
        placeholder_count = normalized_custom.count("TRIGGER_WORD")
        if placeholder_count > 1:
            raise ValueError("selfie custom prompt must bind the trigger once")
        if placeholder_count == 1:
            prompt = normalized_custom.replace("TRIGGER_WORD", trigger_word)
        else:
            trigger_count = len(trigger_pattern.findall(normalized_custom))
            if trigger_count > 1:
                raise ValueError("selfie custom prompt must bind the trigger once")
            prompt = (
                normalized_custom
                if trigger_count == 1
                else f"{trigger_word}, {normalized_custom}"
            )
    else:
        # An unknown scene contributes no invented description, but the
        # "A photo of <trigger>, " prefix shape is kept identical either way:
        # the no-LoRA reference route strips exactly that prefix, and changing
        # it left a subjectless "A photo of . High quality..." behind.
        scene_text = SELFIE_SCENE_PROMPTS.get(normalized_scene)
        prompt = (
            f"A photo of {trigger_word}, {scene_text}. "
            "High quality, photorealistic, 8k."
            if scene_text
            else f"A photo of {trigger_word}, High quality, photorealistic, 8k."
        )
        # Known styles add a prefix; an unrecognized one simply adds none, as
        # origin/main did. frinz forwards style unvalidated from an HTTP body
        # and from LLM tool arguments, so "cinematic" or "realistic" must keep
        # producing a selfie rather than failing the request.
        if normalized_style == "anime":
            prompt = f"anime style illustration, {prompt}"
        elif normalized_style == "artistic":
            prompt = f"artistic portrait painting style, {prompt}"

    if len(trigger_pattern.findall(prompt)) != 1:
        raise ValueError("selfie prompt must bind the trigger once")

    return ResolvedSelfiePrompt(
        scene=normalized_scene,
        style=normalized_style,
        trigger_word=trigger_word,
        prompt=prompt,
        prompt_sha256=_sha256_text(prompt),
        trigger_word_sha256=_sha256_text(trigger_word),
        seed=seed,
        num_outputs=num_outputs,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
    )


def resolve_lora_selfie_spec(
    *,
    scene: str,
    style: str,
    custom_prompt: str | None,
    trigger_word: str,
    lora_version_id: str,
    lora_encrypted_sha256: str,
    lora_plaintext_sha256: str,
    base_model: str,
    model_version: str,
    trainer_version: str,
    flux_version: str | None = None,
    seed: int = 0,
    num_outputs: int = 1,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: Decimal | float | str = Decimal("4.0"),
) -> tuple[ResolvedSelfiePrompt, ResolvedLoraSelfieSpec]:
    """Resolve the executable prompt and its content-free accepted spec."""
    prompt = resolve_selfie_prompt(
        scene=scene,
        style=style,
        custom_prompt=custom_prompt,
        trigger_word=trigger_word,
        seed=seed,
        num_outputs=num_outputs,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    return prompt, bind_lora_selfie_spec(
        resolved_prompt=prompt,
        lora_version_id=lora_version_id,
        lora_encrypted_sha256=lora_encrypted_sha256,
        lora_plaintext_sha256=lora_plaintext_sha256,
        base_model=base_model,
        model_version=model_version,
        trainer_version=trainer_version,
        flux_version=flux_version,
    )


def bind_lora_selfie_spec(
    *,
    resolved_prompt: ResolvedSelfiePrompt,
    lora_version_id: str,
    lora_encrypted_sha256: str,
    lora_plaintext_sha256: str,
    base_model: str,
    model_version: str,
    trainer_version: str,
    flux_version: str | None = None,
) -> ResolvedLoraSelfieSpec:
    """Bind an actual provider prompt to immutable LoRA identity.

    This is the provider-side reconstruction path for the same content-free
    digest created at quote time. Prompt content remains transient.
    """

    if not isinstance(resolved_prompt, ResolvedSelfiePrompt):
        raise TypeError("resolved selfie prompt must use the canonical contract")

    evidence = {
        "schema_version": SELFIE_SPEC_SCHEMA_VERSION,
        "scene": resolved_prompt.scene,
        "style": resolved_prompt.style,
        "prompt_sha256": resolved_prompt.prompt_sha256,
        "trigger_word_sha256": resolved_prompt.trigger_word_sha256,
        "lora_version_id": lora_version_id,
        "lora_encrypted_sha256": lora_encrypted_sha256,
        "lora_plaintext_sha256": lora_plaintext_sha256,
        "base_model": base_model,
        "model_version": model_version,
        "trainer_version": trainer_version,
        "flux_version": flux_version,
        "seed": resolved_prompt.seed,
        "num_outputs": resolved_prompt.num_outputs,
        "width": resolved_prompt.width,
        "height": resolved_prompt.height,
        "num_inference_steps": resolved_prompt.num_inference_steps,
        "guidance_scale": _decimal_text(resolved_prompt.guidance_scale),
    }
    spec = ResolvedLoraSelfieSpec(
        **{**evidence, "guidance_scale": resolved_prompt.guidance_scale},
        spec_sha256=_sha256_json(evidence),
    )
    return spec


def _normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    normalized = " ".join(prompt.split())
    if not normalized or len(normalized) > 8_000:
        raise ValueError("selfie custom prompt is empty or too long")
    return normalized


def _validate_safe_id(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"selfie {label} must be a string")
    if not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"selfie {label} is invalid")


def _validate_sha256(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"selfie {label} digest must be a string")
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"selfie {label} digest is invalid")


def _validate_generation_parameters(
    *,
    seed: object,
    num_outputs: object,
    width: object,
    height: object,
    num_inference_steps: object,
    guidance_scale: object,
) -> None:
    for value, label in (
        (seed, "seed"),
        (num_outputs, "num_outputs"),
        (width, "width"),
        (height, "height"),
        (num_inference_steps, "inference steps"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"selfie {label} must be an integer")
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError("selfie seed must be between 0 and 4294967295")
    if not 1 <= num_outputs <= 4:
        raise ValueError("selfie num_outputs must be between 1 and 4")
    for value, label in ((width, "width"), (height, "height")):
        if value < 256 or value > 4096 or value % 8:
            raise ValueError(f"selfie {label} must be 256..4096 and divisible by 8")
    if not 1 <= num_inference_steps <= 200:
        raise ValueError("selfie inference steps must be between 1 and 200")
    if (
        not isinstance(guidance_scale, Decimal)
        or not guidance_scale.is_finite()
        or not Decimal(0) <= guidance_scale <= Decimal(30)
    ):
        raise ValueError("selfie guidance scale must be between 0 and 30")


def _decimal(value: Decimal | float | str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("selfie guidance scale is invalid") from exc
    if not result.is_finite():
        raise ValueError("selfie guidance scale is invalid")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
