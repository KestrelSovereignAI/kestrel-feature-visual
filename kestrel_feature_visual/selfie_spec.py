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
_TRIGGER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STYLES = frozenset({"photorealistic", "anime", "artistic"})


@dataclass(frozen=True, slots=True)
class ResolvedSelfiePrompt:
    """Exact final prompt and generation knobs sent to an image provider."""

    scene: str
    style: str
    prompt: str
    prompt_sha256: str
    trigger_word_sha256: str
    seed: int
    num_outputs: int
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: Decimal


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
    if not isinstance(trigger_word, str):
        raise TypeError("LoRA trigger word is invalid")
    normalized_scene = scene.strip().lower()
    normalized_style = style.strip().lower()
    if normalized_scene not in SELFIE_SCENE_PROMPTS:
        normalized_scene = "casual"
    if normalized_style not in _STYLES:
        raise ValueError("selfie style is unsupported")
    if not _TRIGGER_RE.fullmatch(trigger_word):
        raise ValueError("LoRA trigger word is invalid")
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
    guidance = _decimal(guidance_scale)
    if guidance < Decimal(0) or guidance > Decimal(30):
        raise ValueError("selfie guidance scale must be between 0 and 30")

    normalized_custom = _normalize_prompt(custom_prompt)
    if normalized_custom:
        if "TRIGGER_WORD" in normalized_custom:
            prompt = normalized_custom.replace("TRIGGER_WORD", trigger_word)
        else:
            prompt = f"{trigger_word}, {normalized_custom}"
    else:
        prompt = (
            f"A photo of {trigger_word}, {SELFIE_SCENE_PROMPTS[normalized_scene]}. "
            "High quality, photorealistic, 8k."
        )
        if normalized_style == "anime":
            prompt = f"anime style illustration, {prompt}"
        elif normalized_style == "artistic":
            prompt = f"artistic portrait painting style, {prompt}"

    return ResolvedSelfiePrompt(
        scene=normalized_scene,
        style=normalized_style,
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

    for value, label in (
        (lora_version_id, "LoRA version id"),
        (base_model, "base model"),
        (model_version, "model version"),
        (trainer_version, "trainer version"),
    ):
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError(f"selfie {label} is invalid")
    if flux_version is not None and not _SAFE_ID_RE.fullmatch(flux_version):
        raise ValueError("selfie FLUX version is invalid")
    if not _SHA256_RE.fullmatch(lora_encrypted_sha256):
        raise ValueError("selfie encrypted LoRA digest is invalid")
    if not _SHA256_RE.fullmatch(lora_plaintext_sha256):
        raise ValueError("selfie plaintext LoRA digest is invalid")
    if (
        resolved_prompt.scene not in SELFIE_SCENE_PROMPTS
        or resolved_prompt.style not in _STYLES
        or not resolved_prompt.prompt
        or len(resolved_prompt.prompt) > 8_000
        or resolved_prompt.prompt_sha256 != _sha256_text(resolved_prompt.prompt)
        or not _SHA256_RE.fullmatch(resolved_prompt.trigger_word_sha256)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                resolved_prompt.seed,
                resolved_prompt.num_outputs,
                resolved_prompt.width,
                resolved_prompt.height,
                resolved_prompt.num_inference_steps,
            )
        )
        or not isinstance(resolved_prompt.guidance_scale, Decimal)
        or not resolved_prompt.guidance_scale.is_finite()
        or not 0 <= resolved_prompt.seed <= 4_294_967_295
        or not 1 <= resolved_prompt.num_outputs <= 4
        or not 256 <= resolved_prompt.width <= 4096
        or resolved_prompt.width % 8
        or not 256 <= resolved_prompt.height <= 4096
        or resolved_prompt.height % 8
        or not 1 <= resolved_prompt.num_inference_steps <= 200
        or not Decimal(0) <= resolved_prompt.guidance_scale <= Decimal(30)
    ):
        raise ValueError("resolved selfie prompt is invalid")

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
