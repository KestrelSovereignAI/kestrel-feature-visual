# kestrel-feature-visual

Visual identity for Kestrel Sovereign agents — avatar generation, selfies, and LoRA training for character consistency. Uses Replicate for image generation with optional LoRA training for consistent visual identity across generated images.

## Installation

```bash
uv pip install kestrel-feature-visual
```

The feature is auto-discovered by Kestrel Sovereign via the `kestrel_sovereign.features` entry point — install it alongside `kestrel-sovereign` and the `VisualIdentityFeature` registers itself at startup.

## Configuration

| Variable | Description |
|----------|-------------|
| `REPLICATE_API_TOKEN` | Replicate API token for image generation (required for avatar/selfie tools) |

## Tools provided

- `generate_avatar` — Create a portrait avatar from a description
- `generate_selfie` — Generate a selfie in various scenes (casual, portrait, glamour, flirty, cozy, adventure, mysterious, romantic, playful, dreamy, confident)
- `train_lora` — Kick off LoRA training for character consistency (requires `kestrel-feature-lora` or another training provider installed alongside)

## Canonical LoRA selfie specification

Queue-backed providers must use `resolve_lora_selfie_spec()` when a selfie is
quoted and `bind_lora_selfie_spec()` immediately before dispatch. Both paths
produce the same content-free `spec_sha256` over the final prompt hash,
generation parameters, and immutable promoted-LoRA identity. The plaintext
prompt remains transient and is never part of the persisted evidence object.

The `scene` a caller asks for is preserved, never swapped for another.
`SELFIE_SCENE_PROMPTS` is a deliberate subset of the vocabulary downstream consumers
use, so it governs only the descriptive prompt *text*: a scene outside it contributes
no description rather than silently rendering as `casual`. The scene therefore reads
the same on the resolved prompt, the generation config, the returned result, and
`spec_sha256`, and two different scenes can never share a digest.

The public `ResolvedSelfiePrompt` object carries the exact values that must be
sent to the image worker, including seed, dimensions, inference steps, and
guidance scale. Providers must reject a reconstructed digest that differs from
the accepted quote.

The object re-validates its own invariants on construction, not only inside
`resolve_selfie_prompt()`, because downstream consumers treat the type itself
as the trust boundary. It therefore carries the plaintext `trigger_word`
alongside `trigger_word_sha256` and verifies that the digest matches the
trigger and that `prompt` binds that trigger **exactly once** as a whole token.
A directly constructed instance attesting a trigger the prompt never binds — or
binds twice — is rejected, so a valid `spec_sha256` can never describe a prompt
that does not match it. Carrying the trigger in plaintext discloses nothing
further: it is already contained verbatim in `prompt`.

## Dependencies

- `kestrel-sovereign-sdk>=0.25.0,<1` — base `Feature`, `tool`, and `ToolCategory` interfaces
- `replicate>=1.0.4` — Replicate API client
- `httpx>=0.27.0` — HTTP transport

## Development

```bash
uv pip install -e '.[test]'
uv run pytest
```

## License

Apache-2.0
