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

`scene` and `style` are **caller-owned**. This package normalizes them (whitespace
collapsed, lowercased) and bounds their length, but never rejects one for failing to
appear in its own tables — frinz forwards both unvalidated from an HTTP body and from
LLM tool arguments, and deliberately supports free-form prose scenes such as
`stargazing at night with aurora borealis`.

`SELFIE_SCENE_PROMPTS` and the style prefixes therefore govern only the descriptive
prompt *text*: an unrecognized scene contributes no description rather than silently
rendering as `casual`, and an unrecognized style adds no prefix. The scene reads the
same on the resolved prompt, the generation config, the returned result, and
`spec_sha256`, so two different scenes can never share a digest.

On the no-LoRA reference route, when there is neither a custom prompt nor a known
scene description this package sends **no** prompt override, so the catalog worker's
own scene template is used instead of a subjectless stub.

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
