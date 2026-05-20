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

## Dependencies

- `kestrel-sovereign-sdk>=0.14.1,<1` — base `Feature`, `tool`, and `ToolCategory` interfaces
- `replicate>=1.0.4` — Replicate API client
- `httpx>=0.27.0` — HTTP transport

## Development

```bash
uv pip install -e '.[test]'
uv run pytest
```

## License

Apache-2.0
