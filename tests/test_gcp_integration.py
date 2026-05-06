"""GCP-as-training-provider integration test for VisualIdentityFeature.

Originally lived in kestrel-sovereign's tests/integration/test_gcp_compute_e2e.py.
Moved here as part of the open-source split (#462) — this test exercises the
visual feature's provider-selection logic, so it belongs with the feature it
tests, not with the framework.

The test asserts that when GCP credentials are present, the
TrainingProviderFactory routes the visual feature to the GCP-backed provider
(currently exposed as `vertex_ai`). It does NOT exercise any GCP code path
beyond the factory's selection — no actual GCP calls are made (the GCP_PROJECT_ID
is set to a sentinel `test-project`).
"""
import os
import pytest

# Module-level skipif: behaves identically to the original test's gate.
pytestmark = pytest.mark.skipif(
    not os.getenv("GCP_PROJECT_ID"),
    reason="GCP_PROJECT_ID not set (set to any value to exercise GCP-as-provider selection)",
)


@pytest.mark.asyncio
async def test_gcp_provider_selected_when_only_gcp_creds_present(monkeypatch):
    """When GCP_PROJECT_ID is the only cloud cred set, the training provider
    factory selects the GCP/vertex provider for VisualIdentityFeature."""
    from kestrel_feature_visual.feature import VisualIdentityFeature

    # Clear other cloud creds so factory can't route elsewhere.
    for k in (
        "VASTAI_API_KEY",
        "RUNPOD_API_KEY",
        "GCP_PROJECT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    feature = VisualIdentityFeature(agent=None)
    await feature.initialize()

    # _ensure_lora_services drives the factory-based provider lookup.
    feature._ensure_lora_services()

    # The factory should have selected a provider, and it should be the
    # GCP/vertex backend — not vastai or runpod.
    assert feature._training_provider is not None, (
        "TrainingProviderFactory did not select a provider despite GCP_PROJECT_ID being set"
    )
    provider_name = feature._training_provider.provider_name.lower()
    assert "vertex" in provider_name or "gcp" in provider_name, (
        f"Expected GCP/vertex provider when only GCP creds set, got: {provider_name!r}"
    )
