"""Tests for VisualIdentityFeature (extracted package)."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.tools.result import ToolResult, ToolResultStatus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.agent_id = "test-agent"
    agent.storage = MagicMock()
    agent.storage.files = MagicMock()
    agent.storage.files.store_avatar = AsyncMock(return_value="hash-abc123")
    return agent


@pytest_asyncio.fixture
async def feature_standalone():
    """Create a VisualIdentityFeature in standalone mode (no agent)."""
    from kestrel_feature_visual.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=None)
    await feature.initialize()
    return feature


@pytest_asyncio.fixture
async def feature_with_agent(mock_agent):
    """Create a VisualIdentityFeature with a mock agent."""
    from kestrel_feature_visual.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=mock_agent)
    await feature.initialize()
    return feature


# =============================================================================
# Initialization Tests
# =============================================================================

class TestVisualIdentityFeatureInit:
    """Tests for VisualIdentityFeature initialization."""

    @pytest.mark.asyncio
    async def test_standalone_mode_initialization(self):
        """Test initialization without an agent (standalone mode)."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=None)
        await feature.initialize()

        assert feature.agent is None
        assert feature.name == "VisualIdentityFeature"

    @pytest.mark.asyncio
    async def test_with_agent_initialization(self, mock_agent):
        """Test initialization with an agent."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=mock_agent)
        await feature.initialize()

        assert feature.agent is mock_agent

    def test_tool_description(self):
        """Test the feature has a description."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        feature = VisualIdentityFeature(agent=None)
        desc = feature.tool_description

        assert "visual" in desc.lower()
        assert "avatar" in desc.lower() or "selfie" in desc.lower()


# =============================================================================
# Scene Prompts Tests
# =============================================================================

class TestScenePrompts:
    """Tests for scene prompt configuration."""

    def test_all_core_scenes_defined(self):
        """Verify all core scene types have prompts."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        expected_scenes = [
            "portrait", "casual", "glamour", "flirty", "cozy",
            "adventure", "mysterious", "romantic", "playful",
            "dreamy", "confident"
        ]

        for scene in expected_scenes:
            assert scene in VisualIdentityFeature.SCENE_PROMPTS, f"Missing scene: {scene}"

    def test_scene_prompts_non_empty(self):
        """Verify all scene prompts have content."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        for scene, prompt in VisualIdentityFeature.SCENE_PROMPTS.items():
            assert len(prompt) > 10, f"Scene '{scene}' prompt too short"


# =============================================================================
# Tool Decorator Tests
# =============================================================================

class TestToolDecorators:
    """Tests for tool decorator presence and configuration."""

    @pytest.mark.asyncio
    async def test_generate_selfie_has_tool_decorator(self, feature_standalone):
        """Verify generate_selfie has the @tool decorator."""
        assert hasattr(feature_standalone.generate_selfie, "_tool_schema")
        schema = feature_standalone.generate_selfie._tool_schema
        assert schema["name"] == "generate_selfie"

    @pytest.mark.asyncio
    async def test_generate_avatar_has_tool_decorator(self, feature_standalone):
        """Verify generate_avatar has the @tool decorator."""
        assert hasattr(feature_standalone.generate_avatar, "_tool_schema")
        schema = feature_standalone.generate_avatar._tool_schema
        assert schema["name"] == "generate_avatar"

    @pytest.mark.asyncio
    async def test_train_lora_has_tool_decorator(self, feature_standalone):
        """Verify train_lora has the @tool decorator."""
        assert hasattr(feature_standalone.train_lora, "_tool_schema")
        schema = feature_standalone.train_lora._tool_schema
        assert schema["name"] == "train_lora"


# =============================================================================
# Generate Avatar Tests
# =============================================================================

class TestGenerateAvatar:
    """Tests for generate_avatar tool."""

    @pytest.mark.asyncio
    async def test_avatar_when_disabled(self, feature_standalone):
        """Test avatar generation when service is disabled."""
        feature_standalone.enabled = False
        feature_standalone.service = None

        result = await feature_standalone.generate_avatar(
            description="A friendly looking person"
        )

        assert isinstance(result, ToolResult)
        assert result.status is ToolResultStatus.ERROR
        assert "not available" in result.error.lower()


# =============================================================================
# Generate Selfie Tests
# =============================================================================

class TestGenerateSelfie:
    """Tests for generate_selfie error handling."""

    @pytest.mark.asyncio
    async def test_selfie_when_disabled(self, feature_standalone):
        """Test selfie generation when service is disabled."""
        feature_standalone.enabled = False
        feature_standalone.service = None

        result = await feature_standalone.generate_selfie(scene="casual")

        assert isinstance(result, ToolResult)
        assert result.status is ToolResultStatus.ERROR
        assert ("not available" in result.error.lower()
                or "replicate" in result.error.lower())


# =============================================================================
# GenerationConfig companion-context helper
# =============================================================================

class TestBuildGenerationConfig:
    """Tests for the companion-context GenerationConfig builder."""

    def test_carries_companion_context(self):
        """Populated companion-context fields land on the config."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        config = VisualIdentityFeature._build_generation_config(
            prompt="a photo",
            companion_id="comp-123",
            companion_did="did:key:abc",
            scene="beach",
            avatar_reference_url="https://avatar.example/a.png",
            requested_by="user-9",
            engine_hint="pulid-avatar",
        )

        assert config.prompt == "a photo"
        assert config.companion_id == "comp-123"
        assert config.companion_did == "did:key:abc"
        assert config.scene == "beach"
        assert config.avatar_reference_url == "https://avatar.example/a.png"
        assert config.requested_by == "user-9"
        assert config.engine_hint == "pulid-avatar"

    def test_unset_context_fields_default_to_none(self):
        """Unset companion-context fields default to None; lora_path optional."""
        from kestrel_feature_visual.feature import VisualIdentityFeature

        config = VisualIdentityFeature._build_generation_config(prompt="p")

        assert config.lora_path is None
        assert config.companion_id is None
        assert config.companion_did is None
        assert config.scene is None
        assert config.avatar_reference_url is None
        assert config.requested_by is None
        assert config.engine_hint is None


# =============================================================================
# No-LoRA reference-image route
# =============================================================================

class _Capabilities:
    """Minimal stand-in for ProviderCapabilities.supports_reference_image."""

    def __init__(self, supports_reference_image: bool):
        self.supports_reference_image = supports_reference_image


class _FakeProvider:
    """Provider double recording the config it receives."""

    def __init__(self, provider_name: str, supports_reference_image: bool):
        self.provider_name = provider_name
        self.capabilities = _Capabilities(supports_reference_image)
        self.received_config = None
        self.generate_calls = 0

    async def generate_image(self, config, **kwargs):
        from kestrel_sovereign.features.training import (
            GenerationResult,
            GenerationState,
        )

        self.received_config = config
        self.generate_calls += 1
        return GenerationResult(
            job_id="job-1",
            state=GenerationState.COMPLETED,
            images=["data:image/png;base64,ZmFrZQ=="],
            elapsed_seconds=1.5,
        )


class _QueueProvider:
    """Queue-based provider double: accepts the job and returns a non-terminal
    ``pending`` result carrying a ``job_id`` but no image (frinz #558 shape)."""

    def __init__(self, provider_name: str = "catalog_worker"):
        self.provider_name = provider_name
        self.capabilities = _Capabilities(True)
        self.received_config = None
        self.generate_calls = 0

    async def generate_image(self, config, **kwargs):
        from kestrel_sovereign.features.training import (
            GenerationResult,
            GenerationState,
        )

        self.received_config = config
        self.generate_calls += 1
        # Enqueue accepted: no image yet, only a job handle to poll.
        return GenerationResult(
            job_id="queue-job-42",
            state=GenerationState.PENDING,
            images=[],
        )


class _FakeConn:
    """Minimal asyncpg-connection double returning a fixed row for fetchrow."""

    def __init__(self, row):
        self._row = row

    async def fetchrow(self, query, *args):
        return self._row


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    """db_pool double: every ``acquire()`` yields a conn returning ``row``.

    ``row`` is a plain dict, which supports both ``row["k"]`` and ``row.get``
    exactly like an ``asyncpg.Record``, so it satisfies every companions
    lookup the feature performs.
    """

    def __init__(self, row):
        self._conn = _FakeConn(row)

    def acquire(self):
        return _FakeAcquire(self._conn)


class TestNoLoraReferenceRoute:
    """Tests for the no-LoRA / PuLID-avatar reference-image route."""

    @pytest.mark.asyncio
    async def test_no_lora_routes_to_reference_provider(self, feature_standalone):
        """No LoRA + supports_reference_image provider → generate_image(config)
        with companion_id / avatar_reference_url / scene populated."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None  # no existing LoRA lookup

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            reference_image="https://avatar.example/a.png",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.OK
        assert provider.generate_calls == 1

        config = provider.received_config
        assert config is not None
        assert config.companion_id == "comp-123"
        assert config.avatar_reference_url == "https://avatar.example/a.png"
        assert config.scene == "beach"
        assert config.lora_path is None
        assert config.engine_hint == "pulid-avatar"

        assert result.data["used_lora"] is False
        assert result.data["reference_used"] is True
        assert result.data["image_url"].startswith("data:image/png")
        assert result.data["backend"] == "catalog_worker"

    @pytest.mark.asyncio
    async def test_no_lora_no_reference_provider_falls_back_to_training(
        self, feature_standalone
    ):
        """No LoRA + no reference-capable provider + allow_training=False →
        the existing needs_training failure (unchanged behavior)."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        provider = _FakeProvider("runpod", supports_reference_image=False)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        with patch(
            "kestrel_feature_visual.feature.TrainingProviderFactory"
        ) as factory:
            factory.get_generation_provider.return_value = None
            result = await feature.generate_selfie(
                scene="casual",
                companion_id="comp-123",
                allow_training=False,
            )

        assert result.status is ToolResultStatus.ERROR
        assert result.data.get("needs_training") is True
        assert provider.generate_calls == 0

    @pytest.mark.asyncio
    async def test_lora_path_unchanged_by_reference_route(self, feature_standalone):
        """A caller passing an explicit LoRA path still uses the LoRA path and
        never routes through the reference-image branch (regression)."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        provider = _FakeProvider("runpod", supports_reference_image=False)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="casual",
            companion_id="comp-123",
            lora_model_path="gs://bucket/lora.safetensors",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.OK
        assert result.data["used_lora"] is True
        # The LoRA path sends the LoRA weights, not a None reference config.
        assert provider.received_config.lora_path == "gs://bucket/lora.safetensors"

    @pytest.mark.asyncio
    async def test_queued_provider_result_is_success(self, feature_standalone):
        """A queue-based provider that returns pending + job_id (no image yet)
        is a successful async enqueue, not a failure."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        provider = _QueueProvider("catalog_worker")
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            reference_image="https://avatar.example/a.png",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.OK
        assert provider.generate_calls == 1
        assert result.data["queued"] is True
        assert result.data["job_id"] == "queue-job-42"
        assert result.data["status"] == "pending"
        assert result.data["reference_used"] is True
        assert result.data["used_lora"] is False
        assert result.data["backend"] == "catalog_worker"
        # No image URL is present yet — it lands out-of-band.
        assert "image_url" not in result.data

    @pytest.mark.asyncio
    async def test_companion_did_and_avatar_config_url_passed(
        self, feature_standalone
    ):
        """The no-LoRA route looks up companion ``did`` and the
        ``avatar_config['url']`` reference and passes both to the provider."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = _FakePool(
            {
                "image_url": None,  # wizard companions store the ref under url
                "avatar_config": {"url": "https://wizard.example/av.png"},
                "did": "did:key:xyz",
            }
        )

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.OK
        assert provider.generate_calls == 1
        config = provider.received_config
        assert config.companion_did == "did:key:xyz"
        assert config.avatar_reference_url == "https://wizard.example/av.png"

    @pytest.mark.asyncio
    async def test_no_avatar_url_does_not_dispatch_reference_route(
        self, feature_standalone
    ):
        """No LoRA + reference-capable provider but NO resolvable avatar URL →
        do not enqueue an unrouteable job; fall through to the training/fail
        behavior instead."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = _FakePool(
            {"image_url": None, "avatar_config": {}, "did": "did:key:xyz"}
        )

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="casual",
            companion_id="comp-123",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.ERROR
        assert result.data.get("needs_training") is True
        assert provider.generate_calls == 0


# =============================================================================
# Train LoRA Tests
# =============================================================================


class TestTrainLora:
    """Tests for train_lora tool's ToolResult contract on the failure
    paths reachable without RunPod credentials. The success path is
    not exercised here because it requires real cloud creds; the
    @cloud_resource integration tests cover that surface."""

    @pytest.mark.asyncio
    async def test_train_lora_without_companion_id_returns_failed(
        self, feature_standalone
    ):
        """No companion_id and no agent.companion_context to fall back on
        → ToolResult.failed with a descriptive error."""
        result = await feature_standalone.train_lora()

        assert isinstance(result, ToolResult)
        assert result.status is ToolResultStatus.ERROR
        assert "companion_id" in result.error.lower()
