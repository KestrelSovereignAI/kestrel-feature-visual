"""Tests for VisualIdentityFeature (extracted package)."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
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
        self.received_kwargs = None
        self.generate_calls = 0

    async def generate_image(self, config, **kwargs):
        from kestrel_sovereign.features.training import (
            GenerationResult,
            GenerationState,
        )

        self.received_config = config
        self.received_kwargs = kwargs
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


class _FailedProvider:
    """Provider double that records its config and rejects generation."""

    provider_name = "catalog_worker"
    capabilities = _Capabilities(True)

    def __init__(self):
        self.received_config = None

    async def generate_image(self, config, **kwargs):
        from kestrel_sovereign.features.training import (
            GenerationResult,
            GenerationState,
        )

        self.received_config = config
        return GenerationResult(
            job_id="queue-job-rejected",
            state=GenerationState.FAILED,
            images=[],
            error="catalog rejected LoRA binding",
        )


class TestTrainedLoraContext:
    """Authenticated context carried through the trained-LoRA route (#18)."""

    @staticmethod
    def _promoted_row():
        return {
            "image_url": "https://stored.example/avatar.png",
            "did": "did:key:companion-123",
            "avatar_config": {
                "url": "https://older.example/avatar.png",
                "lora_model_path": (
                    "catalog://companions/comp-123/lora/sha256:promoted"
                ),
                "lora_ipfs_cid": "bafy-promoted-lora",
                "trigger_word": "TOKPROMOTED",
                "flux_version": "flux1",
            },
        }

    @pytest.mark.asyncio
    async def test_queue_receives_complete_trained_lora_context(
        self, feature_standalone
    ):
        """The promoted-LoRA path uses the canonical config builder and a
        queue acceptance remains an honest queued success with LoRA context."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = _FakePool(self._promoted_row())

        provider = _QueueProvider("catalog_worker")
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="beach",
            reference_image="https://caller.example/untrusted-avatar.png",
            companion_id="comp-123",
            allow_training=False,
            requested_by="user-9",
        )

        assert result.status is ToolResultStatus.OK
        assert provider.generate_calls == 1
        config = provider.received_config
        assert config.prompt.startswith("A photo of TOKPROMOTED")
        assert config.lora_path == (
            "catalog://companions/comp-123/lora/sha256:promoted"
        )
        assert config.trigger_word == "TOKPROMOTED"
        assert config.companion_id == "comp-123"
        assert config.companion_did == "did:key:companion-123"
        assert config.scene == "beach"
        # The trained path carries only the server-owned avatar reference;
        # caller input cannot replace it.
        assert config.avatar_reference_url == "https://stored.example/avatar.png"
        assert config.requested_by == "user-9"
        assert provider.received_kwargs == {
            "lora_ipfs_cid": "bafy-promoted-lora",
            "flux_version": "flux1",
        }

        assert result.data["queued"] is True
        assert result.data["job_id"] == "queue-job-42"
        assert result.data["status"] == "pending"
        assert result.data["companion_id"] == "comp-123"
        assert result.data["used_lora"] is True
        assert result.data["reference_used"] is False
        assert "image_url" not in result.data

    @pytest.mark.asyncio
    async def test_bound_agent_identity_wins_on_trained_lora_path(
        self, feature_standalone
    ):
        """A prompt-injected owner cannot override the bound agent user, and
        an equal tool companion is normalized to the host-owned identity."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = _FakePool(self._promoted_row())
        bound_companion_id = UUID("00000000-0000-0000-0000-000000000123")

        class _Agent:
            def __init__(self):
                self.companion_context = {
                    "companion_id": bound_companion_id,
                    "user_id": "bound-user-7",
                }

        feature.agent = _Agent()
        provider = _QueueProvider("catalog_worker")
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="cozy",
            # Same identity, but from the tool boundary as a string. The
            # authoritative host-owned UUID must be carried to the provider.
            companion_id=str(bound_companion_id),
            allow_training=False,
            requested_by="attacker-supplied-user",
        )

        assert result.status is ToolResultStatus.OK
        assert provider.received_config.companion_id is bound_companion_id
        assert provider.received_config.requested_by == "bound-user-7"
        assert provider.received_config.scene == "cozy"
        assert result.data["used_lora"] is True

    @pytest.mark.asyncio
    async def test_standalone_requested_by_is_preserved_for_trained_lora(
        self, feature_standalone
    ):
        """The trusted non-agent caller remains able to attribute the job."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = _FakePool(self._promoted_row())

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="portrait",
            companion_id="comp-123",
            allow_training=False,
            requested_by="authenticated-rest-user",
        )

        assert result.status is ToolResultStatus.OK
        assert provider.received_config.requested_by == "authenticated-rest-user"
        assert provider.received_config.companion_id == "comp-123"
        assert result.data["used_lora"] is True

    @pytest.mark.asyncio
    async def test_rejected_trained_lora_request_never_claims_lora_use(
        self, feature_standalone
    ):
        """A provider failure after receiving context is still a failure and
        cannot emit the post-LoRA ``used_lora=true`` success marker."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = _FakePool(self._promoted_row())

        provider = _FailedProvider()
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            allow_training=False,
            requested_by="user-9",
        )

        assert provider.received_config.companion_id == "comp-123"
        assert provider.received_config.lora_path.endswith("sha256:promoted")
        assert result.status is ToolResultStatus.ERROR
        assert "catalog rejected LoRA binding" in result.error
        assert not result.data or "used_lora" not in result.data

    @pytest.mark.asyncio
    async def test_bound_companion_mismatch_stops_before_context_lookup(
        self, feature_standalone
    ):
        """A tool-supplied companion cannot redirect a bound LoRA request."""
        feature = feature_standalone
        feature.enabled = True

        class _Agent:
            def __init__(self):
                self.companion_context = {
                    "companion_id": "bound-companion",
                    "user_id": "bound-user",
                }

        feature.agent = _Agent()
        provider = _QueueProvider("catalog_worker")
        feature._get_training_provider = lambda *a, **k: provider

        async def _lookup_must_not_run(*_args, **_kwargs):
            raise AssertionError("tenant guard allowed a server-side lookup")

        feature._lookup_companion_did = _lookup_must_not_run
        feature._lookup_avatar_url = _lookup_must_not_run

        result = await feature.generate_selfie(
            scene="casual",
            companion_id="different-companion",
            lora_model_path="catalog://attacker/lora",
            requested_by="different-user",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.ERROR
        assert result.data["code"] == "companion_id_mismatch"
        assert provider.generate_calls == 0


class TestNoLoraReferenceRoute:
    """Tests for the no-LoRA / PuLID-avatar reference-image route."""

    @pytest.mark.asyncio
    async def test_no_lora_routes_to_reference_provider(self, feature_standalone):
        """No LoRA + supports_reference_image provider → generate_image(config)
        with companion_id / avatar_reference_url / scene populated.

        Mocks _lookup_avatar_url so the SSRF-guarded route uses a
        server-owned URL rather than the caller-supplied reference_image
        (codex round-2 P1: unresolvable / private-host caller URLs are
        rejected by _reference_url_is_safe)."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None  # no existing LoRA lookup

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        async def _fake_lookup(_cid):
            return "https://avatar.example/a.png"

        feature._lookup_avatar_url = _fake_lookup

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
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

        # See test_no_lora_routes_to_reference_provider for why we mock the
        # server-owned URL lookup instead of relying on caller-supplied
        # reference_image (codex round-2 P1 SSRF guard).
        async def _fake_lookup(_cid):
            return "https://avatar.example/a.png"
        feature._lookup_avatar_url = _fake_lookup

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
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

    @pytest.mark.asyncio
    async def test_caller_reference_url_rejected_when_no_stored_avatar(
        self, feature_standalone
    ):
        """Codex round-2 P1 SSRF regression: when the companion has no
        server-owned avatar and the caller-supplied reference_image is not
        SSRF-safe (unresolvable / private-host / non-http), the no-LoRA
        route MUST NOT dispatch to the provider. Otherwise a caller can
        aim the queue worker at localhost / 169.254.169.254 / RFC1918."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        async def _no_stored(_cid):
            return None
        feature._lookup_avatar_url = _no_stored

        # Rogue URL: http scheme, private-host, would SSRF the worker.
        result = await feature.generate_selfie(
            scene="casual",
            companion_id="comp-123",
            reference_image="http://169.254.169.254/latest/meta-data/",
            allow_training=False,
        )

        # Falls through to the LoRA-required error (needs_training) rather
        # than dispatching to the provider.
        assert result.status is ToolResultStatus.ERROR
        assert provider.generate_calls == 0

    @pytest.mark.asyncio
    async def test_explicit_requested_by_reaches_provider_config(
        self, feature_standalone
    ):
        """REST path: an explicit ``requested_by`` propagates into the
        provider's ``GenerationConfig.requested_by`` (issue #12). Frinz's
        REST endpoint passes the authenticated user id here so the
        dashboard's requested_by-scoped poll can find the queued job."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        async def _fake_lookup(_cid):
            return "https://avatar.example/a.png"
        feature._lookup_avatar_url = _fake_lookup

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            allow_training=False,
            requested_by="user-1",
        )

        assert result.status is ToolResultStatus.OK
        assert provider.generate_calls == 1
        assert provider.received_config.requested_by == "user-1"

    @pytest.mark.asyncio
    async def test_chat_path_auto_fills_requested_by_from_agent_context(
        self, feature_standalone
    ):
        """Regression: chat path with no explicit ``requested_by`` still
        auto-fills it from the agent's companion_context user_id."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        class _Agent:
            companion_context = {
                "companion_id": "comp-123",
                "user_id": "ctx-user-7",
            }
        feature.agent = _Agent()

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        async def _fake_lookup(_cid):
            return "https://avatar.example/a.png"
        feature._lookup_avatar_url = _fake_lookup

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.OK
        assert provider.generate_calls == 1
        assert provider.received_config.requested_by == "ctx-user-7"

    @pytest.mark.asyncio
    async def test_agent_context_wins_over_tool_supplied_requested_by(
        self, feature_standalone
    ):
        """Codex round-2 P1 on #12: ``generate_selfie`` is an @tool so every
        kwarg is LLM-controllable via prompt injection. When an agent context
        is bound, tool-supplied ``requested_by`` MUST be ignored — otherwise a
        prompt-injected chat message could attribute the queued job to another
        user (cross-tenant leak: their dashboard's requested_by-scoped poll
        would then find it). Only the standalone REST path (no agent context)
        honors the explicit kwarg."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        class _Agent:
            companion_context = {
                "companion_id": "comp-123",
                "user_id": "ctx-user-7",
            }
        feature.agent = _Agent()

        provider = _FakeProvider("catalog_worker", supports_reference_image=True)
        feature._get_training_provider = lambda *a, **k: provider
        feature._ensure_lora_services = lambda: False

        async def _fake_lookup(_cid):
            return "https://avatar.example/a.png"
        feature._lookup_avatar_url = _fake_lookup

        result = await feature.generate_selfie(
            scene="beach",
            companion_id="comp-123",
            allow_training=False,
            requested_by="attacker-supplied-other-user",
        )

        assert result.status is ToolResultStatus.OK
        # Agent context wins; the attacker-supplied value is ignored.
        assert provider.received_config.requested_by == "ctx-user-7"

    @pytest.mark.asyncio
    async def test_tenant_boundary_refuses_mismatched_companion_id(
        self, feature_standalone
    ):
        """Codex round-2 P1 tenant regression: when the agent has a bound
        companion_context, a tool-call companion_id that doesn't match must
        be refused before any avatar / did lookup fires — otherwise a
        caller can drive selfie generation and vault-write against another
        user's companion."""
        feature = feature_standalone
        feature.enabled = True
        feature.db_pool = None

        # Simulate a per-companion agent binding.
        class _Agent:
            companion_context = {
                "companion_id": "bound-owner-id",
                "user_id": "user-1",
            }
        feature.agent = _Agent()

        # If the tenant guard fails, the code path would try to look up
        # avatar/DID and possibly dispatch. Blow up loudly if that happens.
        async def _boom(*_a, **_k):
            raise AssertionError("tenant guard bypassed — lookup fired")
        feature._lookup_avatar_url = _boom
        feature._lookup_companion_did = _boom

        result = await feature.generate_selfie(
            scene="casual",
            companion_id="attacker-supplied-different-id",
            allow_training=False,
        )

        assert result.status is ToolResultStatus.ERROR
        assert result.data.get("code") == "companion_id_mismatch"
        assert result.data.get("status_code") == 403


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
