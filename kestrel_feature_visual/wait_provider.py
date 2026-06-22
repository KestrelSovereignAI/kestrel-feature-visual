"""Waitable provider for LoRA training jobs (``lora_train:<companion_id>:<job_id>``).

Wraps the unified training provider's ``get_status`` poll — the same single
status read the blocking loop in
:meth:`~kestrel_feature_visual.feature.VisualIdentityFeature._train_lora_for_companion`
uses — and classifies it onto the generic :class:`Outcome` vocabulary. The
poll loop, cap, and ToolResult mapping live in the host's wait engine
(``kestrel_sovereign.waits.engine``); this provider only reads and classifies
one observation.

The wait engine splits a ``"<kind>:<rest>"`` reference on the FIRST ``:`` only,
so the *handle* this provider receives is ``"<companion_id>:<job_id>"`` — a
self-describing handle that carries the companion to finalize against without a
durable in-flight registry. We split it again on the first ``:`` (a job_id may
itself contain colons, so the companion_id is the head and the job_id the
remainder).

CRITICAL — finalize-once: a Waitable that reported DONE without performing the
finalization the blocking loop performs would leave the GPU pod BILLING and the
LoRA unpersisted (codex P1). On a terminal poll this provider therefore routes
through the feature's SINGLE idempotent
:meth:`VisualIdentityFeature._finalize_training`, which is guarded so the
blocking path and this path together run the terminal side effects exactly
once.

Auto-wake: dispatch now records each in-flight job durably in the companion's
``avatar_config`` (``lora_training_status="running"``), so this provider
implements ``active_handles`` (making it a structural ``MonitorableWaitable``).
The host reconciler enumerates those handles and wakes the agent when training
completes even if no one explicitly waited. ``signal`` is still ``None`` because
there is no terminal-state signal emitter for this kind yet; the reconciler
classifies via ``poll``.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

from kestrel_sdk.tools import Outcome, WaitStatus


class LoraTrainingWaitable:
    """Polls a LoRA training job by ``<companion_id>:<job_id>`` handle."""

    kind: ClassVar[str] = "lora_train"
    # No terminal-state signal emitter for training yet, so there is no
    # signal-resume rail to drive; blocking waits and explicit mode="signal"
    # watches both still work via the host reconciler polling this provider.
    signal: ClassVar[Optional[str]] = None

    def __init__(self, feature: "object") -> None:
        # The owning VisualIdentityFeature; provides _training_provider and the
        # shared idempotent _finalize_training.
        self._feature = feature

    @staticmethod
    def _parse(handle: str) -> Tuple[str, str]:
        """Split ``"<companion_id>:<job_id>"`` on the FIRST ``:``.

        The companion_id is the head; the job_id is the remainder (and MAY
        itself contain colons, e.g. ``"runpod:abc-123"``).
        """
        companion_id, sep, job_id = handle.partition(":")
        if not sep:
            # No colon: treat the whole thing as the job_id with no companion.
            return "", handle
        return companion_id, job_id

    async def poll(self, handle: str) -> WaitStatus:
        companion_id, job_id = self._parse(handle)

        # Reject malformed handles (missing the companion prefix or job id)
        # BEFORE touching the provider. A bare ``lora_train:<job_id>`` would
        # otherwise finalize the real job against companion_id="" — updating
        # zero rows, marking the bare job_id in ``_finalized_jobs``, and tearing
        # down the pod — so a later correct ``companion:job_id`` poll would be
        # skipped by the guard, stranding the real companion at "running" with no
        # LoRA. Handles always come from this feature's own dispatch as
        # ``<companion_id>:<job_id>``; anything else is a caller error.
        if not companion_id or not job_id:
            return WaitStatus(
                Outcome.FAILED,
                f"malformed LoRA wait handle {handle!r}; expected '<companion_id>:<job_id>'",
                data={"companion_id": companion_id, "job_id": job_id},
            )

        # Lazily (re)initialize the provider. After a restart, the feature's
        # _training_provider is None until some tool calls _ensure_lora_services,
        # but active_handles can still surface persisted ``running`` jobs — so
        # recreate the configured provider here rather than reporting FAILED and
        # breaking auto-wake/reconciliation.
        #
        # Use the provider the job was DISPATCHED on, recorded in the
        # companion's avatar_config (``lora_provider``), not just the current
        # default: in a multi-provider host — or after provider priority changed
        # while the job ran — polling the default backend would query the wrong
        # service and could fail or finalize with the wrong provider metadata.
        recorded_provider = await self._recorded_provider(companion_id)
        provider = None
        getter = getattr(self._feature, "_get_training_provider", None)
        if callable(getter):
            try:
                provider = getter(recorded_provider) if recorded_provider else getter()
            except Exception:
                provider = None
        if provider is None:
            provider = getattr(self._feature, "_training_provider", None)
        if provider is None:
            return WaitStatus(
                Outcome.FAILED,
                "training provider unavailable",
                data={"companion_id": companion_id, "job_id": job_id},
            )

        status = await provider.get_status(job_id)

        # Resolve terminal classification defensively. Prefer the host
        # TrainingState enum, but fall back to comparing the raw state value so
        # this provider works even when the host type is not importable (e.g.
        # the feature is loaded without kestrel-sovereign core present).
        state_value = self._state_value(status.state)
        terminal_done, terminal_failed, terminal_cancelled = self._classify(status.state, state_value)

        provider_name = getattr(provider, "provider_name", None)
        progress = getattr(status, "progress", None)
        error = getattr(status, "error", None)
        # Pass the provider-specific detail dict straight through to the
        # finalizer; it resolves the real output path (e.g. Vertex AI's
        # ``gcs_output_path``) from it. ``TrainingStatus.provider_details`` is a
        # dict (defaults to ``{}``); tolerate a missing attr defensively.
        provider_details = getattr(status, "provider_details", None) or {}
        output_path = getattr(status, "output_path", None) or provider_details.get("output_path")

        payload = {
            "companion_id": companion_id,
            "job_id": job_id,
            "state": state_value,
            "progress": progress,
        }

        if terminal_done or terminal_failed or terminal_cancelled:
            # Route through the shared idempotent finalizer so the pod is torn
            # down and (on success) the LoRA persisted exactly once, whether
            # this provider or the blocking loop observed terminal first.
            terminal_state = self._terminal_state_for(
                state_value, terminal_done, terminal_failed
            )
            lora_path = await self._feature._finalize_training(
                companion_id=companion_id,
                job_id=job_id,
                terminal_state=terminal_state,
                provider=provider_name,
                # trigger_word intentionally None: the finalizer recovers the
                # canonical trigger_word from the dispatch-recorded
                # avatar_config rather than clobbering it with null.
                trigger_word=None,
                output_path=output_path,
                error=error,
                provider_details=provider_details,
            )

            # The finalizer only adds the job to ``_finalized_jobs`` once its
            # terminal persistence has SUCCEEDED (or there was no db_pool to
            # persist into). If a transient DB error left the job UNfinalized,
            # the terminal result is not yet durable — report PENDING so the
            # wait loop / reconciler keeps polling and retries the write rather
            # than telling the caller the LoRA is ready when it isn't.
            finalized = getattr(self._feature, "_finalized_jobs", None)
            if finalized is not None and job_id not in finalized:
                payload["finalization_pending"] = True
                return WaitStatus(
                    Outcome.PENDING,
                    f"LoRA training {job_id} terminal but finalization not yet persisted; retrying",
                    data=payload,
                )

            if terminal_done:
                payload["lora_path"] = lora_path or output_path or (
                    f"{provider_name}:{job_id}" if provider_name else None
                )
                return WaitStatus(
                    Outcome.DONE,
                    f"LoRA training {job_id} completed",
                    data=payload,
                )

            payload["error"] = error
            verb = "failed" if terminal_failed else "cancelled"
            return WaitStatus(
                Outcome.FAILED,
                error or f"LoRA training {job_id} {verb}",
                data=payload,
            )

        return WaitStatus(
            Outcome.PENDING,
            f"LoRA training {job_id} status: {state_value}",
            data=payload,
        )

    # ------------------------------------------------------------------
    # State classification (defensive against an unimportable host enum)
    # ------------------------------------------------------------------

    @staticmethod
    def _state_value(state: object) -> str:
        """Best-effort string for the state (``.value`` if it's an enum)."""
        return str(getattr(state, "value", state))

    def _classify(self, state: object, state_value: str) -> Tuple[bool, bool, bool]:
        """Return ``(is_done, is_failed, is_cancelled)`` for a state.

        Tries the host :class:`TrainingState` enum first; falls back to a
        string compare on the lowercased value so the provider remains correct
        when kestrel-sovereign core is not importable.
        """
        try:
            from kestrel_sovereign.features.training.types import TrainingState

            return (
                state == TrainingState.COMPLETED,
                state == TrainingState.FAILED,
                state == TrainingState.CANCELLED,
            )
        except Exception:
            v = state_value.lower()
            return (v == "completed", v == "failed", v == "cancelled")

    def _terminal_state_for(
        self, state_value: str, terminal_done: bool, terminal_failed: bool
    ) -> object:
        """Build the ``terminal_state`` arg for ``_finalize_training``.

        Returns the host :class:`TrainingState` enum member when importable,
        else a tiny shim whose ``== TrainingState.X`` comparisons in the
        finalizer still resolve via the host enum inside that method.
        """
        try:
            from kestrel_sovereign.features.training.types import TrainingState

            if terminal_done:
                return TrainingState.COMPLETED
            if terminal_failed:
                return TrainingState.FAILED
            return TrainingState.CANCELLED
        except Exception:
            # _finalize_training imports TrainingState itself; if that import
            # fails there too, this fallback value is compared and won't match
            # COMPLETED, so finalization safely treats it as non-success.
            if terminal_done:
                return "completed"
            if terminal_failed:
                return "failed"
            return "cancelled"

    # ------------------------------------------------------------------
    # Auto-wake enumeration (MonitorableWaitable)
    # ------------------------------------------------------------------

    async def active_handles(self) -> list[str]:
        """Enumerate in-flight training jobs as ``"<companion_id>:<job_id>"``.

        Reads the durable in-flight records dispatch writes into
        ``companions.avatar_config`` (``lora_training_status="running"``) and
        returns one handle per running job so the host reconciler can poll and
        wake on completion without anyone having explicitly waited.

        Cheap, side-effect-free, tolerant of a missing db_pool: returns ``[]``
        when nothing is in flight or no pool is configured.
        """
        db_pool = getattr(self._feature, "db_pool", None)
        if db_pool is None:
            return []

        handles: list[str] = []
        try:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, avatar_config FROM companions "
                    "WHERE avatar_config->>'lora_training_status' = 'running'"
                )
        except Exception:
            return []

        for row in rows or []:
            companion_id = row["id"]
            raw_config = row["avatar_config"]
            config = self._coerce_config(raw_config)
            if not config:
                continue
            job_id = config.get("lora_job_id")
            if companion_id and job_id:
                handles.append(f"{companion_id}:{job_id}")
        return handles

    async def _recorded_provider(self, companion_id: str) -> Optional[str]:
        """Return the provider this companion's job was dispatched on, if known.

        Reads the dispatch-recorded ``lora_provider`` from the feature's
        in-flight metadata so a reconciler waking a persisted job polls the
        SAME backend it was started on, not the current default. Tolerant of a
        missing helper / db / row — returns ``None`` to fall back to default.
        """
        if not companion_id:
            return None
        lookup = getattr(self._feature, "_lookup_inflight_metadata", None)
        if not callable(lookup):
            return None
        try:
            recorded = await lookup(companion_id)
        except Exception:
            return None
        if not recorded:
            return None
        return recorded.get("lora_provider")

    @staticmethod
    def _coerce_config(raw_config: object) -> dict:
        """Coerce a possibly-JSON-string ``avatar_config`` into a dict."""
        if raw_config is None:
            return {}
        if isinstance(raw_config, dict):
            return raw_config
        if isinstance(raw_config, str):
            import json

            try:
                parsed = json.loads(raw_config)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}
