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

Poll-only: this provider does NOT implement ``active_handles`` — auto-wake of
unattended jobs would need a durable in-flight job registry (future work). Both
a blocking ``wait("lora_train:<companion>:<job>")`` and a ``mode="signal"``
watch work today via the host engine; ``signal`` is ``None`` because there is
no terminal-state signal emitter for this kind yet.
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
        output_path = getattr(status, "output_path", None) or (
            (status.provider_details or {}).get("output_path")
            if getattr(status, "provider_details", None)
            else None
        )

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
                trigger_word=None,
                output_path=output_path,
                error=error,
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
