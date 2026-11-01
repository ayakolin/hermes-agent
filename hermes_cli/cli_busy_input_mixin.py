"""The classic CLI's busy-input router: what Enter, Tab, the external editor and /steer do while a turn runs.

Every entry goes through one ordered path (``_busy_submit``). A dispatcher session's text goes through a
serial worker so the ladder's RPCs never run on the prompt_toolkit key thread (the gateway answers a
socket's requests one at a time, so a slow correction would freeze Enter), and the worker keeps submission
order: text submitted while it is busy joins its FIFO, and the worker decides course from the state captured
at SUBMIT time, not when it gets to run. A dispatcher session never writes ``_interrupt_queue``: the ladder
redirects, steers, or queues, and a refusal the owner marks undelivered is queued as the next prompt.
"""
from __future__ import annotations

import logging
import queue
import time

from agent.i18n import t

logger = logging.getLogger(__name__)


def _preview(text):
    text = text or ""
    return text[:80] + ("..." if len(text) > 80 else "")


class CLIBusyInputMixin:
    def _mirror_server_busy(self):
        """Adopt the owner's busy mode for the open session (``config.get busy``): a dispatcher session runs
        the redirect-first ladder whatever the CLI's own startup config says, and its route decides whether
        the ordered worker handles the corrections. Off the gateway (a test double, no session) this is a
        no-op and the CLI keeps its local mode."""
        store = self._gateway_queue_store()
        if store is None:
            return
        from hermes_cli.gateway_client import GatewayClientError
        try:
            reply = store.call("config.get", session_id=self.session_id, key="busy") or {}
        except GatewayClientError as exc:
            logger.debug("config.get busy failed: %s", exc)
            return
        self._adopt_server_busy(reply)

    def _adopt_server_busy(self, reply):
        from agent.busy_input_local import dispatch_policy_applies
        if reply.get("value") in ("interrupt", "steer", "queue"):
            self.busy_input_mode = reply["value"]
        self._busy_dispatch = dispatch_policy_applies(None, reply.get("dispatch") is True)

    def _busy_submit(self, text, images, payload, mode=None):
        """Route one submission typed while the agent runs (the entry every busy path shares)."""
        mode = mode or self.busy_input_mode
        if not self._busy_dispatches():
            return self._busy_route(text, images, payload, mode, running=True)
        # A dispatcher session: the owner resolves the ladder verb by verb, so the RPCs run off the key
        # thread, in one FIFO. "Busy at submit time" is captured here, so an item the worker reaches after
        # the turn ended still queues in submission order instead of jumping the local FIFO.
        self._busy_worker_submit({"text": text, "images": images, "payload": payload, "mode": mode,
                                  "running": getattr(self, "_agent_running", False) is True})
        return None

    def _busy_dispatches(self):
        """A dispatcher session whose corrections the owner resolves through the ladder (set by
        ``_mirror_server_busy``; a partially built CLI is classic)."""
        return getattr(self, "_busy_dispatch", False) is True

    def _busy_route(self, text, images, payload, mode, *, running):
        """Apply the mode to one busy submission: the user's steer mode, else redirect, else the ladder's steer
        rung (a dispatcher session), else the interrupt queue (classic ``interrupt``) or the next turn.

        ``running`` is what ``_agent_running`` was when the user SUBMITTED (never read later: a worker item
        that finds the turn over must still queue where its position says, not jump the local FIFO)."""
        from hermes_cli.gateway_agent import STEERED, VOICE_STOPPED, Unknown
        steer = self._busy_steer(text, images, mode)
        if steer is True:
            return "consumed"  # the user's steer mode took it (or consumed a voice stop)
        if steer is False or mode == "steer":
            mode = "queue"  # the steer mode refused it, or a steer's text cannot ride along: the next turn
        if mode == "queue" or images or not text:
            return self._busy_queue(text, images, payload, running=running)
        from agent.busy_input_local import supports_redirect
        agent = getattr(self, "agent", None)  # a CLI still building its agent has none: classic fallback below
        if supports_redirect(agent):
            answer = self._busy_call(agent.redirect, text)
            if answer is True or answer == STEERED:
                self._busy_print(f"  {t('cli.tui.redirected_turn', preview=_preview(text))}")
                return "redirect"
            if answer == VOICE_STOPPED:
                return "consumed"  # TUI-26: the owner ended the voice chat; nothing to queue
            if isinstance(answer, Unknown):
                # Sent, outcome unknown (a Codex steer may already hold it): neither re-queued nor interrupted.
                self._busy_print(f"  {t('cli.tui.redirect_unknown', reason=answer.reason)}")
                return "unknown"
        if not self._busy_dispatches():
            self._interrupt_queue.put(payload)
            self._busy_log_interrupt(payload)
            return "interrupt"
        # The ladder's steer rung: a dispatcher's declined redirect (no model request in flight, or none
        # supported) falls to steer, which a running turn accepts, before anything waits.
        if self._busy_steer(text, images, "steer") is True:
            return "consumed"
        # Still not taken: queue it, never a hard interrupt (its pending steers would be dropped, #86134).
        return self._busy_queue(text, images, payload, running=running)

    def _busy_steer(self, text, images, mode):
        """The steer MODE, when the user chose it (``/busy steer``, Tab-queue's inverse): ``True`` taken (or a
        voice stop, consumed), ``False`` refused, None when this submission is not a steer. ``interrupt`` mode
        is redirect-first, even for a dispatcher session; the ladder's own steer rung runs as that redirect's
        fallback below."""
        agent = getattr(self, "agent", None)
        if mode != "steer" or images or not text or not hasattr(agent, "steer"):
            return None
        answer = self._busy_call(agent.steer, text)
        from hermes_cli.gateway_agent import VOICE_STOPPED
        if answer == VOICE_STOPPED:
            return True  # TUI-26: consumed as a voice stop; nothing to queue either
        if answer:
            self._busy_print(f"  {t('cli.tui.steered', preview=_preview(text))}")
            return True
        return False

    def _busy_call(self, method, text):
        from hermes_cli.gateway_agent import Refusal
        try:
            return method(text)
        except Exception as exc:  # health: allow BLE001 -- a failed steer is a refusal the caller queues
            self._busy_print(f"  {t('cli.tui.steer_failed_queued', error=exc)}")
            return Refusal("error", undelivered=False)

    def _busy_queue(self, text, images, payload, *, running):
        """The next-turn rung: the gateway's durable FIFO for a text-only prompt while a turn runs, else the
        local ``_pending_input`` (main's own queue)."""
        if text and not images and running and self._gateway_queue_store() is not None:
            self._queue_enqueue(text, running=True)
            return "queue"
        self._pending_input.put(payload)
        preview = text if text else t("cli.tui.images_attached_preview", len(images))
        self._busy_print(f"  {t('cli.tui.queued_next_turn', preview=_preview(preview))}")
        return "queue"

    def _busy_print(self, line):
        from cli import _cprint
        _cprint(line)

    def _busy_log_interrupt(self, payload):
        from cli import _hermes_home
        try:
            with open(_hermes_home / "interrupt_debug.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} ENTER: queued interrupt msg={str(payload)[:60]!r}, "
                        f"agent_running={self._agent_running}\n")
        except OSError as exc:  # the debug trail is best effort: a read-only home must not break Enter
            logger.debug("interrupt debug log unavailable: %s", exc)

    # -- the serial worker ---------------------------------------------------------------------
    def _busy_worker_submit(self, item):
        """Append to the worker's FIFO (never blocks the key thread) and make sure it runs."""
        if getattr(self, "_busy_worker", None) is None:
            from agent.memory_provider import spawn_context_thread
            self._busy_worker = queue.Queue()
            # spawn_context_thread, not threading.Thread: the worker delivers prompts for this CLI's
            # profile, and ContextVars (profile scope, approval callbacks) do not cross a bare thread.
            self._busy_worker_thread = spawn_context_thread(
                self._busy_worker_loop, name="hermes-cli-busy", args=(self._busy_worker,))
            self._busy_worker_thread.start()
        self._busy_worker.put(item)

    def _busy_worker_loop(self, work):
        while True:
            item = work.get()
            try:
                self._busy_route(item["text"], item["images"], item["payload"], item["mode"],
                                 running=item["running"])
            except Exception as exc:  # health: allow BLE001 -- one failed route must not kill the worker
                self._busy_print(f"  busy input failed: {exc}")
            finally:
                work.task_done()

    def _busy_worker_idle(self):
        """The worker has nothing left to deliver (tests and shutdown wait on this)."""
        worker = getattr(self, "_busy_worker", None)
        if worker is None:
            return True
        with worker.mutex:
            return not worker.queue and worker.unfinished_tasks == 0
