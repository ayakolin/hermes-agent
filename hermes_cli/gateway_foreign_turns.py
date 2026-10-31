"""Notification turns the classic CLI missed the start of, painted from their receipts.

``gateway_tools_notify.queue_admitted_turn`` paints a gateway-admitted notification turn when its
``message.start`` is still in the stream. A start that a local turn's drain or ``_drain_stale`` dropped is
remembered with its card instead (``SessionState.missed_cards``) and recovered here on an idle tick: one turn
at a time, like ``recover_queued``. The owner already ran it, so the CLI never submits a prompt for it — it
paints main's ``◈ Background Process Finished: ...`` header and completes the reply from the receipt
(``GatewayAgent``'s adopted turn, the same ``recovered`` handoff ``/queue`` retries use).
"""
from __future__ import annotations

_NEVER_RAN = {"cancelled", "rejected"}


def recover_automation(cli, store) -> None:
    """Idle tick, after the stream had no start for the remembered cards: paint the first one it can.

    ``admitted_turn`` carries what ``GatewayAgent.run_conversation`` reads off an adopted turn (its admission,
    text, generation, ``recovered`` and the receipt when it already settled), so the recovered turn streams
    nothing live and completes from the receipt."""
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_tools_notify import _card_row, _put_first, _recovered_turn_pending
    state = store.connection.state(getattr(cli, "session_id", None))
    if not state.missed_cards or _recovered_turn_pending(cli, state):
        return  # one recovered turn at a time; the next is recovered after this one ran
    for admission, card in list(state.missed_cards.items()):
        try:
            receipt = store.call("prompt.receipt", session_id=cli.session_id, admission_id=admission,
                                 include_result=True) or {}
        except GatewayClientError:
            return  # unread: try again on the next tick
        status = receipt.get("status")
        if status == "queued":
            continue  # not admitted yet: its start is still to come
        if status == "terminal" and receipt.get("outcome") in _NEVER_RAN:
            state.missed_cards.pop(admission, None)
            continue
        if status not in ("started", "terminal"):
            continue  # unknown: the owner resolves it; until then the card waits
        state.missed_cards.pop(admission, None)
        adopted = {"admission_id": admission, "text": card["text"],
                   "execution_generation": receipt.get("execution_generation", card.get("execution_generation")),
                   "recovered": True}
        if status == "terminal":
            adopted["receipt"] = receipt
        state.admitted_turn = adopted
        _put_first(cli._pending_input, _card_row(admission, card))
        return  # one turn at a time
