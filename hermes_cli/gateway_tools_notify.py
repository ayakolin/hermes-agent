"""Background-process notification turns the gateway admits for this CLI's session (W11).

On main the classic CLI drained ``tools.process_registry`` itself, queued a
``ProcessNotificationBatch`` into ``_pending_input`` and ran the ``[IMPORTANT: Background process
...]`` turn. Over the session authority the process runs in the gateway: the gateway drains the
completion, admits the notification turn (``authority.admit_automation``) and runs it. The CLI
neither drains the registry nor claims delivery (IPS-086) — it only watches its session's stream.

The claim's ``message.start`` carries the admission's async-result card (``display_kind`` /
``display_text`` / model ``text``; ``gateway.session_automation.admission_display``). The idle tick
turns it into main's ``TimelineNotification``, so main's input path paints the same ``◈ Background
Process Finished: ...`` header, and ``GatewayAgent.run_conversation`` pumps that already-admitted
turn (no ``prompt.submit``) through the same callbacks as a typed turn.
"""
from __future__ import annotations

import queue


def gateway_store(cli):
    """The CLI's gateway session store, or None (a test double / an unavailable store)."""
    from hermes_cli.gateway_session_store import GatewaySessionStore
    store = getattr(cli, "_session_db", None)
    return store if isinstance(store, GatewaySessionStore) else None


def _card(params):
    payload = params.get("payload")
    if not isinstance(payload, dict):
        return None
    kind, display, text = payload.get("display_kind"), payload.get("display_text"), payload.get("text")
    if not kind or not isinstance(display, str) or not display.strip() or not isinstance(text, str) or not text:
        return None
    return payload


def queue_admitted_turn(cli, store) -> None:
    """Idle tick: find a gateway-admitted notification turn's ``message.start`` for this session and
    queue main's ``TimelineNotification`` for it. Frames before it are the between-turn leftovers
    ``run_conversation`` would discard anyway; frames after it stay queued for the pump.

    When the stream holds no start, one this viewer dropped earlier is painted from its receipt
    (``recover_automation``): its turn already ran in the owner, so nothing else would show it."""
    from hermes_cli.gateway_foreign_turns import recover_automation
    from tools.process_registry_notifications import TimelineNotification
    connection, session_id = store.connection, getattr(cli, "session_id", None)
    state = connection.state(session_id)
    if _adoption_waiting(cli, state):
        return  # an adopted /queue turn is waiting for the input loop: its frames stay queued for its pump
    if not _recovered_turn_pending(cli, state):
        state.admitted_turn = None  # a header queued earlier whose turn never ran is stale now
    events = connection.events
    while True:
        try:
            frame = events.get_nowait()
        except queue.Empty:
            recover_queued(cli, store)
            recover_automation(cli, store)
            return
        if isinstance(frame, Exception):
            from hermes_cli.gateway_connection import lost_before
            if lost_before(frame, getattr(connection, "epoch", None)):
                continue  # a connection a reconnect already replaced
            events.put(frame)  # the disconnect belongs to the next turn
            return
        params = frame.get("params") if isinstance(frame, dict) else None
        if not isinstance(params, dict) or params.get("session_id") != session_id:
            continue
        if params.get("type") == "review.summary":
            # The background review (/refine, the post-turn review) reports after its turn settled.
            from hermes_cli.gateway_agent import review_summary_line
            line = review_summary_line(params.get("payload"))
            if line:
                cli._agent_status_print(line)
            continue
        if params.get("type") != "message.start" or not params.get("admission_id"):
            continue
        queued_text = state.queued_inputs.get(params["admission_id"])
        if queued_text is not None:
            # A ``/queue`` prompt this viewer admitted (#134889 N5): the owner runs it now, and main's
            # input path paints and pumps it as a typed turn. Literal: it is a prompt, never a command.
            # Adopted by admission id (the item carries it), never by text: the input path may rewrite
            # the text, and a second ``prompt.submit`` would run the prompt twice.
            from cli import _SeededQueryMessage
            _adopt(cli, state, {"admission_id": params["admission_id"],
                                "execution_generation": params.get("execution_generation")}, queued_text)
            return
        card = _card(params)
        if card is None:
            continue
        state.admitted_turn = {"admission_id": params["admission_id"], "text": card["text"],
                               "execution_generation": params.get("execution_generation")}
        state.missed_cards.pop(params["admission_id"], None)  # painted now, nothing to recover
        _put_first(cli._pending_input, _card_row(params["admission_id"], card))
        return


def _card_row(admission_id, card):
    """The ``◈`` row main's input path paints for an admitted notification turn, tagged with its admission so
    an idle tick can tell its own recovered row from a stale header."""
    from tools.process_registry_notifications import TimelineNotification
    row = TimelineNotification(card["text"], card["display_text"], card["display_kind"],
                               card.get("notification_category") or "result")
    row.admission_id = admission_id
    return row


def _adopt(cli, state, adopted, text):
    from cli import _SeededQueryMessage
    state.missed.discard(adopted["admission_id"])
    state.adopted[adopted["admission_id"]] = adopted
    _put_first(cli._pending_input, _SeededQueryMessage(text, admission_id=adopted["admission_id"]))


def _adoption_waiting(cli, state) -> bool:
    """An adopted item is still in ``_pending_input`` (one ``/queue clear`` dropped is forgotten)."""
    if not state.adopted:
        return False
    pending = getattr(cli, "_pending_input", None)
    if pending is None:
        return False
    with pending.mutex:
        waiting = {getattr(item, "admission_id", None) for item in pending.queue}
    waiting.add(state.armed_admission)  # taken by the input loop, its run_conversation not started yet
    for admission in [a for a in state.adopted if a not in waiting]:
        state.adopted.pop(admission, None)
    return bool(state.adopted)


def _recovered_turn_pending(cli, state) -> bool:
    """A notification turn recovered from its receipt is still waiting for the input loop: its
    ``admitted_turn`` must survive the next tick's reset (which only clears a stale header)."""
    adopted = getattr(state, "admitted_turn", None)
    pending = getattr(cli, "_pending_input", None)
    if not adopted or not adopted.get("recovered") or pending is None:
        return False
    with pending.mutex:
        return any(getattr(item, "admission_id", None) == adopted.get("admission_id") for item in pending.queue)


_NEVER_RAN = {"cancelled", "rejected"}


def recover_queued(cli, store) -> None:
    """Idle tick, after the stream had no ``message.start`` for them: this viewer's ``/queue``
    admissions the owner started or settled without this viewer painting the turn (a local turn's
    drain or pump dropped the start, or it ran across a reconnect, which replays no frames). Read on
    a new published pending list or a dropped start only, never every tick.

    Still running: adopted and pumped live, its reply completed from the receipt (``recovered``).
    Settled: adopted with its saved result, painted without a pump. One that never ran (cancelled,
    rejected) is forgotten: the only point where ``queued_inputs`` drops a row unpainted.
    """
    from hermes_cli.gateway_client import GatewayClientError
    state = store.connection.state(getattr(cli, "session_id", None))
    if not state.queued_inputs or (state.recovered_seq == state.pending_seq and not state.missed):
        return
    state.recovered_seq = state.pending_seq
    listed = {row.get("admission_id"): row.get("status") for row in state.pending or () if isinstance(row, dict)}
    for admission, text in list(state.queued_inputs.items()):
        if admission in state.adopted or (admission not in state.missed and listed.get(admission) == "queued"):
            continue
        try:
            receipt = store.call("prompt.receipt", session_id=cli.session_id, admission_id=admission,
                                 include_result=True) or {}
        except GatewayClientError:
            state.recovered_seq = -1  # unread: try again on the next tick
            continue
        status = receipt.get("status")
        if status == "queued":
            state.missed.discard(admission)
            continue
        if status == "terminal" and receipt.get("outcome") in _NEVER_RAN:
            state.queued_inputs.pop(admission, None)
            state.missed.discard(admission)
            continue
        if status not in ("started", "terminal"):
            # unknown: the owner resolves it (``/discard`` or recovery settles it terminal, and the next
            # read forgets or paints it). Until then it stays here and is re-read once per new pending
            # list, never every tick; the row is rare and the read is one receipt.
            continue
        adopted = {"admission_id": admission, "execution_generation": receipt.get("execution_generation"),
                   "recovered": True}
        if status == "terminal":
            adopted["receipt"] = receipt
        _adopt(cli, state, adopted, text)
        return  # one turn at a time: the next one is recovered after this one ran


def missed_start(state, params) -> None:
    """A frame dropped outside its turn: a ``/queue`` admission's ``message.start`` is remembered so
    the idle tick paints that turn from its receipt instead of losing it. An admitted notification turn's
    start is remembered with its card for ``recover_automation``: the user would otherwise see only the
    turn after it, never the "background work finished" line driving it."""
    if params.get("type") != "message.start" or not params.get("admission_id"):
        return
    if params["admission_id"] in state.queued_inputs:
        state.missed.add(params["admission_id"])
        return
    card = _card(params)
    if card is None:
        return
    store_missed_card(state, params["admission_id"], {**card, "execution_generation": params.get("execution_generation")})


_MAX_MISSED_CARDS = 20


def store_missed_card(state, admission_id, card):
    """Remember one dropped admission's card, oldest evicted first (a start whose admission can never be read
    must not grow the map without bound)."""
    state.missed_cards[admission_id] = card
    while len(state.missed_cards) > _MAX_MISSED_CARDS:
        state.missed_cards.pop(next(iter(state.missed_cards)))


def _put_first(pending, item):
    """Ahead of anything already queued: the gateway is running this turn NOW (typed inputs the
    post-turn path re-queued are admitted behind it), and a typed turn run first would drop it."""
    with pending.mutex:
        pending.queue.appendleft(item)
        pending.unfinished_tasks += 1
        pending.not_empty.notify()


def arm_adopted(cli, item) -> None:
    """The input loop took ``item``: when it is an adopted ``/queue`` admission, the next
    ``run_conversation`` pumps exactly that admission (``take_admitted_turn``)."""
    admission = getattr(item, "admission_id", None)
    store = gateway_store(cli) if admission else None
    if store is not None:
        store.connection.state(getattr(cli, "session_id", None)).armed_admission = admission


def take_admitted_turn(agent, text):
    """The admitted turn ``run_conversation(text)`` replays instead of submitting; None (and the
    stale notification adoption dropped) for any other turn. A ``/queue`` admission is matched by the
    id ``arm_adopted`` recorded, a notification turn by its model text."""
    state = agent._state()
    armed, state.armed_admission = getattr(state, "armed_admission", None), None
    adopted = state.adopted.pop(armed, None) if armed else None
    if adopted is not None:
        state.queued_inputs.pop(armed, None)  # painted now: the admission's turn is this one
        return adopted
    adopted = getattr(state, "admitted_turn", None)
    state.admitted_turn = None
    if not adopted or adopted.get("text") != text:
        return None
    return adopted
