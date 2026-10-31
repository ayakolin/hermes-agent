# Dispatcher mode: where the decision lives

Dispatcher mode is decided once, where a session is created, and frozen. What follows is the wiring per
surface; the concept and the configuration are in
[Dispatcher mode](../../user-guide/features/dispatcher-mode.md).

Everything reads the frozen answer through `agent/dispatcher_session.py`:

| Name | Purpose |
|---|---|
| `dispatcher_applies(config, *, surface, push_capable, safe_mode, explicit_toolsets, disabled)` | the creation-time predicate |
| `dispatcher_toolsets(config)` | the posture toolset plus validated `dispatcher.extra_toolsets` |
| `freeze_dispatcher_session(agent, enabled_toolsets)` | sets `agent._dispatcher_session`, once |
| `agent_is_dispatcher(agent)` | the frozen flag; the only reader of `delegate_task`'s variant and the prompt |
| `session_dispatcher_toolsets(db, session_id, decide)` | the session-row ladder for surfaces without a policy |
| `toolsets_pinned(config, platform)` | a saved `platform_toolsets.<platform>` |
| `session_marker(agent)` / `persist_session_marker(agent)` | the `sessions.model_config['dispatcher']` marker |

## Local sessions (classic CLI, Ink TUI, Desktop)

`gateway/session_local.py::stage_local_session` is the one caller that passes `dispatcher_session=True` to
`gateway/session_policy.py::build_policy`. `gateway/session_policy_dispatcher.py::local_dispatcher` answers
the predicate (a managed worker executes out of process and is excluded), and the result is frozen as
`LocalSessionPolicy.dispatcher` together with the toolsets it produced. Resume migration, cron, hosted rooms
and bypass launches build a policy without the opt-in.

- `current_toolsets` returns `policy.toolsets` for a dispatcher and never re-resolves the profile, so a
  config edit mid-conversation cannot change `enabled_toolsets` (and the agent-cache signature).
- `session_policy_refresh` still follows `disabled_toolsets`, `platform_toolsets` and the MCP server
  definitions (side agents read the first two), but its own toolset tuple stays as created.
- `classic_toolsets(config)` is the surface's selection without dispatcher mode: `/bg`, `/refine` and the
  cold review run it, because they do the work a dispatcher hands off.
- `session_policy_credentials.policy_identity` hashes the field only when set, so an older gateway's
  reference still matches.
- `gateway/session_onboarding.py` opts the onboarding chat out (`dispatcher_session=False`).

## Chat platforms (Telegram, Discord, Slack, ...)

`gateway/run_turn_runner.py::_run_sync_scoped` asks `gateway/run_turn_dispatcher.py::dispatcher_turn_toolsets`
in the `elif policy is None` branch (a hosted room's override wins and stays), before `_resolve_turn_agent`:
the agent-cache signature includes `enabled_toolsets`, so the turn's selection must be stable.

The ladder, in order:

1. The cached agent of this session id — its `_dispatcher_session` and toolsets. An entry cached for another
   session id is another conversation and is not read.
2. The `dispatcher` marker on the conversation's compression tip; its stored toolsets are the answer.
3. A tip that already built a prompt (or pinned its tools[]) without a marker is classic.
4. Otherwise the live config decides once, and the agent records the marker when it builds.

A saved `platform_toolsets.<platform>` or an adapter's `toolsets_for_source()` (webhook routes) pins the
selection, and a pinned session stays classic.

## Legacy tui_gateway (`tui_gateway/agent_factory.py`)

`_make_agent` builds through `_session_enabled_toolsets(platform, db, session_id)`: the dispatcher selection
plus the client surface's toolsets (`project` / `desktop_ui` / `catalog`), else `_load_enabled_toolsets`.
`_dispatcher_selection` answers the same ladder and refuses a plugin session whose source is not `tui` or
`desktop`, a pinned `HERMES_TUI_TOOLSETS` and an onboarding chat. The agentless read-backs
(`methods_tools._session_toolsets`, `methods_connectors`) go through the same helper, so
`tools.list` / `tools.show` report the surface the agent will carry.

## The rest of the runtime

- `agent/agent_init.py::_load_tools` freezes the flag before the first tool build (it picks `delegate_task`'s
  variant through `tools/delegate_tool_config.py::agent_joins_delegation`, which returns False for a
  dispatcher whatever `delegation.gateway_mode` says), and `_init_session_state` seeds the marker only for a
  dispatcher agent.
- `tools/mcp_tool_agent.py::_resolve_refresh_toolsets` is the single refresh choke point: for a dispatcher it
  ignores the classic `enabled_override` every rebuild path passes, and `persist_agent_tool_names` re-writes
  the marker next to the tools[] pin.
- `agent/background_review.py::build_cache_parity_fork` copies the flag onto a same-model fork (it replays the
  parent's cached prompt) and clears it on a routed fork (which builds its own prompt for a review job).
