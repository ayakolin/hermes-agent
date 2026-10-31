"""Default configuration data for the delegation and dispatcher sections of config.yaml.

Pure-data leaf module split out of config_defaults.py (which may only shrink). Comments are the
user-facing docs of config.yaml.
"""

DELEGATION_DEFAULTS = {
    "backend": "default",
    "backends": {},
    "model": "",  # e.g. "google/gemini-3-flash-preview" (empty = inherit parent)
    "provider": "",  # e.g. "openrouter" (empty = inherit parent provider + credentials)
    # Fallback chain for delegated children (same entry format as the top-level list).
    # For an unpinned child, null = inherit the parent chain; [] = disable fallback.
    # A child pinned by provider, endpoint, or model gets no fallback unless this
    # setting declares one explicitly.
    "fallback_providers": None,
    "base_url": "",  # direct OpenAI-compatible endpoint for subagents
    "api_key": "",  # key for delegation.base_url (falls back to OPENAI_API_KEY)
    # Wire protocol for delegation.base_url: "chat_completions" | "codex_responses" |
    # "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix); set
    # explicitly for non-standard endpoints.
    "api_mode": "",
    # Per-child request settings on every delegation call (all resolution branches). Top-level
    # keys = API kwargs (e.g. service_tier); "extra_body" sub-dict merges into extra_body, e.g.
    # {"extra_body": {"provider": {"sort": "throughput"}}}. Explicit values win OVER
    # runtime/parent overrides (extra_body deep-merged 1 level).
    "request_overrides": {},
    # compression_threshold_tokens: optional absolute cap on a subagent's compaction TRIGGER
    # (not the request payload), applied as the lower of this and the child's ratio threshold.
    # 0 (default) = no subagent-specific cap; children compact where the parent does — the lower
    # of 0.50 x window and the global compression.threshold_tokens cap. A replay of a 1,393-agent
    # run showed 200K-400K caps within 5% of each other in cost once cache prefixes are intact,
    # and every compaction is a chance to lose detail, so the default stays off. A token count >= 16000 enables it;
    # other values (true, "200k") are config errors: warned and ignored.
    "compression_threshold_tokens": 0,
    # When delegate_task narrows child toolsets, keep the parent's enabled MCP toolsets (so
    # toolsets=["web"] doesn't strip MCP). false = strict intersection.
    "inherit_mcp_toolsets": True,
    # Per-subagent iteration cap (own budget, independent of the parent's).
    "max_iterations": 250,
    # Hard per-summary char ceiling on subagent results, layered on the dynamic budget (each
    # summary is sized to the parent's remaining context headroom; trimmed text spills to
    # ~/.hermes/cache/delegation/ with a head+tail window + read_file offset footer, nothing
    # lost). 0 disables the ceiling; the dynamic budget still applies.
    "max_summary_chars": 24000,
    # Inactivity cap per child (seconds, floor 30) — time with NO progress, not total runtime. 0 = no cap:
    # children fail only from real errors (API, tools, iteration budget). A progressing child (including one
    # waiting on a multi-minute completion) restarts the window; a frozen one is caught.
    "child_timeout_seconds": 0,
    # Subagent effort: "ultra" | "max" | "xhigh" | "high" | "medium" | "low" | "minimal" | "none" (empty = inherit)
    "reasoning_effort": "",
    # Max parallel children per batch AND max concurrent background delegation units; async
    # dispatches beyond it run synchronously. Floor 1, no ceiling.
    "max_concurrent_children": 10,
    # Background fan-outs return as ONE message when the whole call finishes. true = each task
    # (or `group`) returns on its own as it finishes — more new turns for the orchestrator.
    "independent_completions": False,
    "gateway_mode": "auto",  # gateway: inline (join in the turn) | background | auto = inline where interim msgs are off
    # Orchestrator role controls. Depth floored at 1, no ceiling; each level multiplies cost.
    "max_spawn_depth": 1,  # 1 = flat, 2 = orchestrator→leaf, 3+ = deeper
    "orchestrator_enabled": True,  # kill switch for role="orchestrator"
    # Total subagents a finite one-shot run (hermes chat -q / --oneshot) may spawn; 0 = unlimited.
    # Each child re-pays a cold system prompt and re-explores the repo, and one-shot spawns are mostly
    # "review my own work" rather than parallel work (agent/oneshot_footprint.py).
    "oneshot_max_children": 2,
    # Subagent threads ALWAYS resolve approvals non-interactively (the parent TUI owns stdin;
    # input() from a worker would deadlock). false = auto-deny, true = auto-approve "once"; both
    # log a warning audit line. true only for trusted batch work. A dispatcher session's executor with no
    # approver (a chat platform's card lives for one turn) is refused as "no_approver" and asks via needs_input;
    # true would make every such executor run dangerous commands unasked, so leave it false there.
    "subagent_auto_approve": False,
    # Subagent background processes (task_id "sa-...") route notify_on_complete / watch_pattern
    # notifications to the PARENT; false suppresses them (the child's result is the
    # deliverable). Async-delegation results are NEVER suppressed.
    "surface_child_process_notifications": False,
    # Dispatcher sessions only (agent/dispatcher_session.py). Pool-full tasks wait in a per-session queue of
    # this many slots instead of running inline; a full queue is a tool error, never a synchronous run.
    "queue_limit": 32,
    # Seconds the gateway waits after a background completion before draining, so completions that land
    # together become one turn. 0 disables batching.
    "completion_batch_window_seconds": 0.3,
    # Dispatcher sessions: after a turn whose tool calls were all successful dispatches (and that already
    # showed the user text), end the turn without another model call. false restores the status-line call.
    "end_turn_on_dispatch": True,
    # Executor baseline for dispatcher sessions: trusted config, never model-chosen. Only a dispatcher
    # session's delegate_task children use it; other sessions keep "a child never exceeds its parent".
    # mcp: "all" (MCP servers enabled in the profile) | "none" | [server names].
    "executor": {
        "toolsets": [
            "terminal", "file", "web", "browser", "code_execution",
            "vision", "image_gen", "skills", "todo", "session_search",
        ],
        "mcp": "all",
    },
}


# Dispatcher mode: the main agent answers at once and hands long work to background executor subagents
# (delegation.executor). Decided once at session creation; never switched mid-conversation (prompt cache).
DISPATCHER_DEFAULTS = {
    "enabled": True,
    # Extra toolsets appended to the dispatcher posture toolset, e.g. ["session_search"].
    "extra_toolsets": [],
    "ack": {
        "reminder": True,       # one-shot "the user has seen no text yet" note on the first tool result
        "max_nudges": 1,        # end-of-turn nudges when a human turn ended with no visible text; 0 = off
        "force_interim": True,  # deliver pre-tool-call text immediately on every platform
        "early_interim": True,  # messaging platforms: flush the preamble when the first tool call starts
    },
    "latency": {
        "memory_budget_s": 2.0,         # external memory tick + prefetch budget before the first token; 0 = legacy
        "pre_llm_call_budget_s": 3.0,   # total budget for pre_llm_call plugin callbacks; 0 = per-callback only
        "skills_recheck_s": 30,         # min seconds between full skills-index comparisons; 0 = every turn
        "defer_compaction": True,       # defer compaction to a background-prepared summary below the ceiling
        "defer_ceiling_ratio": 0.85,
        "stream_deltas": True,          # platforms that opt in to streaming stream for dispatcher sessions
    },
    "busy": {
        "policy": "dispatch",   # dispatch: redirect -> steer -> queue, never a hard interrupt | inherit
        "quick_lane": True,     # answer from a cache-parity fork while redirect/steer cannot land yet
        "stall_seconds": 4.0,   # seconds an accepted redirect/steer may stay unconsumed before the quick lane
    },
}
