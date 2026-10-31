"""The console banner ``agent_init._load_tools`` prints once its tool snapshot is built.

Split out of the ``agent_init`` facade (which may only shrink): the banners are stdout formatting with no
bearing on the snapshot, so they live here and the load path stays a sequence of decisions.
"""
from __future__ import annotations


def announce_tool_snapshot(agent, enabled_toolsets, disabled_toolsets) -> None:
    """Print what was loaded (or why nothing was); a quiet agent prints nothing at all."""
    if agent.quiet_mode:
        return
    if agent.tools:
        print(f"🛠️  Loaded {len(agent.tools)} tools: {', '.join(sorted(agent.valid_tool_names))}")
        if enabled_toolsets:
            print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
        if disabled_toolsets:
            print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        import model_tools
        requirements = model_tools.check_toolset_requirements()
        missing_reqs = [name for name, available in requirements.items() if not available]
        if missing_reqs:
            agent._safe_print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}", diagnostic=True)
    else:
        print("🛠️  No tools loaded (all tools filtered out or unavailable)")
    if agent.save_trajectories:
        print("📝 Trajectory saving enabled")
    if agent.ephemeral_system_prompt:
        prompt_preview = agent.ephemeral_system_prompt[:60] + "..." if len(agent.ephemeral_system_prompt) > 60 else agent.ephemeral_system_prompt
        print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
    if agent._use_prompt_caching:
        if agent._use_native_cache_layout and agent.provider == "anthropic":
            source = "native Anthropic"
        elif agent._use_native_cache_layout:
            source = "Anthropic-compatible endpoint"
        else:
            source = "Claude via OpenRouter"
        print(f"💾 Prompt caching: ENABLED ({source}, {agent._cache_ttl} TTL)")
