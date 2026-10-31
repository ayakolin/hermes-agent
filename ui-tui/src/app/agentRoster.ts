import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useMemo } from 'react'

import type { SubagentListResponse } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'
import type { GatewayClient } from '../gatewayClient.js'

import { asRpcResult } from '../lib/rpc.js'
import { turnController } from './turnController.js'
import { useTurnSelector } from './turnStore.js'
import { $uiState, getUiState } from './uiStore.js'

// Session-local presentation only; never persisted to config.
export const $agentDockCollapsed = atom(false)

const EMPTY: SubagentListResponse = { subagents: [], delegations: [] }
export const $agentSnapshot = atom<{ sid: string | null; data: SubagentListResponse }>({ sid: null, data: EMPTY })

export function applyAgentSnapshot(sid: string | null, data: SubagentListResponse = EMPTY) {
  const previous = $agentSnapshot.get()

  if (previous.sid !== sid || JSON.stringify(previous.data) !== JSON.stringify(data)) {
    $agentSnapshot.set({ sid, data })
  }
}

/** One roster snapshot reached the app: merge it into the view and reconcile the turn's retained rows
 *  against it (a running row the owner no longer lists is gone — its completion was lost). */
export function applyRosterSnapshot(sid: string | null, data: SubagentListResponse = EMPTY) {
  applyAgentSnapshot(sid, data)
  turnController.reconcileSubagents(new Set((data.subagents ?? []).map(row => row.subagent_id)))
}

const ROSTER_PUSH_DEBOUNCE_MS = 250
let rosterPushTimer: null | ReturnType<typeof setTimeout> = null

/** Re-read `subagent.list` now (coalesced): a pushed `subagent.*` event or a moved active count means the
 *  roster changed, and a spawn between two 1.5 s polls would otherwise stay invisible until the next one.
 *  The poll stays as the fallback (a dropped frame, a gateway restart). */
export function refreshAgentRoster(gw: GatewayClient, sid: null | string) {
  if (!sid) {
    return
  }

  if (rosterPushTimer !== null) {
    clearTimeout(rosterPushTimer)
  }

  rosterPushTimer = setTimeout(() => {
    rosterPushTimer = null

    if (getUiState().sid !== sid) {
      return // the user moved on: another session's roster is not this one
    }

    gw.request<SubagentListResponse>('subagent.list', { session_id: sid })
      .then(raw => {
        const result = asRpcResult<SubagentListResponse>(raw)

        if (result && getUiState().sid === sid) {
          applyAgentSnapshot(sid, result)
        }
      })
      .catch(() => {})
  }, ROSTER_PUSH_DEBOUNCE_MS)
}

export function mergeAgentRoster(events: SubagentProgress[], data: SubagentListResponse): SubagentProgress[] {
  const merged = new Map(events.map(s => [s.id, s]))

  for (const [index, s] of data.subagents.entries()) {
    const previous = merged.get(s.subagent_id)
    merged.set(s.subagent_id, {
      depth: s.depth ?? 0,
      index,
      parentId: s.parent_id ?? null,
      notes: [],
      thinking: [],
      taskCount: 1,
      ...previous,
      id: s.subagent_id,
      goal: s.goal || previous?.goal || 'Starting agent',
      delegationId: s.delegation_id ?? previous?.delegationId,
      model: s.model ?? previous?.model,
      startedAt: s.started_at != null ? s.started_at * 1000 : previous?.startedAt,
      // Snapshot replies may predate progress/completion events already rendered.
      status: previous && previous.status !== 'queued' ? previous.status : s.status === 'queued' ? 'queued' : 'running',
      toolCount: Math.max(s.tool_count ?? 0, previous?.toolCount ?? 0),
      tools: previous?.tools.length ? previous.tools : s.last_tool ? [s.last_tool] : []
    })
  }

  // Async records are completion units, not agents. Independent completions can
  // give them different IDs from the actual children; never invent extra rows.

  return [...merged.values()]
}

export function useAgentRoster() {
  const events = useTurnSelector(s => s.subagents)
  const snapshot = useStore($agentSnapshot)
  const { sid } = useStore($uiState)

  return useMemo(() => mergeAgentRoster(events, snapshot.sid === sid ? snapshot.data : EMPTY), [events, snapshot, sid])
}
