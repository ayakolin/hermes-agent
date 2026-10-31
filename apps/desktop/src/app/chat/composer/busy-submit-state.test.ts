import { describe, expect, it } from 'vitest'

import { deriveBusySubmitState } from './busy-submit-state'

import type { BusyInputMode } from '@/store/busy-input-mode'

const base = {
  attachmentCount: 0,
  blockingPrompt: false,
  busy: true,
  busyInputMode: null as BusyInputMode | null,
  compacting: false,
  hasComposerPayload: false,
  hasSteerHandler: true,
  isSteerableText: true
}

describe('deriveBusySubmitState while the busy policy is still loading', () => {
  it('leaves Send enabled so the draft queues instead of dying', () => {
    // A null policy used to disable Send for steerable text, so Enter returned true having done nothing.
    expect(deriveBusySubmitState(base).canSubmit).toBe(true)
  })

  it('still queues an unloaded policy rather than interrupting', () => {
    expect(deriveBusySubmitState(base).busyAction).toBe('queue')
  })

  it('an explicit interrupt keeps redirecting', () => {
    expect(deriveBusySubmitState({ ...base, busyInputMode: 'interrupt' }).busyAction).toBe('interrupt')
  })

  it('idle with no payload cannot submit', () => {
    expect(deriveBusySubmitState({ ...base, busy: false }).canSubmit).toBe(false)
  })
})
