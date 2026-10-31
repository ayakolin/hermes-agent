import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetCardRows } from '../app/cardRows.js'
import { resetConnectionOperationsForTests } from '../app/connectionOperationStore.js'
import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, patchTurnState, resetTurnState } from '../app/turnStore.js'
import { resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

// Replays a gateway-admitted automation turn (a background completion) through the real event handler, the
// way a live session receives it: `message.start` carrying the admission's card, then `message.complete`.
// Answers whether Ink paints a notification row for it, and how many.
vi.mock('../lib/openExternalUrl.js', () => ({ openExternalUrl: () => true }))

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys: vi.fn() },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: (title: string, sections: any[]) =>
        appended.push({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
      setHistoryItems: vi.fn()
    },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  }) as any

const TEXT = '[IMPORTANT: Background process proc_1 completed (exit 0).\nOutput:\nDONE]'
const DISPLAY = 'Background Process Finished: sleep 6; echo DONE'

const replay = async (appended: Msg[]) => {
  const handler = createGatewayEventHandler(buildCtx(appended) as any)
  const sid = 'local-abc'

  await handler({
    admission_id: 'adm-9',
    payload: { display_kind: 'process_complete', display_text: DISPLAY, text: TEXT },
    session_id: sid,
    type: 'message.start'
  } as any)
  await handler({
    payload: { status: 'complete', text: 'done.' },
    session_id: sid,
    type: 'message.complete'
  } as any)
}

describe('an automation message.start card', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    resetOverlayState()
    resetServerRequestsForTests()
    resetConnectionOperationsForTests()
    turnController.fullReset()
    resetCardRows()
  })

  it('draws exactly one notification row across start and complete', async () => {
    const appended: Msg[] = []
    resetTurnState()
    patchTurnState({ todos: [] }) // this suite starts from a clean turn
    await replay(appended)
    const rows = appended.filter(msg => msg.kind === 'event')
    expect(rows.map(row => row.text)).toEqual([DISPLAY])
  })

  it('draws it once when the same admission is replayed', async () => {
    const appended: Msg[] = []
    resetTurnState()
    patchTurnState({ todos: [] })
    await replay(appended)
    const handler = createGatewayEventHandler(buildCtx(appended) as any)
    await handler({
      admission_id: 'adm-9',
      payload: { display_kind: 'process_complete', display_text: DISPLAY, text: TEXT },
      session_id: 'local-abc',
      type: 'message.start'
    } as any)
    expect(appended.filter(msg => msg.kind === 'event').map(row => row.text)).toEqual([DISPLAY])
  })

  it('draws nothing for a typed turn', async () => {
    const appended: Msg[] = []
    resetTurnState()
    patchTurnState({ todos: [] })
    const handler = createGatewayEventHandler(buildCtx(appended) as any)
    await handler({ admission_id: 'adm-1', session_id: 'local-abc', type: 'message.start' } as any)
    await handler({ payload: { status: 'complete', text: 'hi' }, session_id: 'local-abc', type: 'message.complete' } as any)
    expect(appended.filter(msg => msg.kind === 'event')).toEqual([])
  })
})
