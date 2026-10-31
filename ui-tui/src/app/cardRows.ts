// Admission cards this process already drew. A start replayed after a reconnect (or an idle replay of the
// same admission) must not repeat its `◈` row: the notification belongs to the admission, not to the frame.
const CARD_ROWS_LIMIT = 200

const drawn = new Set<string>()

/** True the first time this session's admission card is drawn; false on every replay. */
export function claimCardRow(sessionId: string, admissionId: string): boolean {
  const key = `${sessionId}\u0000${admissionId}`

  if (drawn.has(key)) {
    return false
  }

  drawn.add(key)

  if (drawn.size > CARD_ROWS_LIMIT) {
    drawn.delete(drawn.values().next().value as string)
  }

  return true
}

/** The notification card an admitted automation turn's `message.start` carries (`display_kind` /
 *  `display_text`, `gateway.session_automation.admission_display`). Null for an ordinary typed turn, and for
 *  an admission whose card this process already drew. */
export function admissionCard(ev: { admission_id?: null | string; payload?: unknown; session_id?: null | string }): null | string {
  const payload = ev.payload as { display_kind?: unknown; display_text?: unknown } | undefined
  const kind = payload?.display_kind
  const display = payload?.display_text

  if (typeof kind !== 'string' || !kind || typeof display !== 'string' || !display.trim() || !ev.admission_id) {
    return null
  }

  return claimCardRow(ev.session_id ?? '', ev.admission_id) ? display.trim() : null
}

export function resetCardRows(): void {
  drawn.clear()
}
