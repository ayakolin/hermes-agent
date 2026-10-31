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

export function resetCardRows(): void {
  drawn.clear()
}
