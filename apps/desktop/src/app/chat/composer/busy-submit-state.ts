import type { BusyInputMode } from '@/store/busy-input-mode'

export type ComposerBusyAction = 'interrupt' | 'steer' | 'queue' | 'stop'

export interface BusySubmitInputs {
  attachmentCount: number
  blockingPrompt: boolean
  busy: boolean
  busyInputMode: BusyInputMode | null
  compacting: boolean
  hasComposerPayload: boolean
  hasSteerHandler: boolean
  isSteerableText: boolean
}

export interface BusySubmitState {
  busyAction: ComposerBusyAction
  canSteer: boolean
  canSubmit: boolean
}

/** The Send button's gate plus the busy-turn routing (steer / queue / stop). */
export function deriveBusySubmitState({
  attachmentCount,
  blockingPrompt,
  busy,
  busyInputMode,
  compacting,
  hasComposerPayload,
  hasSteerHandler,
  isSteerableText
}: BusySubmitInputs): BusySubmitState {
  // An unloaded busy policy (null) is a queue, not a dead Send: the submit hook takes that branch and queues
  // the draft, so the button must stay enabled while the policy is still being read.
  const canSubmit = busy || hasComposerPayload

  // Steer only makes sense mid-turn, text-only (the gateway can't carry images
  // into a tool result) and never for a slash command (those execute inline).
  // A blocking prompt (approval/sudo/secret) also rules it out: the tool batch
  // is parked on the user, so a steer can't reach the model — text queues.
  // Compaction does not: the gateway holds the correction until it finishes.
  const canSteer = busy && !blockingPrompt && hasSteerHandler && attachmentCount === 0 && isSteerableText

  // Ordinary busy Send follows backend policy; attachments queue and empty stops. Null is a queue too,
  // matching what the submit hook does with it (never an unintended interrupt).
  const busyAction: ComposerBusyAction = canSteer
    ? (busyInputMode ?? 'queue')
    : compacting || hasComposerPayload
      ? 'queue'
      : 'stop'

  return { busyAction, canSteer, canSubmit }
}
