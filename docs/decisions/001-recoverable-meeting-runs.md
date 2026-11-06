# Recoverable meeting runs

Status: Accepted

## Context

ASR is expensive, and an unavailable LLM endpoint should not discard completed
transcription. The previous pipeline wrote its document only after every stage
had succeeded and removed temporary audio on failure.

## Decision

Use a persistent work directory with a versioned JSON checkpoint. Save each
completed ASR track, diarization result, polished transcript, and summary using
atomic file replacement. Resume is explicit; an ordinary run starts fresh.

Stage keys hash input content and the settings or upstream results relevant to
that stage. Configuration values and environment secrets are not stored in the
checkpoint. A changed key causes recomputation, so changing LLM settings can
retain ASR results. Checkpoints retain original ASR text independently of the
polished document.

Publish the readable document after speaker assignment and update it after
polishing and summarization. Each output file is atomically replaced; Markdown
and its JSON sidecar are separate files, not a transactional pair.

## Consequences

Work directories survive success and failure, allowing retry without repeating
completed inference. They contain meeting audio and text and consume disk until
removed by the user. Existing meeting JSON remains a document interchange
format; checkpoint metadata is kept separately so `summarize` remains compatible.
