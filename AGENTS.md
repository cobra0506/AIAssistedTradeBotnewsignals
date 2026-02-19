# Rules for AI agents (Codex)

## Communication
- Explain in simple language.
- No jargon. If a technical word is required, define it in 1 short line.
- Keep answers short and actionable.
- Be concise
- Include key reasoning steps briefly
- Avoid long explanations unless asked
- Prefer bullet points over paragraphs

## Project safety (DO NOT BREAK SHARED CODE)
- This codebase is shared by the Backtester AND the Paper Trader (and later the Real Trader).
- Any change MUST keep both systems working.
- Before changing a shared module (strategies, strategy_builder, signals_library, indicators_library, shared/*, trading/*):
  - Identify who imports it (backtester + paper trader at minimum).
  - Prefer changes that are backward-compatible.
  - If a breaking change is unavoidable: update all dependent callers in the same change set.

## Signal schema (NON-NEGOTIABLE)
- Only these signals are allowed:
  - OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, HOLD
- Never reintroduce BUY/SELL/HOLD or any legacy schema.
- If something expects BUY/SELL, fix the caller to use the new schema instead.

## Edit style
- Never rewrite whole files unless explicitly asked.
- Prefer small edits and show exact “replace this with this”.
- Keep changes inside the requested file unless absolutely necessary.
- After any code change: show a short `git diff` summary and what command(s) to run to verify.

## Two-agent workflow (Builder + Reviewer)
You MUST follow this workflow unless the user explicitly says "skip review".

### Role: REVIEWER (default first)
- Do NOT modify files.
- Read the request and the relevant code.
- Output:
  1) Risks / things that could break backtester or paper trader
  2) Proposed minimal change plan (file + function names)
  3) Exact acceptance checklist (what must be true after changes)

### Role: BUILDER (only after Reviewer plan)
- Apply ONLY the minimal changes approved by the Reviewer plan.
- Do not make extra “nice to have” refactors.
- After edits, output:
  - What changed (files + brief)
  - Why it won’t break the other system(s)
  - Commands to run

### Role: REVIEWER (after Builder edits)
- Do NOT modify files.
- Review the diff and confirm:
  - Signal schema still correct
  - No shared module breakage
  - Imports still valid
  - No new assumptions added
- If approved, say: "APPROVED ✅"
- If not, say: "CHANGES REQUIRED ❌" and list exact fixes

## Verification commands (use as applicable)
- `python -m compileall .`
- Run backtest from your normal entry point
- Run paper trader startup / dry-run mode (if available)
