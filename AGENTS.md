# Rules for AI agents (Codex)

## Communication
- Explain in simple language.
- No jargon. If a technical word is required, define it in 1 short line.
- Keep answers short and actionable.
- Be concise.
- Include key reasoning steps briefly.
- Avoid long explanations unless asked.
- Prefer bullet points over paragraphs.
- Talk like a helpful teammate, not a formal report.
- Do not use headings like `Reviewer`, `Builder`, `Risks checked`, `Minimal plan`, `Acceptance checklist`, `What changed`, `Commands run`, or `Short git diff summary` in normal replies.
- Default reply style after work is:
  - what I did
  - what I found
  - what you should do next
- Keep it natural, like chatting to a friend, unless the user explicitly asks for a formal review.

## Project safety (DO NOT BREAK SHARED CODE)
- This codebase is shared by the Backtester AND the Paper Trader (and later the Real Trader).
- Any change MUST keep both systems working.
- Before changing a shared module (`strategies`, `strategy_builder`, `signals_library`, `indicators_library`, `shared/*`, `trading/*`):
  - Identify who imports it (backtester + paper trader at minimum).
  - Prefer changes that are backward-compatible.
  - If a breaking change is unavoidable: update all dependent callers in the same change set.

## Signal schema (NON-NEGOTIABLE)
- Only these signals are allowed:
  - `OPEN_LONG`, `CLOSE_LONG`, `OPEN_SHORT`, `CLOSE_SHORT`, `HOLD`
- Never reintroduce `BUY/SELL/HOLD` or any legacy schema.
- If something expects `BUY/SELL`, fix the caller to use the new schema instead.

## Edit style
- Never rewrite whole files unless explicitly asked.
- Prefer small edits.
- Keep changes inside the requested file unless absolutely necessary.
- After any code change: briefly say which files changed and what command(s) to run to verify.

## Internal workflow
- Internally review first, then edit, then review again.
- Keep that workflow mostly internal unless the user explicitly asks for the formal review format.
- Do not dump internal checklists into normal replies.

## Verification commands (use as applicable)
- `python -m compileall .`
- Run backtest from your normal entry point.
- Run paper trader startup / dry-run mode (if available).
