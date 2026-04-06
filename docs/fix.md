🧠 PROMPT

You are working inside an existing repository: ml-finance-risk-system.

Your task is NOT to design architecture or describe anything.

Your task is STRICTLY to IMPLEMENT a fully working system based on the existing SPEC.md and current codebase.

🚨 CRITICAL PROBLEMS TO FIX

The current TUI system is broken and incomplete:

❌ 1. TUI is not connected to real data
All panels show N/A or empty state
No real-time pipeline exists
No live updates from MEXC
No working orchestration loop
❌ 2. Symbol selection is wrong

Symbol is passed only via CLI:

py -m tui BTCUSDT
MUST be moved into TUI runtime
User must be able to:
search symbol inside UI
switch symbol without restart
❌ 3. No real-time inference loop
Models are not continuously running
No async scheduler feeding updates
No streaming updates into UI state
❌ 4. UI shows N/A because:
Feature pipeline is not connected
ONNX inference is not executed live
Orchestrator is missing or broken
🎯 REQUIRED FINAL BEHAVIOR
1. Single command start
py -m project

OR

py -m project tui

NO symbol required at startup.

2. Symbol selection INSIDE TUI

Inside UI:

Press S → opens symbol search overlay
User can:
type symbol (BTCUSDT, ETHUSDT, etc.)
select from list
Symbol switch triggers:
immediate data fetch
feature rebuild
model inference restart

NO restart of application allowed.

3. Real-time loop MUST work

Implement working pipeline:

async loop (1s tick)
    ↓
fetch OHLCV from MEXC
    ↓
build features
    ↓
run ONNX inference (all models)
    ↓
aggregate meta signal
    ↓
push result into TUI state
    ↓
UI re-renders automatically

This MUST be continuous.

4. Fix "N/A" issue completely

Ensure:

MEXC API adapter is working
fallback logic removed or fixed
feature pipeline ALWAYS returns valid numpy arrays
ONNX inference ALWAYS executes (or logs error clearly)
no silent failures allowed

If a model fails:
→ show ERROR state in UI, NOT N/A

5. Add proper STATE SYSTEM

Create a real reactive state store:

Must include:

current_symbol
current_timeframe
latest_ohlcv
latest_features
latest_predictions
latest_meta_signal
risk_state
history (last 50 signals)

TUI MUST ONLY read from state store.

NO direct API calls from UI.

6. Fix architecture wiring

You MUST ensure:

Orchestrator is REAL

It must:

run async loop
manage scheduler
trigger inference
update state store
Scheduler is REAL
periodic fetch per symbol
NOT blocking
Model registry is REAL
loads ONNX once
reuses sessions
no reloading per tick
7. TUI requirements

Implement using Textual:

Must include:
Header:
symbol (clickable / editable)
connection status
latency
Meta panel:
LONG / SHORT / NEUTRAL
Model panel:
scalp / intraday / swing
Risk panel:
real risk score
History panel:
live updating
Status bar:
last update time
cache status
errors
8. Performance rules
inference loop < 200ms target
UI render non-blocking
all I/O async
no blocking pandas operations in main thread
9. REMOVE CLI SYMBOL DEPENDENCY

DELETE behavior:

py -m project BTCUSDT ❌

REPLACE WITH:

py -m project

and inside TUI:

[S] → choose symbol
10. FINAL ACCEPTANCE CRITERIA

System is ONLY correct if:

✔ UI updates every few seconds
✔ switching symbol works without restart
✔ predictions are changing live
✔ no N/A anywhere
✔ ONNX inference actually runs
✔ MEXC data is live
✔ state store drives UI
✔ orchestrator controls everything

🚫 DO NOT DO
do NOT redesign architecture
do NOT write documentation
do NOT explain code
do NOT create pseudo-implementation
do NOT leave placeholders
do NOT return "N/A safe fallback"
✅ OUTPUT

Return:

modified project code
working TUI
working orchestrator
working MEXC live pipeline
working ONNX inference loop
🧨 END