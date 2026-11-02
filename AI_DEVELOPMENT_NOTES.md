# AI Development Notes

> **Read before submitting:** the sections below describe what actually happened while this project was built with an AI assistant. Part 6 is for **you** to fill in with your own experience. The assessment asks you to explain and modify the code, so run it, read it, and change something small before you submit.

## 1. Which AI tools were used
- **Claude (Anthropic)**, used in a chat session with file-upload and a code sandbox: read the assessment brief, job description and CSV; designed the architecture; wrote the source files, tests and documentation; ran the parts of the code that could run in its sandbox.
- **At runtime** the application can use **Google Gemini** (plain HTTP call, key from `GEMINI_API_KEY`) to interpret questions. **Gemini was never called while building or testing**: the tests replace the network call with a fake.
- *(Add any other tools you personally used, e.g. ChatGPT, Copilot, Cursor. Delete this line if none.)*

## 2. How AI was used during development
1. **Read the inputs first.** The brief, the job description and the CSV were read before any code was written. The CSV was profiled with Pandas (types, blanks, duplicates, date format, invalid values, returns).
2. **Decided the architecture** around one rule: *the LLM only classifies the question; Python validates; SQL calculates; Python writes the answer.* This directly answers the brief's requirement to explain how unsupported or incorrect answers are prevented.
3. **Generated the code module by module** (cleaning -> database -> analytics -> interpretation -> API -> UI), smoke-testing each module on the real CSV before moving on.
4. **Cross-checked numbers independently.** Totals from SQL were compared with a separate Pandas calculation; product, region and month breakdowns were each confirmed to add up to the same grand total (₹14,39,200).
5. **Wrote tests that do not just assert constants:** expected values are recomputed with Pandas from the raw CSV, and a small dataset with hand-worked answers proves that Returned sales are excluded (counting them would change the top product).
6. **Mutation-checked the tests.** Bugs were injected on purpose into a scratch copy (returns counted as revenue, discount ignored, dates parsed month-first, SQL guard removed, LLM-intent whitelist removed, database opened read-write, zero quantity accepted, fallback mislabelled as LLM, month validation removed). Every one made tests fail, so the suite does detect those faults.

## 3. Problems encountered
| Problem | Detail |
|---|---|
| **Dirty data** | `TXN0034` has quantity 0, `TXN0051` has no unit price, `TXN0033` is Returned. Dates are `DD-MM-YYYY`, which a careless parser reads as month-first. |
| **Ambiguous requirements** | The brief does not say how returns affect revenue or what currency applies. Chosen: returns excluded entirely; INR assumed. Both are documented and the currency is configurable. |
| **The literal example question can't be answered as written** | "How did revenue change between two months?" names no months. The app asks which months instead of guessing. |
| **Silent wrong answers** | "Total revenue in March" would naively return the all-time total. Added an explicit check that declines month filters it cannot honour. |
| **LLM output is untrusted** | A model could return an invented intent, extra text, or markdown fences. Handled by JSON extraction + whitelist + month-format validation, with automatic fallback. |
| **Prompt braces** | The JSON example inside the system prompt contains `{ }`, and the month list is filled in with `str.format()`, so the braces are doubled; a test checks the prompt renders. |
| **Sandbox limits** | The build environment had no network, so `pytest`, FastAPI and Streamlit **could not be installed there**. |
| **Shell quirk** | `mkdir -p dir/{a,b}` created a folder literally named `{a,b}` because the shell was `dash`. Spotted in the `ls` output, deleted and recreated. |
| **Hard-coded text** | A first version of the "data quality" sentence hard-coded the rejection reasons. Found during review; now generated from the actual reasons. |
| **Library drift** | `use_container_width` in Streamlit is being deprecated, so it was removed; month names sort alphabetically in charts, so the chart uses `YYYY-MM` keys. |

## 4. How problems were debugged
- **Data issues:** profiled the raw CSV before designing; every rule maps to something observed in the file or a documented assumption.
- **Logic errors:** compared each SQL result with an independent Pandas result, then checked that the pieces sum to the total.
- **Safety:** wrote adversarial tests (SQL text, `DROP TABLE`, injection-style month strings, hostile LLM replies, LLM outage) and confirmed the table is intact afterwards and the connection is physically read-only.
- **Framework-free design for testability:** because FastAPI/Streamlit were not installable in the sandbox, all logic lives in plain Python modules (`data_loader`, `database`, `analytics`, `ai_handler`) and the web layer is a thin wrapper. The tests for those modules were executed with a small stand-in test runner (**142 passed, 0 failed** after the Gemini integration; `test_api.py` skipped). That stand-in is not part of this repo.

### What was *not* verified by the AI (please check on your machine)
- `tests/test_api.py` (FastAPI `TestClient`) was written but **not executed** in the sandbox; it is skipped there because FastAPI is not installed. Run `pytest -v` locally.
- `app/main.py` and `frontend/streamlit_app.py` were syntax-checked only. Start both and click through the five example questions.
- A **real** Gemini call was never made. `_call_llm` follows Google's documented REST format but has not run against the live service; the default model name in `config.py` may need changing via `GEMINI_MODEL`.
- Dependency versions in `requirements.txt` are minimum floors and were not installed together in one environment here.

## 5. What I would improve with one more day
1. Run and fix anything found in the first real end-to-end pass (live LLM call, Streamlit click-through, `pytest` in a clean venv); pin exact working versions.
2. Support filters: revenue by category/channel, single-month questions, "top N", and date ranges, still through whitelisted query functions.
3. Let the LLM extract *parameters* (month, product, region) using Gemini's structured-output (JSON schema) option and a validated Pydantic schema.
4. Add an optional, LLM-written *narrative* of the result that is checked so every number in it matches the SQL output.
5. Conversation memory for follow-ups ("and what about March?").
6. Use `Decimal`/integer money, product-name normalisation, and configurable return/refund rules.
7. CI: GitHub Actions running `pytest`, plus a Dockerfile only if deployment is actually needed.
8. Small AWS touch relevant to the job description: deploy the API on Lambda/App Runner and call Bedrock instead of Gemini.

## 6. Gemini integration (second iteration)
**Goal:** satisfy the "AI integration" requirement with the smallest possible change to a working app.

**Starting point.** The first version already had an LLM path, but through LangChain with OpenAI/Anthropic, never run live. The brief for this iteration asked for Gemini and no AI framework, so that one function was replaced rather than adding a second path next to it.

**What changed**
| File | Change |
|---|---|
| `app/ai_handler.py` | `_call_llm` now sends one HTTPS request to Gemini's `generateContent` endpoint using `requests`. The four non-total intents were renamed to the required names. The failure note now shows our own safe message (e.g. `Gemini returned HTTP 429`) |
| `app/config.py` | Settings reduced to `GEMINI_API_KEY`, `GEMINI_MODEL`, `LLM_TIMEOUT` |
| `app/main.py` | `/health` reports `llm_provider: "gemini"` when a key is set |
| `requirements.txt` | LangChain packages removed; **no new dependency** (`requests` was already there) |
| `.env.example`, `README.md`, tests | Updated; new `tests/test_gemini_integration.py` |

**Unchanged on purpose:** CSV cleaning rules, revenue calculation, SQL queries, database code, validation, the keyword matcher, API response shape, Streamlit layout.

**Flow:** question -> (SQL/delete text refused) -> Gemini returns JSON intent -> Python checks the intent against five names and the months against `YYYY-MM` and the available data -> fixed SQL -> Python formats the answer. Any problem (no key, timeout, HTTP error, odd reply) switches to the existing keyword matcher and the response says so.

**Security decisions**
- Gemini receives only the question and the list of months: no data, no SQL, no results.
- Only `intent`, `month_a`, `month_b` are read from its reply; every other field is ignored, so an LLM cannot inject numbers or SQL.
- The API key is sent in a request header, never in the URL (error messages from `requests` include URLs), and error bodies are never copied into responses or logs; only the HTTP status code is.
- Key comes from the environment / `.env` (git-ignored); `LLMSettings` hides it from `repr`; `/health` never returns it.
- Timeout is always set (default 15 s), so a slow Gemini cannot hang the API.
- The database connection is opened read-only as the last line of defence.

**Testing the integration.** `requests.post` is replaced by a fake, so the real request-building, response-parsing, validation and SQL code run. Covered: valid intent, month parameters, garbled/invalid replies, wrong response structure, `unsupported`, ambiguous comparison, missing key (Gemini never called), timeout / connection error / HTTP 400, 403, 429, 500, 503, key never present in URL or output, no business data in the request, bogus LLM-supplied numbers ignored (hand-calculated database gives 1140), LLM-supplied SQL never executed. Mutation checks: putting the key in the URL, copying error bodies, removing the timeout, removing the intent whitelist, bypassing the key check, trusting LLM numbers, and removing the fallback each made tests fail.

## 7. Your own notes (edit before submitting)
- What I changed or customised myself: *…*
- A bug or surprise I hit when I ran it: *…*
- One design decision I can defend in the interview: *…*
