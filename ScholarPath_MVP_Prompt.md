# ScholarPath MVP — Build Prompt

You are a senior Python engineer. Build **ScholarPath**, an AI-powered platform that helps students discover educational opportunities (degree programs, scholarships, fellowships, bootcamps, online courses) that match their academic background and career goals.

Deliver a working, runnable, tested MVP. Not a sketch, not pseudocode, not an architecture document. Every file you create must be complete and executable.

## Non-negotiables

1. **LangGraph orchestrates exactly three agents.** Each agent is a graph node with its own responsibility, its own input/output schema, its own tools, and its own system prompt. No agent may do another agent's job. If the three agents could be swapped for three plain LLM calls with different names, you have failed.
2. **Runs end-to-end from one CLI command** and writes a report file.
3. **Testable offline.** The full test suite passes with no API keys, using a fake LLM and a local dataset.
4. **Small and clean.** No extra agents, no vector database, no auth, no frontend framework, no abstractions for hypothetical futures. Structure it so a fourth agent or a new tool is a one-file addition.

## Stack

Python 3.11+, `uv` (pip fallback), `langgraph`, `langchain-core`, `langchain-anthropic` (model from env `ANTHROPIC_MODEL`, default `claude-sonnet-5-5`), `pydantic` v2, `tavily-python` (optional web search), `fastapi` + `uvicorn` (thin), `pytest`, `ruff`. Do not add dependencies beyond this list.

## The three agents

### Agent 1 — Reader (Profile Analyst)

- **Input:** raw student text: a free-form paragraph and/or pasted CV or transcript text, read from a `.txt`/`.md` file or stdin.
- **Output:** `StudentProfile` (pydantic): `education_level`, `field_of_study`, `gpa` (optional, normalized to a 0–4 scale), `institution`, `graduation_year`, `skills`, `interests`, `career_goals`, `constraints` (countries, budget, language, timeline, study mode), plus `missing_fields: list[str]` and `confidence: float`.
- **Behavior:** uses structured output (`with_structured_output`). Never invents values: unknowns are `None` and listed in `missing_fields`. Rule-based post-validation after the LLM call: GPA scale conversion, graduation-year sanity, dedupe skills.
- **Specialization:** the only agent that touches raw user text. It has no tools and produces no prose.

### Agent 2 — Searcher (Opportunity Scout)

- **Input:** `StudentProfile`.
- **Output:** ranked `list[Opportunity]` (pydantic): `name`, `provider`, `type` (enum: degree, scholarship, fellowship, bootcamp, course), `url`, `country`, `deadline`, `cost_or_funding`, `eligibility_summary`, `source` (catalog | web), `match_score` (0–100), `match_reasons: list[str]`, `concerns: list[str]`.
- **Tools** (LangChain `@tool`):
  - `search_local_catalog(query, filters)` over `data/opportunities.json`. Seed it with 25–40 realistic, clearly labeled sample entries covering all five types and several regions.
  - `search_web(query)` via Tavily. Enabled only when `TAVILY_API_KEY` is set; otherwise skipped and the skip is recorded in state.
- **Behavior:** a bounded tool-calling loop (`create_react_agent` or a manual tool node), maximum 6 tool calls. Generates 3–5 targeted queries from the profile, dedupes candidates by normalized name + provider, scores each against the profile with explicit reasons and concerns, returns the top 8. Writes a `search_log` (queries issued, hits per source) to state.
- **Specialization:** the only agent with tools. It never writes text addressed to the user.

### Agent 3 — Reporter (Advisor)

- **Input:** profile + ranked opportunities + search_log.
- **Output:** a markdown report: profile summary, top matches (each with why it fits, concerns, deadline, concrete next action), profile gaps and how filling them would improve results, a 30/60/90-day action plan, and a data sources and limitations section. Saved to `reports/<timestamp>_<slug>.md`.
- **Behavior:** grounded. It may reference only opportunities present in state. Add a post-check: every opportunity name in the report must exist in state; if not, regenerate once with a correction message, then fail loudly. Tone: concise student advisor.
- **Specialization:** the only agent that writes to the user. It has no tools and never searches.

## Graph (LangGraph `StateGraph`)

- **State:** one `ScholarPathState` (TypedDict or pydantic): `raw_input`, `profile`, `opportunities`, `search_log`, `report`, `report_path`, `errors`, `clarification_rounds`, `refinement_rounds`, `interactive`.
- **Nodes:** `reader` → `searcher` → `reporter`.
- **Conditional edges (required, this is the orchestration):**
  - **After reader:** if `confidence < 0.5` or a critical field is missing (`field_of_study`, `career_goals`) and `interactive=True` and `clarification_rounds < 1`: call `interrupt()` asking the user for the missing information, merge the answer into `raw_input`, re-run reader. Otherwise proceed with gaps recorded.
  - **After searcher:** if fewer than 3 opportunities have `match_score >= 50` and `refinement_rounds < 1`: route back to searcher with a `broaden=True` flag that relaxes filters. Otherwise go to reporter.
  - **reporter → END.**
- **Checkpointer:** `MemorySaver` with a `thread_id` per run so interrupt and resume work.
- Compile once in `graph.py` and expose `build_graph()`.

## Interfaces

- **CLI:** `python -m scholarpath run --input examples/student_1.txt [--interactive] [--out reports/]`. Prints one progress line per node and the final report path. `python -m scholarpath demo` runs every file in `examples/`.
- **API:** `POST /recommend` with `{"text": "..."}` → `{"profile": ..., "opportunities": [...], "report_markdown": "..."}`. One file, no auth, non-interactive mode only.

## Project layout

```
scholarpath/
  __init__.py
  __main__.py
  config.py          # env loading, settings
  state.py           # ScholarPathState
  schemas.py         # StudentProfile, Opportunity, enums
  llm.py             # get_llm(); returns a fake chat model when SCHOLARPATH_FAKE_LLM=1
  graph.py           # build_graph(), routing functions
  agents/
    reader.py        # prompt constant + node function
    searcher.py
    reporter.py
  tools/
    catalog.py
    web_search.py
  cli.py
  api.py
data/opportunities.json
examples/student_1.txt, student_2.txt, student_3.txt   # varied: strong profile, vague profile, constrained budget
tests/
README.md
.env.example
pyproject.toml
Makefile             # setup, test, lint, demo
```

## Tests (pytest, no network, no API keys)

- `test_reader.py`: fake LLM returns fixed JSON → profile parsed; `missing_fields` correct; GPA on a 100-point scale is normalized.
- `test_searcher.py`: catalog tool returns expected matches for a profile; dedupe and top-k work; web search is skipped cleanly when no key is set.
- `test_reporter.py`: grounding check rejects a report naming an opportunity not in state and triggers one regeneration.
- `test_graph.py`: low-confidence profile in interactive mode hits the interrupt; sparse results route back to searcher exactly once; happy path reaches END with a report file written.
- One `@pytest.mark.integration` test against the real API, auto-skipped when `ANTHROPIC_API_KEY` is absent.

## Engineering rules

- Type hints everywhere. Pydantic for all LLM input and output. No bare dicts passed between nodes.
- Prompts live as constants at the top of each agent file.
- `logging` for internals; `print` only in the CLI.
- A single tool failure is written to `state.errors` and never crashes the graph.
- `ruff check .` and `ruff format --check .` pass.
- README covers: what it does, a mermaid diagram of the graph, setup, run, test, and a short "how to add a fourth agent or a new tool" section.

## Process

1. Scaffold, schemas, state, fake LLM.
2. Agents one at a time, each with its tests green before moving on.
3. Graph with routing tests.
4. CLI and API.
5. README.

Then run `pytest` and `python -m scholarpath demo` with `SCHOLARPATH_FAKE_LLM=1`, and once more with a real key if one is available. Paste the actual terminal output. If anything fails, fix it; do not describe the fix.

## Definition of done

- `uv sync && pytest` passes offline.
- `python -m scholarpath run --input examples/student_1.txt` writes a report into `reports/`.
- Each agent demonstrably differs in schema, tools, and prompt.
- The graph has two conditional edges, both exercised by tests.
- No TODOs, no placeholder functions, no "in a real implementation..." comments.
