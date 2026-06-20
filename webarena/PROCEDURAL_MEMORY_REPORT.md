# WebArena Procedural Memory Report

Generated: 2026-06-19

## Current Run Metrics

Canonical completed tasks counted from `webarena.158` onward:

```text
Completed tasks: 29
Successes: 17
Success rate: 58.62%
Average steps/task: 3.41
```

Latest completed tasks:

```text
226 fail  steps=4
227 pass  steps=2
228 pass  steps=2
229 pass  steps=10
230 fail  steps=4
231 pass  steps=2
232 pass  steps=2
233 pass  steps=2
234 fail  steps=7
235 fail  steps=3
238 fail  steps=6
239 fail  steps=5
240 fail  steps=6
```

Memory database state:

```text
Procedures:          19
Procedure edges:     256
Negative memories:   28
Retrieval events:    380
Selected events:     306
Overall selected rate: ~80.5%
```

Memory storage size:

```text
webarena/memory/procedural/procedural_memory.sqlite3   3.0 MB
webarena/memory/procedural/procedure_embeddings.json   208 KB
webarena/memory/procedural/procedural_manifest.json    4 KB
Total: ~3.2 MB
```

## EC2 / WebArena Setup

The EC2 instance hosts the WebArena websites. The agent and LLM run from the local/server machine, while Playwright opens pages served by EC2.

Flow:

```text
Local Python agent
  -> Playwright browser
  -> EC2 WebArena services
  -> DOM/accessibility text extraction
  -> LLM action decision
  -> Playwright executes click/fill/select
```

Important environment variables:

```bash
WA_SHOPPING="http://18.191.180.130"
WA_SHOPPING_ADMIN="http://18.191.180.130:8083/admin"
WA_REDDIT="http://18.191.180.130:8080/forums/all"
WA_GITLAB="http://18.191.180.130:9001/explore"
WA_WIKIPEDIA="http://18.191.180.130:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
WA_MAP="http://18.191.180.130:443"
WA_HOMEPAGE="http://18.191.180.130"
WA_FULL_RESET="http://18.191.180.130:7565"
```

## Screenshot Warning

This warning is not a serious issue:

```text
Warning: use_screenshot is set to True, but the chat model does not support vision.
Disabling use_screenshot.
```

It means the model is not seeing screenshot images. Instead, it reads the page through DOM/accessibility text.

DOM means the structured HTML representation of the page. Example:

```text
Visible page:
Search box, Search button, Product link, Price

DOM/accessibility text:
[216] textbox "Search"
[221] button "Search"
[1404] link "Amazon Basics Cable"
[1516] tab "Reviews"
```

This is enough for most WebArena shopping/order/review tasks.

## Memory Storage

Main directory:

```text
webarena/memory/procedural/
```

Main files:

```text
procedural_memory.sqlite3
procedure_embeddings.json
procedural_manifest.json
```

SQLite is disk-backed. It stores:

```text
procedures          successful reusable workflows
procedure_edges     graph structure
negative_memory     failed patterns and avoid rules
retrieval_events    every retrieval call and candidate decision
```

Vectors are stored permanently on disk in:

```text
procedure_embeddings.json
```

In normal FAISS mode, those vectors are loaded into RAM and FAISS builds an in-memory index:

```text
Disk: procedure_embeddings.json
RAM:  FAISS IndexFlatIP search index
```

With fast mode:

```bash
WEBARENA_FAST_MEMORY=1
```

we skip FAISS/SentenceTransformer loading and use SQLite + family/lexical/graph scoring.

## Retrieval Flow

Current implementation retrieves on every step:

```text
goal + current page observation + site
```

The agent calls:

```python
procedural_memory.prompt(goal, observation, site)
```

Retrieval stages:

```text
1. Classify task family.
2. Generate candidates.
3. Score candidates.
4. Apply threshold.
5. Select top memories.
6. Inject selected memories into the LLM prompt.
7. Log accepted, raw, and rejected candidates.
```

Task families include:

```text
price_range
product_capacity_search
review_extraction
review_lookup
order_status_total
bought_option_lookup
fulfilled_order_total
product_search
```

Thresholds:

```text
same-family min score: 0.42
cross-family min score: 0.48
```

Scoring uses:

```text
semantic similarity or fast lexical-family score
lexical overlap
page activation overlap
site match
past success/outcome score
graph score
task-family match
action-structure match
negative-memory penalty
```

## Prompting

The LLM receives:

```text
System instructions
Current WebArena goal
Current DOM/accessibility observation
Action history
Error history
Procedural memory
Negative memory
Task-family guardrails
```

Memory is injected as guidance, not hard execution. The model still chooses the next action.

Prompt includes strict answer rules:

```text
Use send_msg_to_user only when ready.
Answer only the requested value/entities.
For price ranges, use "$min - $max".
For order-status tasks, match status exactly.
For no-match tasks, answer the no-match result plainly.
```

## Stop / Loop Guard

We added a prompt stop condition:

```text
If all requested answer fields are known, stop browsing and call send_msg_to_user.
```

We also added a conservative runtime guard for price-range loops. If the agent repeatedly toggles sorting and already has both min and max prices, it forces:

```python
send_msg_to_user("$min - $max")
```

This is scoped only to price-range tasks so it does not interfere with normal pagination, review scanning, or order-history flows.

## LLM Abstraction

After each task, the pipeline runs:

```bash
procedural_memory.py ingest-result
```

It reads:

```text
results/webarena.X/experiment.log
results/webarena.X/summary_info.json
config_files/X.json
```

For successful tasks, Gemini 2.5 Pro creates a reusable memory:

```text
family
goal_pattern
activation_keywords
general_strategy
answer_format
steps
avoid rules
```

For failed tasks, it creates negative memory:

```text
failure family
failure pattern
abstract failure
avoid instructions
penalty
```

## Browser Execution

BrowserGym uses Playwright, not Selenium.

Action flow:

```text
LLM outputs action
BrowserGym parses action
Playwright executes it
Page loads or changes
BrowserGym extracts DOM/accessibility tree
Agent receives new observation
Retrieval + LLM decide next action
```

Actions look like:

```python
click("1382")
fill("216", "Amazon basic")
select_option("1387", "Price")
send_msg_to_user("$0.01 - $28942.99")
```

Element IDs are page-specific and change after navigation, so memories must never blindly reuse old IDs.

## Optimizations Implemented

Current optimizations:

```text
SQLite disk-backed memory
compact procedure JSON
task-family classification
fast retrieval mode
negative-memory filtering
raw/rejected candidate logging
task-family guardrails
answer normalization
price-range loop guard
FAISS AVX2 support in normal mode
BM25 + sentence-transformer hybrid retrieval in normal mode
LLM-based procedural abstraction
top-k memory injection to keep context compact
```

The biggest speed optimization is:

```bash
WEBARENA_FAST_MEMORY=1
```

This avoids loading SentenceTransformer/FAISS for every short-lived WebArena task process.

## Current Limitations

Not yet implemented in WebArena memory:

```text
vector quantization
IVFPQ
HNSW
DiskANN
SQLite FTS5
persistent retrieval daemon
learned reranker
token-budget adaptive compression
shared in-RAM FAISS service
```

The strongest next production improvement would be a persistent memory service:

```text
one long-running process
FAISS index kept warm in RAM
SQLite connection kept open
embedding model kept warm
pipeline calls retrieval over local API
```

That would remove repeated load overhead and make full FAISS retrieval practical again.
