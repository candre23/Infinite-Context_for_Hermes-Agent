# Infinite Context for Hermes Agent

Version 0.9.2

Infinite Context is a Hermes Agent `ContextEngine` plugin for long-running conversations and durable memory. It keeps the authoritative Hermes transcript intact while selecting a bounded provider prompt from recent conversation, retrieved historical context, and scoped durable memories.


## Features

- Keeps recent complete turns verbatim.
- Hybrid lexical + semantic retrieval of older conversation chunks.
- Structured-anchor matching for IDs, paths, hashes, dotted names, and similar identifiers.
- Request-only trimming of oversized historical tool results without changing the stored transcript.
- Ingests Hermes `<persisted-output>` spillover files before they disappear.
- Background idle-time memory curation using the configured local inference backend.
- Durable memory provenance, revisions, confidence, importance, reinforcement, access tracking, and reversible retirement.
- Memory scopes:
  - `global` — eligible across chats.
  - `project` — shared only by chats bound to the same project namespace.
  - `session` — limited to the source chat.
- Manual project binding plus conservative automatic association when a new chat explicitly says it is continuing a known project from other chats.
- Diagnostic/status commands and SQLite-backed persistence.


## How Infinite Context compares to other Hermes memory systems

Infinite Context takes a different approach than other memory modules. It is not just a database of remembered facts. It maintains a pristine record of all interactions and acts as a "context engine", deciding which portions of that history history should actually be shown to the model on each turn.

**Mem0** is a good choice when you want hands-off fact extraction, semantic search, deduplication, and either hosted or self-managed storage. Its focus is primarily on turning conversations into compact long-term memories and retrieving those memories later. Infinite Context is preferable when preserving and selectively recalling the **original conversational context itself** matters more than simply extracting summaries from it.

**Honcho** goes further in the direction of user modeling. It builds persistent representations of the user, session summaries, conclusions, and dialectic reasoning about preferences and behavior. That can be excellent for personalization and multi-agent systems. Infinite Context is deliberately less interpretive: it emphasizes provenance, retrieval of source material, project/session scoping, and avoiding the conversion of assistant speculation into durable truth.

**Hindsight** is strong when relationships between facts are important. It uses entity resolution, knowledge-graph-style memory, multi-strategy retrieval, and a special reflection operation for synthesizing information across memories. Infinite Context is simpler structurally, but has an advantage for extremely long Hermes sessions because it directly controls the model's working context and can retrieve relevant old turns alongside curated memories.

**OpenViking** is probably the closest philosophical alternative. It is self-hostable, automatically extracts memories, organizes information into a filesystem-style hierarchy, and supports tiered retrieval from summaries through full content. It is especially attractive when you want a browsable knowledge repository containing memories and external resources. Infinite Context is more tightly focused on Hermes conversation continuity: raw transcript history, exact identifiers, evolving versions of facts, session/project/global scopes, and automatic selection of the context sent to the LLM.

Infinite Context is therefore most attractive when the goal is not merely **“remember facts about me”**, but **“let Hermes accumulate an unlimited, exact working history while presenting only the right pieces of that history to the model at the right time.”** It is specifically tailored to local models with finite context windows and to long-running technical or creative projects spread across multiple chats.


## Requirements

- A Hermes Agent build with pluggable `ContextEngine` support.
- Python environment used by Hermes.
- `fastembed>=0.7,<0.8` for semantic retrieval. The installer can install it automatically.
- Local model access for idle-time memory curation. The engine uses the inference configuration already exposed by Hermes.

The semantic model is `BAAI/bge-small-en-v1.5` and is downloaded by FastEmbed on first use.


## Installation

Clone/download the repository, then run:

```bash
./install.sh
```

The installer:

1. finds the Hermes Python runtime;
2. syntax-checks the plugin;
3. installs/checks FastEmbed unless disabled;
4. backs up an existing `infinite_v0` plugin directory;
5. copies the plugin to `~/.hermes/hermes-agent/plugins/context_engine/infinite_v0/`.

If Hermes is installed elsewhere:

```bash
HERMES_REPO=/path/to/hermes-agent ./install.sh
```

To install without semantic embeddings:

```bash
HERMES_INFINITE_SKIP_EMBEDDINGS=1 ./install.sh
```

Ensure Hermes configuration selects the engine:

```yaml
context:
  engine: infinite_v0
```

Restart Hermes after installation and verify with:

```text
/infinite status
```

## Updating

Unless stated otherwise, new versions can simply be installed over the old version with the included script.


## Commands

```text
/infinite status [session-prefix]
/infinite sessions
/infinite embed [session-prefix]
/infinite project status
/infinite project set <project-name>
/infinite project clear
/infinite memory status
/infinite memory list [limit]
/infinite memory run [session-prefix]
/infinite cleanup
```

`/infinite project set` queues the requested project assignment and binds it to the next authoritative interactive request. This avoids relying on stale UI/session state exposed to slash-command handlers.


## Automatic project association

An unbound chat may associate itself with an existing project before retrieval when the user explicitly describes continuity with prior chats, for example:

```text
We're continuing the jukebox software work we've been doing in other chats.
```

Association is deliberately conservative. It requires both continuity/cross-chat language and meaningful evidence from a known project label. Semantic similarity alone cannot silently bind a chat.


## Durable memory behavior

Memory curation runs after an idle period rather than blocking ordinary dialogue. `/infinite memory run` forces an immediate pass when needed.

The curator is designed to prefer durable user preferences, stable environment facts, and project conventions while rejecting transient task state, creative-writing content, assistant-only claims, and secrets/credentials. Stored memories retain source-session/turn evidence.

Hermes' own native memory system remains separate. Infinite Context does not disable or replace Hermes' explicit global-memory behavior.


## Storage

The main database is:

```text
~/.hermes/context_engine/infinite_v0.sqlite3
```

A best-effort diagnostic trace is written to:

```text
~/.hermes/context_engine/infinite_v0_trace.log
```

Hermes' persisted conversation transcript remains authoritative; Infinite Context does not destructively rewrite it.


## Notes

- Project binding currently has limited UI visibility because Hermes' current web dashboard does not expose first-class project controls. A future UI can present project assignment as metadata without placing control instructions in the model-visible prompt.
- Deleting a Hermes session does not automatically delete every consolidated Infinite memory derived from it. `/infinite cleanup` removes orphaned indexed session data, while durable-memory lifecycle remains conservative.
- Importance/retirement thresholds are intentionally conservative and may need tuning with long-term real-world use.


## Version History

**0.9.2**  Added better idle monitoring (user typing registers as activity) when hermes cockpit is present

**0.9.1** Added some memory guardrails to prevent OOM crashes in edge-case scenarios

**0.9.0** Initial release


## AI & Safety Disclaimer

The code and documentation included in this project is primarily vibeslop. The human writing this sentence in particular can barely code and doesn't really understand how any of this works. It Works On My Machine and hasn't caused my genitals to explode, but your mileage may vary. I make absolutely no guarantee as to the safety or security of the contents of this project. Use at your own risk. Or don't.


## License

Infinite Context is released into the public domain under the [`Unlicense`](LICENSE).   
Copyleft 2026   
No Rights Reserved

