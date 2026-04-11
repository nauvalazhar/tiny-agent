# tiny-agent

A coding agent CLI built from scratch in a single TypeScript file. It can read, edit, and create files, run shell commands, search the web, and plan before executing. Built with Bun and the Anthropic SDK.

This started as a learning project to understand how tools like Claude Code, Cursor, and Aider work under the hood. Turns out the core is simpler than you'd think.

<div align="center">
<img width="764" height="548" alt="image" src="https://github.com/user-attachments/assets/3e17e53b-d0a9-4d76-a6ab-59dd3e49b83d" />
</div>


## How it works

The whole thing is a `while(true)` loop:

1. Send the conversation to Claude
2. Stream the response to the terminal
3. If the response has tool calls, execute them
4. Send the results back to Claude
5. Repeat until Claude responds with just text (no tool calls)

That's the entire architecture. Everything else (permissions, streaming, subagents, persistence) is built on top of this loop.

## Features

### Tools

- `read_file` - read files with line numbers
- `edit_file` - find-and-replace with quote normalization
- `write_file` - create or overwrite files
- `glob` - list files by pattern
- `grep` - search file contents with RE2 (no ReDoS)
- `shell` - run shell commands
- `web_search` - search the web via Anthropic's server tool
- `web_fetch` - fetch a URL, convert to markdown, extract info with Haiku
- `spawn_agent` - run a subagent with its own conversation and tools

### Permissions

- Each tool declares its own permission policy: `allow`, `ask`, or `deny`
- Write tools ask for permission. Read tools auto-allow.
- "Always allow" saves your choice for the rest of the session
- The model never sees the permission prompt. It just gets the result or an error.
- Shell commands have a regex allowlist for safe commands like `ls`, `pwd`, `git status`

### Plan mode

- Toggle with `/plan`
- Same agent loop, but write tools are blocked and the system prompt tells Claude to plan instead of execute
- When the plan is ready, Claude calls `exit_plan_mode` which shows the plan and asks if you want to proceed

### Subagents

- `spawn_agent` calls the same `agentLoop` function recursively
- Different system prompt, different tool set, fresh conversation
- The parent only sees the final summary, not the subagent's internal tool calls
- Shares the session (permissions, file timestamps, spinner) with the parent

### Parallel execution

- Tools marked `isConcurrencySafe` run in parallel via `Promise.all`
- Unsafe tools (edit, write, shell) run one at a time
- Permission prompts always run sequentially (can't show two prompts at once)
- Results are merged back in the original order using a Map lookup

### Streaming

- Responses stream token by token to the terminal
- Text deltas print immediately via `process.stdout.write`
- Tool use input (JSON) accumulates silently until the block is complete
- Usage stats (input/output/cached tokens) are captured from stream events

### Context management

- Old tool results are replaced with "[cleared to save context]"
- When token estimate exceeds the threshold, the conversation is summarized by Haiku
- If summarization fails, old messages are dropped as fallback
- Prompt caching on system prompt and tool definitions (90% cheaper on cache hits)

### Persistence

- Each session is stored as a JSONL file (one JSON per line, append-only)
- Sessions live under `~/.tiny-agent/projects/<sanitized-cwd>/`
- `/resume` lists past sessions with the first user message as a label
- `/clear` starts a fresh session file

### Error handling

- SDK errors (401, 403, 429, 529, context length, network) are classified and shown as friendly messages
- Tool errors are returned as strings to the model (not thrown)
- Persistence failures warn but don't crash the conversation
- Compaction failures fall back to dropping old messages
- Stream interruptions return partial content instead of losing everything
- Retry with exponential backoff on 429/529 (1s, 2s, 4s)

### Security

- All file tools validate paths are inside the working directory (no path traversal)
- Shell command allowlist rejects commands with metacharacters (`;`, `|`, `&&`, etc.)
- Web fetch blocks private IPs, localhost, and AWS metadata endpoint
- Web fetch rejects URLs with embedded credentials
- Session and config files are written with mode 0o600

## Setup

```bash
# install dependencies
bun install

# run directly
bun tiny-agent.ts

# or link globally
bun link
# then run from any project directory
tiny-agent
```

On first run, you'll be prompted for your Anthropic API key. It gets validated against the API and saved to `~/.tiny-agent/config.json`.

You can also set `ANTHROPIC_API_KEY` as an environment variable (takes precedence over the config file).

### Commands

| Command   | What it does                                        |
| --------- | --------------------------------------------------- |
| `/plan`   | Toggle plan mode (plan before executing)            |
| `/clear`  | Clear conversation and start a new session          |
| `/resume` | Resume a previous session from this directory       |
| `/login`  | Update your API key                                 |
| `/logout` | Clear your API key and exit                         |
| `/usage`  | Toggle token usage display                          |
| `/tokens` | Show estimated token count for current conversation |

### Project structure

It's one file: `tiny-agent.ts`. That's the whole project.

The file is organized top to bottom:

1. Constants and prompts
2. Type definitions
3. Helper functions (config, path safety, persistence, retry, context management)
4. Tool definitions
5. Stream handler
6. Tool executor
7. Agent loop
8. Onboarding
9. Main entry point

### Dependencies

- `@anthropic-ai/sdk` - Claude API client
- `@inquirer/prompts` - terminal input and select prompts
- `fast-glob` - file pattern matching (used by glob and grep tools)
- `lru-cache` - web fetch result caching
- `ora` - terminal spinner
- `picocolors` - terminal colors
- `re2-wasm` - safe regex (no ReDoS, runs as WASM, no native bindings)
- `turndown` - HTML to markdown conversion (used by web fetch)
- `zod` - schema validation for tool inputs

## How it compares to Claude Code

This project implements the same core patterns:

- Same `while(true)` agent loop
- Same tool use protocol (`tool_use blocks`, `tool_result` responses)
- Same permission model (harness-level, hidden from the model)
- Same streaming approach (`content_block_start/delta/stop` events)
- Same JSONL persistence format
- Same prompt caching strategy
- Same subagent architecture (recursive loop with isolated conversation)
- Same plan mode design (permission flag + system prompt injection)
- Same quote normalization for edit operations

### What Claude Code has that this doesn't:

- Ink (React for terminal) instead of raw stdout
- Ripgrep instead of `fast-glob` + `fs.readFile`
- Mid-stream tool execution
- Escape to interrupt (requires Ink for proper `stdin` handling)
- Git worktrees for isolated execution
- MCP server integration
- Memory system
- Hooks (user-configured shell commands on events)
- Multi-file tool definitions instead of one big file

## Cost

This uses the Anthropic API which costs money per token. Some rough numbers:

- Simple question: ~$0.01
- Read and summarize a file: ~$0.02-0.05
- Edit a file with read+edit: ~$0.03-0.08
- Web search + fetch: ~$0.05-0.15
- Subagent exploration: ~$0.10-0.30

Prompt caching helps a lot on multi-turn sessions. After the first turn, the system prompt and tool definitions are cached (90% cheaper for subsequent turns).

Use `/usage` to toggle per-turn token stats and `/tokens` to check the current conversation size.

## Known issues

- There's a brief "stuck" feeling when the model generates a `tool_use` block with no preceding text (the terminal shows nothing while tool input JSON streams silently)
- Subagents sometimes over-explore if the task description isn't specific enough
- The model occasionally uses markdown despite being told not to

## License

MIT
