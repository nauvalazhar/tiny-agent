#!/usr/bin/env bun

import Anthropic from "@anthropic-ai/sdk";
import type { Stream } from "@anthropic-ai/sdk/streaming";
import { input, select } from "@inquirer/prompts";
import fg from "fast-glob";
import fs from "fs/promises";
import ora from "ora";
import { platform } from "os";
import path from "path";
import pc from "picocolors";
import { RE2 } from "re2-wasm";
import z from "zod";
import TurndownService from "turndown";
import { LRUCache } from "lru-cache";

const client = new Anthropic();
const turndown = new TurndownService();

/**
 * Constants
 */

const MAX_TURNS = 20;
const MAX_MATCHES = 50;
const MAX_LINE_LENGTH = 500;
const MAX_RESULT_CHARS = 10_000;
const KEEP_RECENT_RESULTS = 6;
const KEEP_RECENT_MESSAGES = 4;
const COMPACT_THRESHOLD_TOKENS = 8_000;
const MAX_CONTENT_LENGTH = 100_000;
const WEB_CACHE_TTL_MS = 15 * 60 * 1000; // 15 min
const WEB_CACHE_MAX_ENTRIES = 100;
const WEB_CACHE_MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50 mb

const OUTPUT_PROMPT = `# Output format

  You are running in a terminal that does not render markdown. Do not use markdown formatting:
  - No **bold** or *italic*
  - No # headers
  - No \`\`\`code blocks (just indent code with 2 spaces)
  - No tables
  - Use plain text with simple indentation and dashes for lists.`;

const SYSTEM_PROMPT = `You are a coding assistant that helps users with software engineering tasks. You have access to tools for reading, searching,
  editing files, and running commands.

  # How to work

  - When given a task, start by understanding the codebase. Use list_files and search_files to find relevant files before making changes.
  - Always read a file before editing it. Never guess what is in a file.
  - Use search_files to find code patterns, function definitions, and usage across the project.
  - When you find multiple candidate files, read them to understand which one is relevant.

  # How to edit files

  - Use the edit_file tool for modifications. Provide enough context in old_string to make the match unique.
  - For new files, use write_file.
  - After making edits, verify your changes make sense in context.

  # How to handle ambiguity

  - If the user's request is ambiguous, ask which one they mean.
  - If you are unsure about the right approach, explain your plan before making changes.

  # Communication

  - Be concise. Show what you changed, not everything you considered.
  - When reporting edits, mention the file path and what was changed.
  
  ${OUTPUT_PROMPT}`;

const COMPACT_SYSTEM_PROMPT = `Summarize this conversation between a user and a coding assistant.
  Preserve: file paths, code changes made, current task, and user preferences.
  Be concise but do not lose important details. 
  
  ${OUTPUT_PROMPT}`;

const SUBAGENT_PROMPTS = {
  explorer: `You are an exploration sub-agent. Your job is to investigate ONLY what the task asks for.

  # Rules
  - Stay focused on the exact scope in the task. Do not explore beyond it.
  - If the task mentions a specific directory or file, only investigate that.
  - Use the minimum tool calls needed. Be efficient.
  - When you have enough information, return a concise summary and STOP.
  - Do not modify any files. Do not run shell commands.

  # Tools available
  - read_file, glob, grep — that's it.
  
  ${OUTPUT_PROMPT}`,

  planner: `You are a planning sub-agent. Your job is to analyze a task and produce a clear, step-by-step implementation plan.
    Use read_file, glob, and grep to understand the codebase first if needed.
    Return a numbered list of concrete steps. Do not implement anything.
    
    ${OUTPUT_PROMPT}`,
} as const;

const PLAN_MODE_INSTRUCTION = `
  # PLAN MODE ACTIVE

  You are currently in plan mode. The user wants you to PLAN, not execute.

  - DO NOT edit files. DO NOT run shell commands. DO NOT call write_file or edit_file.
  - You MAY use read_file, glob, grep to investigate the codebase.
  - Once you understand the task, write a clear plan in your response with numbered steps.
  - When the plan is ready, call the exit_plan_mode tool to show it to the user.
  `;

/**
 * Type definitions
 */

type PermissionDecision = "allow" | "ask" | "deny";

type Tool<T extends z.ZodObject<any> = z.ZodObject<any>> = {
  name: string;
  description: string;
  inputSchema: T;
  isInteractive?: boolean;
  call(input: z.infer<T>, session: Session): Promise<string>;
  checkPermissions(input: z.infer<T>): PermissionDecision;
};

type Session = {
  allowedTools: Set<string>;
  readTimestamps: Map<string, number>;
  spinner: ReturnType<typeof ora>;
  planMode: boolean;
};

type AgentLoopConfig = {
  session: Session;
  conversation: Anthropic.MessageParam[];
  tools: Tool[];
  systemPrompt: string;
  label: string;
  maxTurns?: number;
  silent?: boolean;
};

/**
 * In-memory store
 */

const webFetchCache = new LRUCache<string, string>({
  max: WEB_CACHE_MAX_ENTRIES,
  maxSize: WEB_CACHE_MAX_SIZE_BYTES,
  sizeCalculation: (value) => value.length,
  ttl: WEB_CACHE_TTL_MS,
});

function createSession(): Session {
  return {
    allowedTools: new Set(),
    readTimestamps: new Map(),
    spinner: ora({ text: "Cooking...", color: "green" }),
    planMode: false,
  };
}

/**
 * Logging helpers (replaces clack's log.*)
 */

const logger = {
  info: (msg: string) => console.log(pc.cyan("ℹ ") + msg),
  success: (msg: string) => console.log(pc.green("✔ ") + msg),
  warn: (msg: string) => console.log(pc.yellow("⚠ ") + msg),
  error: (msg: string) => console.log(pc.red("✖ ") + msg),
  dim: (msg: string) => console.log(pc.dim(msg)),
  user: (msg: string) => console.log(pc.bold(pc.magenta("You: ")) + msg),
  assistant: (msg: string) => console.log(pc.bold(pc.cyan("Agent: ")) + msg),
};

/**
 * Helper functions
 */

function normalizeQuotes(str: string): string {
  return str
    .replaceAll("\u2018", "'")
    .replaceAll("\u2019", "'")
    .replaceAll("\u201C", '"')
    .replaceAll("\u201D", '"');
}

function findNormalizedString(
  fileContent: string,
  searchString: string,
): string | null {
  if (fileContent.includes(searchString)) {
    return searchString;
  }

  const normalizedFile = normalizeQuotes(fileContent);
  const normalizedSearch = normalizeQuotes(searchString);

  const index = normalizedFile.indexOf(normalizedSearch);

  if (index !== -1) {
    return fileContent.substring(index, index + searchString.length);
  }

  return null;
}

function truncateResult(result: string): string {
  if (result.length <= MAX_RESULT_CHARS) {
    return result;
  }

  return (
    result.slice(0, MAX_RESULT_CHARS) +
    `\n... [truncated, total ${result.length} chars, showing first ${MAX_RESULT_CHARS} chars]`
  );
}

function estimateTokens(conversation: Anthropic.MessageParam[]): number {
  let chars = 0;

  for (const message of conversation) {
    if (typeof message.content === "string") {
      chars += message.content.length;
    } else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if ("text" in block && typeof block.text === "string") {
          chars += block.text.length;
        } else if ("content" in block && typeof block.content === "string") {
          chars += block.content.length;
        } else if ("input" in block) {
          chars += JSON.stringify(block.input).length;
        }
      }
    }
  }

  return Math.ceil(chars / 4);
}

function clearOldToolResults(
  conversation: Anthropic.MessageParam[],
): Anthropic.MessageParam[] {
  const toolResultIndices: number[] = [];

  conversation.forEach((message, index) => {
    if (
      typeof message.content !== "string" &&
      Array.isArray(message.content) &&
      message.content.some((block) => block.type === "tool_result")
    ) {
      toolResultIndices.push(index);
    }
  });

  const indexesToClear = new Set(
    toolResultIndices.slice(0, -KEEP_RECENT_RESULTS),
  );

  return conversation.map((message, index) => {
    if (!indexesToClear.has(index)) return message;

    const content = (message.content as Anthropic.ToolResultBlockParam[]).map(
      (block) => {
        if (block.type === "tool_result") {
          return {
            ...block,
            content: "[cleared to save context]",
          };
        }
        return block;
      },
    );

    return {
      ...message,
      content,
    };
  });
}

async function compactConversation(
  conversation: Anthropic.MessageParam[],
  tokenEstimation: number,
): Promise<{
  conversation: Anthropic.MessageParam[];
  wasCompacted: boolean;
}> {
  if (tokenEstimation < COMPACT_THRESHOLD_TOKENS) {
    return {
      conversation,
      wasCompacted: false,
    };
  }

  const summaryResponse = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system: COMPACT_SYSTEM_PROMPT,
    messages: conversation,
  });

  const summaryText = summaryResponse.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n");

  const recentMessages = conversation.slice(-KEEP_RECENT_MESSAGES);

  const compacted: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: `[conversation summary]\n${summaryText}\n\n[continuing from here]\n`,
    },
    {
      role: "assistant",
      content: "Understood. I have the context from our previous conversation.",
    },
    ...recentMessages,
  ];

  return {
    conversation: compacted,
    wasCompacted: true,
  };
}

function getDecision<T extends z.ZodObject<any>>(
  tool: Tool<T>,
  input: z.infer<T>,
  session: Session,
): PermissionDecision {
  if (session.planMode) {
    const blockedTools = ["edit_file", "write_file", "shell"];
    if (blockedTools.includes(tool.name)) {
      return "deny";
    }
  }

  if (session.allowedTools.has(tool.name)) {
    return "allow";
  }

  return tool.checkPermissions(input);
}

async function askUserPermission(
  toolName: string,
  input: Record<string, unknown>,
  session: Session,
): Promise<boolean> {
  const inputStr = JSON.stringify(input, null, 2).slice(0, 200);

  console.log();
  console.log(pc.yellow("⚠ Permission required"));
  console.log(pc.dim(`  Tool: ${toolName}`));
  console.log(pc.dim(`  Input: ${inputStr}`));
  console.log();

  const choice = await select({
    message: "Allow this tool to run?",
    choices: [
      { name: "Yes", value: "yes" },
      { name: "No", value: "no" },
      { name: `Always allow ${toolName}`, value: "always" },
    ],
  });

  if (choice === "always") {
    console.log(pc.dim(`  → auto-allowing ${toolName} from now on`));

    session.allowedTools.add(toolName);
    return true;
  }

  return choice === "yes";
}

/**
 * Tool definitions and implementations
 */

const createTool = <T extends z.ZodObject<any>>(tool: Tool<T>) => tool;

const getTimeTool: Tool = createTool({
  name: "get_time",
  description: "Get the current time",
  inputSchema: z.object({}),
  async call() {
    return new Date().toString();
  },
  checkPermissions() {
    return "allow";
  },
});

const readFileTool: Tool = createTool({
  name: "read_file",
  description: "Read the contents of a file with line numbers.",
  inputSchema: z.object({
    file_path: z.string().describe("The path to the file to read"),
  }),
  async call(input, session) {
    const { file_path } = input;
    try {
      const fileContents = await fs.readFile(file_path, "utf-8");
      session.readTimestamps.set(path.resolve(file_path), Date.now());

      return fileContents
        .split("\n")
        .map((line, i) => `${i + 1}\t${line}`)
        .join("\n");
    } catch (error) {
      return `Error reading file: ${(error as Error).message}`;
    }
  },
  checkPermissions() {
    return "allow";
  },
});

const editFileTool: Tool = createTool({
  name: "edit_file",
  description:
    "Edit a file by replacing old_string with new_string. " +
    "The old_string must appear exactly once in the file (unless replace_all is true). " +
    "Include enough surrounding context to make the match unique.",
  inputSchema: z.object({
    file_path: z.string().describe("The path to the file to edit"),
    old_string: z.string().describe("The exact string to find"),
    new_string: z
      .string()
      .describe(
        "The new string to replace the old string with (must be different from old_string)",
      ),
    replace_all: z
      .boolean()
      .optional()
      .describe("Replace all occurrences. Defaults to false."),
  }),
  async call(input, session) {
    const { file_path, old_string, new_string, replace_all = false } = input;

    if (old_string === new_string) {
      return "Error: old_string and new_string must be different.";
    }

    if (!(await fs.exists(file_path))) {
      return `Error: File not found: ${file_path}`;
    }

    const content = await fs.readFile(file_path, "utf-8");
    const normalizedOldString = findNormalizedString(content, old_string);

    if (!normalizedOldString) {
      return `Error: old_string not found in file: ${old_string}`;
    }

    if (!replace_all) {
      const count = content.split(normalizedOldString).length - 1;

      if (count > 1) {
        return `Error: old_string appears ${count} times in the file. Include more surrounding context to make it unique, or set replace_all to true.`;
      }
    }

    const resolvedFilePath = path.resolve(file_path);
    const lastReadTimestamp = session.readTimestamps.get(resolvedFilePath);

    if (lastReadTimestamp) {
      try {
        const stat = await fs.stat(file_path);

        if (stat.mtimeMs > lastReadTimestamp) {
          return `Error: The file has been modified since it was last read. Please read the file again to get the latest contents before editing.`;
        }
      } catch {
        return `Error: Unable to access file metadata for: ${file_path}`;
      }
    }

    const updated = replace_all
      ? content.split(normalizedOldString).join(new_string)
      : content.replace(normalizedOldString, new_string);
    await fs.writeFile(file_path, updated, "utf-8");

    session.readTimestamps.set(resolvedFilePath, Date.now());

    return `File edited: ${file_path}`;
  },
  checkPermissions() {
    return "ask";
  },
});

const writeFileTool: Tool = createTool({
  name: "write_file",
  description:
    "Write contents to a file. Create parent directories if needed. Overwrites existing files.",
  inputSchema: z.object({
    file_path: z.string().describe("The path to the file to write"),
    contents: z.string().describe("The contents to write to the file"),
  }),
  async call(input) {
    const { file_path, contents } = input;

    try {
      await fs.mkdir(path.dirname(file_path), { recursive: true });
      await fs.writeFile(file_path, contents, "utf-8");
      return `File written: ${file_path}`;
    } catch (error) {
      return `Error writing file: ${(error as Error).message}`;
    }
  },
  checkPermissions() {
    return "ask";
  },
});

const globTool: Tool = createTool({
  name: "glob",
  description: "List files in a directory.",
  inputSchema: z.object({
    directory: z
      .string()
      .optional()
      .describe("The directory to list. Defaults to current directory."),
    pattern: z
      .string()
      .optional()
      .describe("The glob pattern to match files. Defaults to '**/*'"),
  }),
  async call(input) {
    const { directory = process.cwd(), pattern = "**/*" } = input;

    const searchDir = path.isAbsolute(directory)
      ? directory
      : path.resolve(directory);
    const stat = await fs.stat(searchDir);

    if (!stat.isDirectory()) {
      return `Directory not found: ${directory}`;
    }

    try {
      const entries = await fg(pattern, {
        cwd: searchDir,
        absolute: true,
        dot: true,
        onlyFiles: true,
        stats: true,
        followSymbolicLinks: false,
      });

      entries.sort((a, b) => {
        const aTime = a.stats?.mtimeMs ?? 0;
        const bTime = b.stats?.mtimeMs ?? 0;
        return bTime - aTime;
      });

      const allFiles = entries.map((entry) => entry.path);

      return `Found ${allFiles.length} files:\n` + allFiles.join("\n");
    } catch (error) {
      return `Error listing files: ${(error as Error).message}`;
    }
  },
  checkPermissions() {
    return "allow";
  },
});

const grepTool: Tool = createTool({
  name: "grep",
  description: "Search for a pattern in a file.",
  inputSchema: z.object({
    directory: z
      .string()
      .optional()
      .describe("The directory to search. Defaults to current directory."),
    pattern: z.string().describe("The pattern to search for"),
  }),
  async call(input) {
    const { directory = process.cwd(), pattern } = input;

    const re = new RE2(pattern, "u");

    const searchDir = path.isAbsolute(directory)
      ? directory
      : path.resolve(directory);

    const files = await fg("**/*", {
      cwd: searchDir,
      absolute: true,
      dot: true,
      onlyFiles: true,
      followSymbolicLinks: false,
      // you can grow the list
      ignore: [
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.gif",
        "**/*.bmp",
        "**/*.pdf",
        "**/*.zip",
        "**/*.tar",
        "**/*.gz",
        "**/*.7z",
        "**/*.exe",
        "**/*.bin",
      ],
    });

    const matches: { file: string; line: number; content: string }[] = [];

    for (const file of files) {
      try {
        const contents = await fs.readFile(file, "utf-8");
        const lines = contents.split("\n");

        lines.forEach((line, index) => {
          if (re.test(line)) {
            matches.push({
              file,
              line: index + 1,
              content: line.trim().slice(0, MAX_LINE_LENGTH),
            });
          }
        });
      } catch {
        continue;
      }
    }

    if (matches.length === 0) {
      return `No matches found for pattern: ${pattern}`;
    }

    return (
      matches
        .slice(0, MAX_MATCHES)
        .map((match) => `${match.file}:${match.line}:${match.content}`)
        .join("\n") +
      (matches.length > MAX_MATCHES
        ? `\n...and ${matches.length - MAX_MATCHES} more matches`
        : "")
    );
  },
  checkPermissions() {
    return "allow";
  },
});

const shellTool: Tool = createTool({
  name: "shell",
  description: "Execute a shell command and return the output.",
  inputSchema: z.object({
    command: z.string().describe("The shell command to execute"),
  }),
  async call(input) {
    const { command } = input;

    try {
      const isWindows = platform() === "win32";
      const shell = isWindows ? ["cmd", "/c"] : ["sh", "-c"];

      const proc = Bun.spawn([...shell, command], {
        stdout: "pipe",
        stderr: "pipe",
      });

      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);

      const exitCode = await proc.exited;

      if (exitCode !== 0) {
        return `Command failed (exit ${exitCode}): ${stderr || stdout}`;
      }

      if (stderr) {
        return `Command error output: ${stderr}`;
      }

      return stdout;
    } catch (error) {
      return `Error executing command: ${(error as Error).message}`;
    }
  },
  checkPermissions(input) {
    const { command } = input;

    if (/^(ls|cat|echo|pwd|git\s+(status|diff|log))/.test(command)) {
      return "allow" as const;
    }

    return "ask";
  },
});

const spawnAgentTool: Tool = createTool({
  name: "spawn_agent",
  description:
    "Spawn a sub-agent to handle a focused task in isolation. " +
    "The sub-agent has NO context about the conversation, so be specific in the task description. " +
    "Include exact file paths, directory names, and clear scope. " +
    "Use this for tasks that require many tool calls so the main conversation stays clean.",
  inputSchema: z.object({
    task: z
      .string()
      .describe("A clear, self-contained task description for sub-agent"),
    type: z
      .enum(["explorer", "planner"])
      .describe(
        "explorer: investigates code and returns findings. planner: analyzes a task and returns a step-by-step plan.",
      ),
  }),
  async call(input, session) {
    const { task, type } = input;

    const subTools = tools.filter((t) =>
      ["read_file", "glob", "grep", "get_time"].includes(t.name),
    );

    console.log();
    logger.info(`Spawning ${type} sub-agent ...`);
    console.log();

    const result = await agentLoop({
      // we share session between parent and sub
      session,
      conversation: [{ role: "user", content: task }],
      tools: subTools,
      systemPrompt: SUBAGENT_PROMPTS[type],
      label: `Sub-agent (${type})`,
      maxTurns: 20,
      silent: true,
    });

    console.log();
    logger.info(`Sub-agent finished`);
    console.log();

    return result;
  },
  checkPermissions() {
    return "allow";
  },
});

const exitPlanModeTool: Tool = createTool({
  name: "exit_plan_mode",
  description:
    "Call this when you have finished writing your plan. Shows the plan to the user " +
    "and asks if they want to proceed with implementation.",
  inputSchema: z.object({
    plan: z
      .string()
      .describe("The full plan, formatted clearly with numbered steps"),
  }),
  isInteractive: true,
  async call(input, session) {
    const { plan } = input;

    console.log();
    console.log(pc.bold(pc.cyan("📋 Plan")));
    console.log(pc.dim("─".repeat(50)));
    console.log(plan);
    console.log(pc.dim("─".repeat(50)));
    console.log();

    const choice = await select({
      message: "Proceed with implementation?",
      choices: [
        { name: "Yes, implement the plan", value: "yes" },
        { name: "No, keep planning", value: "no" },
        { name: "Cancel", value: "cancel" },
      ],
    });

    if (choice === "yes") {
      session.planMode = false;
      return "User approved the plan. Plan mode exited. You may now implement the steps.";
    }

    if (choice === "cancel") {
      session.planMode = false;
      return "User cancelled. Stop and wait for further instructions.";
    }

    return "User wants to keep planning. Refine the plan based on their feedback.";
  },
  checkPermissions() {
    return "allow";
  },
});

const webSearchTool: Tool = createTool({
  name: "web_search",
  description:
    "Search the web for information. Returns a list of results with titles, " +
    "URLs, and snippets. Use for current events, documentation, etc.",
  inputSchema: z.object({
    query: z.string().describe("The search query"),
  }),
  async call(input) {
    const { query } = input;

    try {
      const response = await client.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 2048,
        tools: [
          {
            type: "web_search_20250305",
            name: "web_search",
            max_uses: 8,
          } as any,
        ],
        tool_choice: { type: "tool", name: "web_search" },
        messages: [
          {
            role: "user",
            content: `Perform a web search for: ${query}`,
          },
        ],
      });

      const results: string[] = [];
      for (const block of response.content) {
        if (block.type === "text") {
          results.push(block.text);
        }
      }

      return results.join("\n\n") || "No results found.";
    } catch (error) {
      return `Web search failed: ${(error as Error).message}`;
    }
  },
  checkPermissions() {
    return "ask";
  },
});

const webFetchTool: Tool = createTool({
  name: "web_fetch",
  description:
    "Fetch a URL and extract specific information from it. " +
    "Provide a URL and a prompt describing what you want to know from the page. " +
    "A small model reads the page and returns only the relevant info.",
  inputSchema: z.object({
    url: z.string().describe("The URL to fetch"),
    prompt: z
      .string()
      .describe(
        "What you want to know or extract from the page (e.g. 'the installation instructions')",
      ),
  }),
  async call(input) {
    const { url, prompt } = input;
    const cacheKey = `${url}::${prompt}`;

    const cached = webFetchCache.get(cacheKey);

    if (cached) {
      return cached;
    }

    try {
      const parsed = new URL(url);

      if (!["http:", "https:"].includes(parsed.protocol)) {
        return "Error: Only http:// and https:// URLs are supported.";
      }

      const response = await fetch(url, {
        headers: { "User-Agent": "tiny-agent/0.1" },
        redirect: "follow",
      });

      if (!response.ok) {
        return `Error: HTTP ${response.status} ${response.statusText}`;
      }

      const contentType = response.headers.get("content-type") ?? "";
      const raw = await response.text();

      let content: string;
      if (contentType.includes("text/html")) {
        content = turndown.turndown(raw);
      } else {
        content = raw;
      }

      if (content.length > MAX_CONTENT_LENGTH) {
        content = content.slice(0, MAX_CONTENT_LENGTH) + "\n\n[truncated]";
      }

      const modelResponse = await client.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 2048,
        messages: [
          {
            role: "user",
            content: `Here is the content of ${url}:\n\n${content}\n\n---\n\nBased on the content above, ${prompt}`,
          },
        ],
      });

      const text = modelResponse.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n");

      if (text) {
        webFetchCache.set(cacheKey, text);
      }

      return text || "No response from model";
    } catch (error) {
      return `Error fetching URL: ${(error as Error).message}`;
    }
  },
  checkPermissions() {
    return "ask";
  },
});

/**
 * Tool registry
 */

const tools: Tool[] = [
  getTimeTool,
  readFileTool,
  editFileTool,
  writeFileTool,
  globTool,
  grepTool,
  shellTool,
  spawnAgentTool,
  exitPlanModeTool,
  webSearchTool,
  webFetchTool,
];

/**
 * Streaming response handling
 */

async function streamResponse(
  response: Stream<Anthropic.MessageStreamEvent>,
  silent = false,
): Promise<Anthropic.ContentBlockParam[]> {
  const contentBlocks: Anthropic.ContentBlockParam[] = [];

  let currentBlockType: string | null = null;
  let currentToolInput = "";
  let currentToolId = "";
  let currentToolName = "";
  let currentTextIndex = -1;
  let didWrite = false;

  for await (const event of response) {
    switch (event.type) {
      case "content_block_start":
        if (event.content_block.type === "text") {
          currentBlockType = "text";
          currentTextIndex = contentBlocks.length;
          contentBlocks.push({ type: "text", text: "" });
        } else if (event.content_block.type === "tool_use") {
          currentBlockType = "tool_use";
          currentToolId = event.content_block.id;
          currentToolName = event.content_block.name;
          currentToolInput = "";
        }
        break;

      case "content_block_delta":
        if (event.delta.type === "text_delta") {
          const block = contentBlocks[currentTextIndex];
          if (block && block.type === "text") {
            block.text += event.delta.text;
          }
          // stream the text directly
          if (!silent) {
            process.stdout.write(event.delta.text);
            didWrite = true;
          }
        } else if (event.delta.type === "input_json_delta") {
          currentToolInput += event.delta.partial_json;
        }
        break;

      case "content_block_stop":
        if (currentBlockType === "tool_use") {
          try {
            contentBlocks.push({
              type: "tool_use",
              id: currentToolId,
              name: currentToolName,
              input: currentToolInput ? JSON.parse(currentToolInput) : {},
            });
          } catch {
            contentBlocks.push({
              type: "tool_use",
              id: currentToolId,
              name: currentToolName,
              input: {},
            });
          }
          currentToolInput = "";
        }
        currentBlockType = null;
        break;
    }
  }

  if (didWrite) process.stdout.write("\n");

  return contentBlocks;
}

/**
 * Main agent loop
 */

async function agentLoop(config: AgentLoopConfig): Promise<string> {
  const {
    conversation,
    session,
    systemPrompt,
    maxTurns = MAX_TURNS,
    tools: availableTools,
    label,
    silent,
  } = config;

  const spinner = session.spinner;
  let turns = 0;

  const toolsJson: Anthropic.Tool[] = availableTools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: z.toJSONSchema(
      tool.inputSchema,
    ) as Anthropic.Tool["input_schema"],
  }));

  while (true) {
    if (turns > maxTurns) {
      spinner.fail("Max turns exceeded");
      return "Max turns exceeded";
    }

    if (!silent) spinner.start("Cooking...");

    let compressed = clearOldToolResults(conversation);
    const tokenEstimation = estimateTokens(compressed);
    const { conversation: compactedConversation, wasCompacted } =
      await compactConversation(compressed, tokenEstimation);

    if (wasCompacted) {
      if (!silent) spinner.stop();
      logger.info(
        `Conversation compacted to save context (estimated tokens: ${tokenEstimation})`,
      );
      if (!silent) spinner.start("Cooking...");
    }

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      messages: compactedConversation,
      tools: toolsJson,
      system: systemPrompt,
      stream: true,
    });

    // Print agent label, then stream tokens directly under it
    if (!silent) {
      spinner.stop();
      process.stdout.write(pc.bold(pc.cyan(`${label}: `)));
    }

    const contentBlocks: Anthropic.ContentBlockParam[] = await streamResponse(
      response,
      silent,
    );

    conversation.push({
      role: "assistant",
      content: contentBlocks,
    });

    const toolUseBlocks = contentBlocks.filter(
      (block) => block.type === "tool_use",
    );

    if (toolUseBlocks.length === 0) {
      return contentBlocks
        .filter((block) => block.type === "text")
        .map((b) => b.text)
        .join("\n");
    }

    // Execute tools
    const toolResults: Anthropic.ToolResultBlockParam[] = [];

    for (const toolUseBlock of toolUseBlocks) {
      const tool = tools.find((t) => t.name === toolUseBlock.name);

      if (!tool) {
        toolResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: `Tool not found: ${toolUseBlock.name}`,
          is_error: true,
        });
        logger.error("Tool not found: " + toolUseBlock.name);
        continue;
      }

      const toolInputParse = tool.inputSchema.safeParse(toolUseBlock.input);

      if (!toolInputParse.success) {
        toolResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: `Invalid input for tool ${tool.name}: ${toolInputParse.error.message}`,
          is_error: true,
        });
        logger.error(
          `Invalid input for ${toolUseBlock.name}: ${toolInputParse.error.message}`,
        );
        continue;
      }

      const decision = getDecision(tool, toolInputParse.data, session);

      if (decision === "deny") {
        toolResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: "Permission denied.",
          is_error: true,
        });
        continue;
      }

      if (decision === "ask") {
        const allowed = await askUserPermission(
          tool.name,
          toolInputParse.data,
          session,
        );

        if (!allowed) {
          toolResults.push({
            type: "tool_result",
            tool_use_id: toolUseBlock.id,
            content: "Permission denied by user",
            is_error: true,
          });
          continue;
        }
      }

      if (!silent && !tool.isInteractive)
        spinner.start(`Running ${tool.name}...`);

      try {
        const toolResultRaw = await tool.call(toolInputParse.data, session);
        const toolResult = truncateResult(toolResultRaw);

        toolResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: toolResult,
        });

        if (!silent) {
          spinner.stop();
          logger.dim(
            `  → ${tool.name}:\n${toolResult.slice(0, 100)}${toolResult.length > 100 ? "..." : ""}`,
          );
        }
      } catch (error) {
        if (!silent) spinner.stop();
        toolResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: `Error executing tool ${tool.name}: ${(error as Error).message}`,
          is_error: true,
        });
        logger.error(
          `Error executing ${toolUseBlock.name}: ${(error as Error).message}`,
        );
        continue;
      }
    }

    conversation.push({
      role: "user",
      content: toolResults,
    });

    turns++;
  }
}

/**
 * Main entry
 */

async function main() {
  const session = createSession();
  const conversation: Anthropic.MessageParam[] = [];

  console.log();
  console.log(pc.bold(pc.cyan("Tiny Agent")));
  console.log(pc.dim("Type your message, or press Ctrl+C to exit"));
  console.log();

  while (true) {
    let message: string;
    const promptLabel = session.planMode ? pc.yellow("You (plan mode)") : "You";

    try {
      message = await input({
        message: promptLabel,
        theme: {
          style: {
            answer: (text: string) => pc.white(text),
          },
        },
        validate: (value) =>
          value.trim().length > 0 || "Please enter a message",
      });
    } catch {
      // Ctrl+C
      console.log();
      logger.dim("Goodbye!");
      process.exit(0);
    }

    if (message.startsWith("/")) {
      const cmd = message.trim();
      switch (cmd) {
        case "/plan":
          session.planMode = !session.planMode;
          logger.info(
            session.planMode
              ? "Plan mode enabled. Tell me what to plan. To exit, run /plan again."
              : "Plan mode disabled.",
          );
          continue; // skip agent loop, go back to prompt
        case "/clear":
          conversation.length = 0;
          console.clear();
          logger.dim("Conversation cleared.");
          continue;
        default:
          logger.warn(`Unknown command: ${cmd}`);
          continue;
      }
    }

    conversation.push({ role: "user", content: message });

    try {
      const finalSystemPrompt = session.planMode
        ? `${SYSTEM_PROMPT}\n\n${PLAN_MODE_INSTRUCTION}`
        : SYSTEM_PROMPT;

      await agentLoop({
        session,
        conversation,
        tools,
        systemPrompt: finalSystemPrompt,
        label: "Agent",
      });
      console.log();
    } catch (error) {
      logger.error("An error occurred: " + (error as Error).message);
      console.error(error);
    }
  }
}

main();

/**
 * Missing
 * [ ] Paralel tool calls
 * [x] Subagents
 * [x] Web search and fetch
 * [ ] Persistent and clear session
 * [x] Plan mode
 */
