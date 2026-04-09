#!/usr/bin/env bun

import Anthropic from "@anthropic-ai/sdk";
import type { Stream } from "@anthropic-ai/sdk/streaming";
import { input, select } from "@inquirer/prompts";
import { randomUUID } from "crypto";
import fg from "fast-glob";
import { createReadStream } from "fs";
import fs from "fs/promises";
import { LRUCache } from "lru-cache";
import ora from "ora";
import { homedir, platform } from "os";
import path from "path";
import pc from "picocolors";
import { RE2 } from "re2-wasm";
import { createInterface } from "readline";
import TurndownService from "turndown";
import z from "zod";

const turndown = new TurndownService();

/**
 * Constants
 */

const MAX_TURNS = 20;
const MAX_MATCHES = 50;
const MAX_LINE_LENGTH = 500;
const MAX_RESULT_CHARS = 4_000;
const KEEP_RECENT_RESULTS = 6;
const KEEP_RECENT_MESSAGES = 4;
const COMPACT_THRESHOLD_TOKENS = 6_000;
const MAX_CONTENT_LENGTH = 100_000;
const WEB_CACHE_TTL_MS = 15 * 60 * 1000; // 15 min
const WEB_CACHE_MAX_ENTRIES = 100;
const WEB_CACHE_MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50 mb
const CONFIG_DIR = path.join(homedir(), ".tiny-agent");
const PROJECTS_DIR = path.join(CONFIG_DIR, "projects");
const CONFIG_FILE = path.join(CONFIG_DIR, "config.json");

const OUTPUT_PROMPT = `# Output

  You are running in a terminal that does not render markdown:
  - Be terse. Skip preamble and postamble.
  - Don't restate the task.
  - No markdown.
  - If the tool result answers the question, just summarize it. Don't narrate every step.`;

const SYSTEM_PROMPT = `You are a coding assistant. Use tools to read, search, edit files and run commands.

  Rules:
  - Read files before editing them
  - Include enough context in old_string to make it unique
  - Batch independent tool calls in parallel
  - Be concise. Don't explain what you just did.

  # Using web_search
  - When searching for current events, sports standings, news, or anything time-sensitive, include the current year in your query.
  
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

type ClassifiedError = {
  kind:
    | "auth"
    | "permission"
    | "rate_limit"
    | "overloaded"
    | "context_length"
    | "network"
    | "unknown";
  message: string;
};

type Config = {
  apiKey?: string;
};

type PermissionDecision = "allow" | "ask" | "deny";

type Tool<T extends z.ZodObject<any> = z.ZodObject<any>> = {
  name: string;
  description: string;
  inputSchema: T;
  isInteractive?: boolean;
  isConcurrencySafe?: boolean;
  call(input: z.infer<T>, session: Session): Promise<string>;
  checkPermissions(input: z.infer<T>): PermissionDecision;
};

type ApprovedCall = {
  block: Anthropic.ToolUseBlockParam;
  tool: Tool;
  parsedInput: any;
};

type Session = {
  allowedTools: Set<string>;
  readTimestamps: Map<string, number>;
  spinner: ReturnType<typeof ora>;
  planMode: boolean;
  sessionId: string;
  sessionPath: string;
  client: Anthropic;
  showUsage: boolean;
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

type StreamResult = {
  contentBlocks: Anthropic.ContentBlockParam[];
  usage: Anthropic.Usage | null;
};

type TranscriptEntry = {
  type: "user" | "assistant";
  timestamp: string;
  cwd: string;
  message: Anthropic.MessageParam;
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

function createSession(apiKey: string): Session {
  const sessionId = randomUUID();
  const sessionPath = path.join(getProjectDir(), `${sessionId}.jsonl`);

  return {
    allowedTools: new Set(),
    readTimestamps: new Map(),
    spinner: ora({ text: "Cooking...", color: "green" }),
    planMode: false,
    sessionId,
    sessionPath,
    client: new Anthropic({ apiKey }),
    showUsage: true,
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

function classifyError(error: any): ClassifiedError {
  if (error?.status === 401) {
    return {
      kind: "auth",
      message: "Invalid API key. Run /login to update it.",
    };
  }
  if (error?.status === 403) {
    return {
      kind: "permission",
      message: "API key lacks permission for this model.",
    };
  }
  if (error?.status === 429) {
    return {
      kind: "rate_limit",
      message: "Rate limited. Please wait a moment and try again.",
    };
  }
  if (error?.status === 529) {
    return {
      kind: "overloaded",
      message: "Anthropic API is overloaded. Try again shortly.",
    };
  }
  if (
    error?.status === 400 &&
    typeof error?.message === "string" &&
    (error.message.includes("prompt is too long") ||
      error.message.includes("context_length"))
  ) {
    return {
      kind: "context_length",
      message:
        "Conversation is too long. Run /clear to start fresh, or /tokens to check size.",
    };
  }
  if (
    error?.code === "ENOTFOUND" ||
    error?.code === "ECONNREFUSED" ||
    error?.code === "ETIMEDOUT" ||
    error?.code === "ECONNRESET"
  ) {
    return {
      kind: "network",
      message: "Network error. Check your internet connection.",
    };
  }
  return {
    kind: "unknown",
    message: error?.message ?? String(error),
  };
}

async function loadConfig(): Promise<Config> {
  try {
    const content = await fs.readFile(CONFIG_FILE, "utf-8");
    return JSON.parse(content);
  } catch {
    return {};
  }
}

async function saveConfig(config: Config): Promise<void> {
  await fs.mkdir(path.dirname(CONFIG_FILE), { recursive: true });
  await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2), {
    mode: 0o600,
  });
}

async function resolveApiKey(): Promise<string | undefined> {
  if (process.env.ANTHROPIC_API_KEY) {
    return process.env.ANTHROPIC_API_KEY;
  }

  const config = await loadConfig();

  return config.apiKey;
}

async function validateApiKey(
  apiKey: string,
): Promise<{ valid: boolean; reason?: string }> {
  try {
    const client = new Anthropic({ apiKey });
    await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 1,
      messages: [{ role: "user", content: "hi" }],
    });
    return { valid: true };
  } catch (error: any) {
    const classifiedError = classifyError(error);

    if (classifiedError.kind === "rate_limit") {
      return {
        valid: false,
        reason: "Rate limited (key is valid but throttled)",
      };
    }

    return { valid: false, reason: classifiedError.message };
  }
}

function resolveSafePath(inputPath: string): string | null {
  const expanded = inputPath.startsWith("~")
    ? path.join(homedir(), inputPath.slice(1))
    : inputPath;

  const resolved = path.resolve(expanded);
  const cwd = process.cwd();

  if (resolved !== cwd && !resolved.startsWith(cwd + path.sep)) {
    return null;
  }

  return resolved;
}

function sanitizeCwd(cwd: string): string {
  return cwd.replace(/[^a-zA-Z0-9]/g, "-");
}

function getProjectDir() {
  return path.join(PROJECTS_DIR, sanitizeCwd(process.cwd()));
}

async function appendSessionEntry(
  sessionPath: string,
  entry: TranscriptEntry,
): Promise<void> {
  try {
    await fs.mkdir(path.dirname(sessionPath), { recursive: true, mode: 0o700 });

    const line = JSON.stringify(entry) + "\n";
    await fs.appendFile(sessionPath, line, {
      encoding: "utf-8",
      mode: 0o600,
    });
  } catch (error) {
    logger.warn(`Failed to persist session: ${(error as Error).message}`);
  }
}

async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (error: any) {
      const isRetryable = error?.status === 429 || error?.status === 529;
      if (!isRetryable || i === attempts - 1) throw error;
      await new Promise((r) => setTimeout(r, 1000 * 2 ** i));
    }
  }
  throw new Error("unreachable");
}

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
  session: Session,
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

  try {
    const summaryResponse = await session.client.messages.create({
      model: "claude-haiku-4-5-20251001",
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
        content:
          "Understood. I have the context from our previous conversation.",
      },
      ...recentMessages,
    ];

    return {
      conversation: compacted,
      wasCompacted: true,
    };
  } catch (error) {
    logger.warn(
      `Compaction failed (${(error as Error).message}). Dropping old messages instead.`,
    );

    return {
      conversation: conversation.slice(-KEEP_RECENT_MESSAGES * 2),
      wasCompacted: true,
    };
  }
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

async function getSessionLabel(filePath: string): Promise<string> {
  try {
    const fileStream = createReadStream(filePath, { encoding: "utf-8" });
    const rl = createInterface({ input: fileStream, crlfDelay: Infinity });

    for await (const line of rl) {
      if (!line) continue;

      try {
        const entry = JSON.parse(line) as TranscriptEntry;

        if (
          entry.type === "user" &&
          typeof entry.message.content === "string"
        ) {
          const text = entry.message.content.slice(0, 60);
          rl.close();
          fileStream.destroy();
          return text + (entry.message.content.length > 60 ? "..." : "");
        }
      } catch {
        continue;
      }
    }

    return "(empty)";
  } catch {
    return "(unreadable)";
  }
}

async function listSessions(): Promise<
  {
    id: string;
    path: string;
    mtime: Date;
    label: string;
  }[]
> {
  const dir = getProjectDir();

  try {
    const files = await fs.readdir(dir);
    const sessions = await Promise.all(
      files
        .filter((file) => file.endsWith(".jsonl"))
        .map(async (file) => {
          const filePath = path.join(dir, file);
          const stat = await fs.stat(filePath);
          const label = await getSessionLabel(filePath);

          return {
            id: file.replace(".jsonl", ""),
            path: filePath,
            mtime: stat.mtime,
            label,
          };
        }),
    );

    return sessions.sort((a, b) => b.mtime.getTime() - a.mtime.getTime());
  } catch {
    return [];
  }
}

async function loadSession(
  filePath: string,
): Promise<Anthropic.MessageParam[]> {
  try {
    const content = await fs.readFile(filePath, "utf-8");
    const lines = content.split("\n").filter((line) => line.length > 0);

    const messages: Anthropic.MessageParam[] = [];
    let skipped = 0;

    for (const line of lines) {
      try {
        const entry = JSON.parse(line) as TranscriptEntry;
        messages.push(entry.message);
      } catch {
        skipped++;
      }
    }

    if (skipped > 0) {
      logger.warn(`Skipped ${skipped} malformed entries in session file`);
    }

    return messages;
  } catch (error) {
    logger.error(`Failed to load session: ${(error as Error).message}`);
    return [];
  }
}

function formatUsage(usage: Anthropic.Usage): string {
  const fmt = (n: number): string => {
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
    return n.toString();
  };

  const parts = [
    `${fmt(usage.input_tokens)} in`,
    `${fmt(usage.output_tokens)} out`,
  ];

  if (usage.cache_read_input_tokens && usage.cache_read_input_tokens > 0) {
    parts.push(`${fmt(usage.cache_read_input_tokens)} cached`);
  }

  if (
    usage.cache_creation_input_tokens &&
    usage.cache_creation_input_tokens > 0
  ) {
    parts.push(`${fmt(usage.cache_creation_input_tokens)} cache write`);
  }

  return parts.join(", ");
}

/**
 * Tool definitions and implementations
 */

const createTool = <T extends z.ZodObject<any>>(tool: Tool<T>) => tool;

const getTimeTool: Tool = createTool({
  name: "get_time",
  description: "Get the current time",
  inputSchema: z.object({}),
  isConcurrencySafe: true,
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
  isConcurrencySafe: true,
  async call(input, session) {
    const { file_path } = input;

    const safePath = resolveSafePath(file_path);

    if (!safePath) {
      return `Error: path is outside the project directory: ${file_path}`;
    }

    try {
      const fileContents = await fs.readFile(safePath, "utf-8");
      session.readTimestamps.set(safePath, Date.now());

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
    "Replace old_string with new_string in a file. old_string must be unique unless replace_all is true.",
  inputSchema: z.object({
    file_path: z.string().describe("File path"),
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

    const safePath = resolveSafePath(file_path);
    if (!safePath) {
      return `Error: path is outside the project directory: ${file_path}`;
    }

    if (old_string === new_string) {
      return "Error: old_string and new_string must be different.";
    }

    try {
      const content = await fs.readFile(safePath, "utf-8");
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

      const lastReadTimestamp = session.readTimestamps.get(safePath);
      if (!lastReadTimestamp) {
        return "Error: file must be read before editing. Use read_file first.";
      }

      const stat = await fs.stat(safePath);
      if (stat.mtimeMs > lastReadTimestamp) {
        return "Error: The file has been modified since it was last read. Please read the file again to get the latest contents before editing.";
      }

      const updated = replace_all
        ? content.split(normalizedOldString).join(new_string)
        : content.replace(normalizedOldString, new_string);
      await fs.writeFile(safePath, updated, "utf-8");

      session.readTimestamps.set(safePath, Date.now());

      return `File edited: ${safePath}`;
    } catch (error) {
      return `Error editing file: ${(error as Error).message}`;
    }
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

    const safePath = resolveSafePath(file_path);

    if (!safePath) {
      return `Error: path is outside the project directory: ${file_path}`;
    }

    try {
      await fs.mkdir(path.dirname(safePath), { recursive: true });
      await fs.writeFile(safePath, contents, "utf-8");
      return `File written: ${safePath}`;
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
  isConcurrencySafe: true,
  async call(input) {
    const { directory = process.cwd(), pattern = "**/*" } = input;

    const safePath = resolveSafePath(directory);

    if (!safePath) {
      return `Error: path is outside the project directory: ${directory}`;
    }

    try {
      const stat = await fs.stat(safePath);

      if (!stat.isDirectory()) {
        return `Not a directory: ${safePath}`;
      }

      const entries = await fg(pattern, {
        cwd: safePath,
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
  isConcurrencySafe: true,
  async call(input) {
    const { directory = process.cwd(), pattern } = input;

    const safePath = resolveSafePath(directory);

    if (!safePath) {
      return `Error: path is outside the project directory: ${directory}`;
    }

    try {
      const stat = await fs.stat(safePath);

      if (!stat.isDirectory()) {
        return `Not a directory: ${safePath}`;
      }

      const re = new RE2(pattern, "u");

      const files = await fg("**/*", {
        cwd: safePath,
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
    } catch (error) {
      return `Error searching: ${(error as Error).message}`;
    }
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

    if (/[;&|`$()<>\\]/.test(command)) return "ask";

    const safePatterns = [
      /^ls( -[la1-9]+)?( \.?[\w./-]*)?$/,
      /^pwd$/,
      /^echo .{1,100}$/,
      /^cat [\w./-]+$/,
      /^git (status|diff|log)( [\w./-]+)?$/,
    ];
    return safePatterns.some((p) => p.test(command)) ? "allow" : "ask";
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
  isConcurrencySafe: true,
  async call(input, session) {
    const { query } = input;

    try {
      const response = await session.client.messages.create({
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
        system: `Today's date is ${new Date().toISOString().split("T")[0]}. Use the current year in searches when relevant.`,
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
  isConcurrencySafe: true,
  async call(input, session) {
    const { url, prompt } = input;
    const cacheKey = `${url}::${prompt}`;

    const cached = webFetchCache.get(cacheKey);

    if (cached) {
      return cached;
    }

    try {
      const parsed = new URL(url);
      const hostname = parsed.hostname;

      if (
        hostname === "localhost" ||
        hostname.startsWith("127.") ||
        hostname.startsWith("10.") ||
        hostname.startsWith("192.168.") ||
        /^172\.(1[6-9]|2\d|3[01])\./.test(hostname) ||
        hostname === "169.254.169.254"
      ) {
        return "Error: cannot fetch private or local IPs.";
      }

      if (parsed.username || parsed.password) {
        return "Error: URLs with credentials are not allowed.";
      }

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

      const modelResponse = await session.client.messages.create({
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
): Promise<StreamResult> {
  const contentBlocks: Anthropic.ContentBlockParam[] = [];
  let usage: Anthropic.Usage | null = null;

  let currentBlockType: string | null = null;
  let currentToolInput = "";
  let currentToolId = "";
  let currentToolName = "";
  let currentTextIndex = -1;

  try {
    for await (const event of response) {
      switch (event.type) {
        case "message_start":
          if (event.message.usage) {
            usage = {
              ...event.message.usage,
            };
          }
          break;

        case "message_delta":
          if (event.usage && usage) {
            if (typeof event.usage.output_tokens === "number") {
              usage.output_tokens = event.usage.output_tokens;
            }
            if (typeof event.usage.input_tokens === "number") {
              usage.input_tokens = event.usage.input_tokens;
            }
            if (typeof event.usage.cache_read_input_tokens === "number") {
              usage.cache_read_input_tokens =
                event.usage.cache_read_input_tokens;
            }
            if (typeof event.usage.cache_creation_input_tokens === "number") {
              usage.cache_creation_input_tokens =
                event.usage.cache_creation_input_tokens;
            }
          }
          break;

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

            // you can execute the tool here during mid-stream
            // but it's different story
          }
          currentBlockType = null;
          break;
      }
    }
  } catch (error) {
    if (!silent) {
      process.stdout.write("\n");
      logger.warn(`Stream interrupted: ${(error as Error).message}`);
    }

    return { contentBlocks, usage };
  }

  if (!silent) process.stdout.write("\n");
  return { contentBlocks, usage };
}

/**
 * Tool executor
 */

async function executeToolCall<T extends z.ZodObject<any>>(
  toolUseBlock: Anthropic.ToolUseBlockParam,
  tool: Tool<T>,
  inputParsed: z.infer<T>,
  session: Session,
  silent: boolean,
): Promise<Anthropic.ToolResultBlockParam> {
  try {
    const toolResultRaw = await tool.call(inputParsed, session);
    const toolResult = truncateResult(toolResultRaw);

    if (!silent) {
      logger.dim(
        `  → ${tool.name}:\n${toolResult.slice(0, 100)}${toolResult.length > 100 ? "..." : ""}`,
      );
    }

    return {
      type: "tool_result",
      tool_use_id: toolUseBlock.id,
      content: toolResult,
    };
  } catch (error) {
    if (!silent) {
      logger.error(
        `Error executing ${toolUseBlock.name}: ${(error as Error).message}`,
      );
    }

    return {
      type: "tool_result",
      tool_use_id: toolUseBlock.id,
      content: `Error executing tool ${tool.name}: ${(error as Error).message}`,
      is_error: true,
    };
  }
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
    silent = false,
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
      await compactConversation(compressed, tokenEstimation, session);

    if (wasCompacted) {
      if (!silent) spinner.stop();
      logger.info(
        `Conversation compacted to save context (estimated tokens: ${tokenEstimation})`,
      );
      if (!silent) spinner.start("Cooking...");
    }

    const response = await withRetry(() =>
      session.client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 4096,
        messages: compactedConversation,
        tools: toolsJson.map((tool, i) => ({
          ...tool,
          ...(i === toolsJson.length - 1
            ? { cache_control: { type: "ephemeral" as const } }
            : {}),
        })),
        system: [
          {
            type: "text",
            text: `${systemPrompt}\n\nToday: ${new Date().toISOString().split("T")[0]}. Cwd: ${process.cwd()}.`,
            cache_control: { type: "ephemeral" },
          },
        ],
        stream: true,
      }),
    );

    // Print agent label, then stream tokens directly under it
    if (!silent) {
      spinner.stop();
      process.stdout.write(pc.bold(pc.cyan(`${label}: `)));
    }

    const { contentBlocks, usage } = await streamResponse(response, silent);

    conversation.push({
      role: "assistant",
      content: contentBlocks,
    });

    if (!silent && usage && session.showUsage) {
      console.log();
      logger.dim(`└ ${formatUsage(usage)}`);
    }

    await appendSessionEntry(session.sessionPath, {
      type: "assistant",
      timestamp: new Date().toISOString(),
      cwd: process.cwd(),
      message: {
        role: "assistant",
        content: contentBlocks,
      },
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
    const approvedCalls: ApprovedCall[] = [];
    const deniedResults: Anthropic.ToolResultBlockParam[] = [];

    for (const toolUseBlock of toolUseBlocks) {
      const tool = availableTools.find((t) => t.name === toolUseBlock.name);

      if (!tool) {
        if (!silent) {
          logger.error(`Tool not found: ${toolUseBlock.name}`);
        }

        deniedResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: `Tool not found: ${toolUseBlock.name}`,
          is_error: true,
        });
        continue;
      }

      const toolInputParse = tool.inputSchema.safeParse(toolUseBlock.input);

      if (!toolInputParse.success) {
        if (!silent) {
          logger.error(
            `Invalid input for tool ${toolUseBlock.name}: ${toolInputParse.error.message}`,
          );
        }

        deniedResults.push({
          type: "tool_result",
          tool_use_id: toolUseBlock.id,
          content: `Invalid input for tool ${toolUseBlock.name}: ${toolInputParse.error.message}`,
          is_error: true,
        });
        continue;
      }

      const decision = getDecision(tool, toolInputParse.data, session);

      if (decision === "deny") {
        deniedResults.push({
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
          deniedResults.push({
            type: "tool_result",
            tool_use_id: toolUseBlock.id,
            content: "Permission denied by user",
            is_error: true,
          });
          continue;
        }
      }

      approvedCalls.push({
        block: toolUseBlock,
        tool,
        parsedInput: toolInputParse.data,
      });
    }

    const safeCalls: ApprovedCall[] = [];
    const unsafeCalls: ApprovedCall[] = [];

    for (const call of approvedCalls) {
      if (call.tool.isConcurrencySafe) {
        safeCalls.push(call);
      } else {
        unsafeCalls.push(call);
      }
    }

    /**
     * Execute concurrency safe tools in parallel
     */

    if (!silent && safeCalls.length > 0) {
      spinner.start(
        safeCalls.length === 1
          ? `Running ${safeCalls[0].tool.name}...`
          : `Running ${safeCalls.length} tools in parallel...`,
      );
    }

    const safeResults = await Promise.all(
      safeCalls.map(({ block, tool, parsedInput }) =>
        executeToolCall(block, tool, parsedInput, session, silent),
      ),
    );

    if (!silent && safeCalls.length > 0) {
      spinner.stop();
    }

    /**
     * Execute concurrency unsafe tools sequential
     */

    const unsafeResults: Anthropic.ToolResultBlockParam[] = [];

    for (const { block, tool, parsedInput } of unsafeCalls) {
      if (!silent && !tool.isInteractive) {
        spinner.start(`Running ${tool.name}...`);
      }

      const result = await executeToolCall(
        block,
        tool,
        parsedInput,
        session,
        silent,
      );

      unsafeResults.push(result);

      if (!silent && !tool.isInteractive) {
        spinner.stop();
      }
    }

    /**
     * You can do this too for some models:
     * [...deniedResults, ...safeResults, ...unsafeResults]
     */
    const resultById = new Map<string, Anthropic.ToolResultBlockParam>();
    for (const r of deniedResults) resultById.set(r.tool_use_id, r);
    for (const r of safeResults) resultById.set(r.tool_use_id, r);
    for (const r of unsafeResults) resultById.set(r.tool_use_id, r);

    for (const toolUseBlock of toolUseBlocks) {
      const result = resultById.get(toolUseBlock.id);
      if (result) toolResults.push(result);
    }

    conversation.push({
      role: "user",
      content: toolResults,
    });

    await appendSessionEntry(session.sessionPath, {
      type: "user",
      timestamp: new Date().toISOString(),
      cwd: process.cwd(),
      message: {
        role: "user",
        content: toolResults,
      },
    });

    turns++;
  }
}

/**
 * Onboarding
 */

async function onboarding(): Promise<string> {
  console.log();
  console.log(pc.bold(pc.cyan("Welcome to Tiny Agent!")));
  console.log(pc.dim("Let's set up your Anthropic API key."));
  console.log(pc.dim("Get one at https://console.anthropic.com/settings/keys"));
  console.log();

  while (true) {
    let apiKey: string;
    try {
      apiKey = await input({
        message: "API key",
        theme: {
          style: {
            answer: (text: string) => pc.white(text),
          },
        },
        validate: (value) => {
          if (!value.trim()) return "API key is required";
          if (!value.startsWith("sk-ant-")) {
            return "Invalid API key format (should start with 'sk-ant-')";
          }
          return true;
        },
      });
    } catch {
      console.log();
      logger.dim("Cancelled.");
      process.exit(0);
    }

    const spinner = ora({
      text: "Validating API key...",
      color: "cyan",
    }).start();
    const result = await validateApiKey(apiKey.trim());
    spinner.stop();

    if (result.valid) {
      await saveConfig({ apiKey: apiKey.trim() });
      logger.success(`API key validated and saved to ${CONFIG_FILE}`);
      console.log();
      return apiKey.trim();
    }

    logger.error(`Validation failed: ${result.reason}`);
    console.log();
  }
}

/**
 * Main entry
 */

async function main() {
  let apiKey = await resolveApiKey();

  if (!apiKey) {
    apiKey = await onboarding();
  }

  const session = createSession(apiKey);
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

        case "/login": {
          const newKey = await onboarding();
          session.client = new Anthropic({ apiKey: newKey });
          logger.info("API key updated.");
          console.log();
          continue;
        }

        case "/logout":
          await saveConfig({});
          logger.info(
            "API key cleared. Run `tiny-agent` again to log in with a new key.",
          );
          console.log();
          process.exit(0);

        case "/resume": {
          const sessions = await listSessions();

          if (sessions.length === 0) {
            logger.warn("No previous sessions found for this directory.");
            continue;
          }

          let choice: string;
          try {
            choice = await select({
              message: "Resume which session?",
              choices: sessions.slice(0, 10).map((session) => ({
                name: `${session.label} - ${session.mtime.toLocaleString()}`,
                value: session.path,
              })),
            });
          } catch {
            logger.dim("Resume cancelled");
            continue;
          }

          const loadedSession = await loadSession(choice);
          conversation.length = 0;
          conversation.push(...loadedSession);

          session.sessionPath = choice;
          session.sessionId = path.basename(choice, ".jsonl");

          logger.info(`Resumed ${loadedSession.length} messages.`);

          continue;
        }

        case "/clear": {
          conversation.length = 0;

          const newSessionId = randomUUID();
          session.sessionId = newSessionId;
          session.sessionPath = path.join(
            getProjectDir(),
            `${newSessionId}.jsonl`,
          );

          console.clear();
          logger.dim("Conversation cleared.");
          console.log();
          continue;
        }

        case "/usage":
          session.showUsage = !session.showUsage;
          logger.info(`Usage display ${session.showUsage ? "on" : "off"}`);
          continue;

        case "/tokens": {
          const estimate = estimateTokens(conversation);
          logger.info(`Estimated tokens: ${estimate}`);
          continue;
        }

        default:
          logger.warn(`Unknown command: ${cmd}`);
          continue;
      }
    }

    conversation.push({ role: "user", content: message });

    await appendSessionEntry(session.sessionPath, {
      type: "user",
      timestamp: new Date().toISOString(),
      cwd: process.cwd(),
      message: {
        role: "user",
        content: message,
      },
    });

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
      const classifiedError = classifyError(error);
      logger.error(classifiedError.message);

      if (classifiedError.kind === "unknown") {
        console.error(error);
      }
    }
  }
}

main();
