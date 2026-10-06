export type ClientToolDefinition = {
  name: string;
  namespace?: string;
  namespaceDescription?: string;
  description?: string;
  inputSchema: Record<string, unknown>;
};

export type ClientToolOutput = {
  callId: string;
  name?: string;
  namespace?: string;
  output: string;
  isError?: boolean;
};

export type PendingClientToolCall = {
  callId: string;
  itemId: string;
  name: string;
  namespace?: string;
  arguments: string;
};

export type ResolvedToolChoice = {
  tools: ClientToolDefinition[];
  instruction?: string;
  required: boolean;
  maxParallelToolCalls?: number;
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function schemaOrDefault(value: unknown): Record<string, unknown> {
  return (
    asRecord(value) ?? {
      type: "object",
      properties: {},
    }
  );
}

function pushUnique(
  out: ClientToolDefinition[],
  seen: Set<string>,
  definition: ClientToolDefinition,
): void {
  if (!definition.name.trim()) {
    throw new Error("Function tool is missing name");
  }
  const identity = JSON.stringify([definition.namespace ?? null, definition.name]);
  if (seen.has(identity)) {
    const qualifiedName = definition.namespace
      ? `${definition.namespace}.${definition.name}`
      : definition.name;
    throw new Error(`Duplicate function tool name: ${qualifiedName}`);
  }
  seen.add(identity);
  out.push(definition);
}

function functionDefinition(tool: Record<string, unknown>): ClientToolDefinition {
  const wrapped = asRecord(tool.function);
  const fn = wrapped ?? tool;
  if (typeof fn.name !== "string") {
    throw new Error("Function tool is missing name");
  }
  return {
    name: fn.name,
    description:
      typeof fn.description === "string" ? fn.description : undefined,
    inputSchema: schemaOrDefault(fn.parameters),
  };
}

/**
 * Parse OpenAI Chat Completions (`function` wrapper), Responses (flat
 * function), namespaced Responses functions, and legacy Chat `functions`
 * definitions. Unsupported tools are rejected because the ACP bridge cannot
 * execute them on behalf of the caller.
 */
export function parseOpenAiFunctionTools(
  tools?: readonly unknown[],
  functions?: readonly unknown[],
  options: { allowNamespaces?: boolean } = {},
): ClientToolDefinition[] {
  const out: ClientToolDefinition[] = [];
  const seen = new Set<string>();

  for (const value of tools ?? []) {
    const tool = asRecord(value);
    if (!tool) throw new Error("Invalid tool definition");
    if (tool.type === "namespace") {
      if (!options.allowNamespaces) {
        throw new Error("Namespace tools are only supported by the Responses API");
      }
      if (typeof tool.name !== "string" || !tool.name.trim()) {
        throw new Error("Namespace tool is missing name");
      }
      if (!Array.isArray(tool.tools)) {
        throw new Error(`Namespace ${tool.name} is missing tools array`);
      }
      for (const member of tool.tools) {
        const nested = asRecord(member);
        if (!nested) {
          throw new Error(`Invalid function tool in namespace ${tool.name}`);
        }
        if (nested.type !== "function") {
          throw new Error(
            `Unsupported tool type in namespace ${tool.name}: ${
              typeof nested.type === "string" ? nested.type : "unknown"
            }`,
          );
        }
        pushUnique(out, seen, {
          ...functionDefinition(nested),
          namespace: tool.name,
          ...(typeof tool.description === "string"
            ? { namespaceDescription: tool.description }
            : {}),
        });
      }
      continue;
    }
    if (tool.type !== "function") {
      throw new Error(
        `Unsupported tool type: ${
          typeof tool.type === "string" ? tool.type : "unknown"
        }`,
      );
    }
    pushUnique(out, seen, functionDefinition(tool));
  }

  for (const value of functions ?? []) {
    const fn = asRecord(value);
    if (!fn || typeof fn.name !== "string") {
      throw new Error("Function is missing name");
    }
    pushUnique(out, seen, {
      name: fn.name,
      description:
        typeof fn.description === "string" ? fn.description : undefined,
      inputSchema: schemaOrDefault(fn.parameters),
    });
  }

  return out;
}

export function parseAnthropicFunctionTools(
  tools?: readonly unknown[],
): ClientToolDefinition[] {
  const out: ClientToolDefinition[] = [];
  const seen = new Set<string>();
  for (const value of tools ?? []) {
    const tool = asRecord(value);
    if (!tool || typeof tool.name !== "string") {
      throw new Error("Anthropic tool is missing name");
    }
    pushUnique(out, seen, {
      name: tool.name,
      description:
        typeof tool.description === "string" ? tool.description : undefined,
      inputSchema: schemaOrDefault(tool.input_schema),
    });
  }
  return out;
}

type ToolChoiceSelector = { name: string; namespace?: string };

function namedChoice(choice: unknown): ToolChoiceSelector | undefined {
  const rec = asRecord(choice);
  if (!rec) return undefined;
  const fn = asRecord(rec.function);
  const name =
    typeof rec.name === "string"
      ? rec.name
      : typeof fn?.name === "string"
        ? fn.name
        : undefined;
  const namespace =
    typeof rec.namespace === "string"
      ? rec.namespace
      : typeof fn?.namespace === "string"
        ? fn.namespace
        : undefined;
  if (name !== undefined) return { name, namespace };
  return undefined;
}

export function resolveToolChoice(
  tools: readonly ClientToolDefinition[],
  choice: unknown,
  opts: { parallelToolCalls?: boolean; anthropic?: boolean } = {},
): ResolvedToolChoice {
  if (choice === "none" || asRecord(choice)?.type === "none") {
    return { tools: [], required: false };
  }

  const required =
    choice === "required" ||
    choice === "any" ||
    asRecord(choice)?.type === "any";
  const selected = namedChoice(choice);
  const choiceRecord = asRecord(choice);
  if (choiceRecord?.type === "namespace") {
    throw new Error("Namespace tool_choice must identify a function");
  }
  if (choiceRecord?.namespace != null && typeof choiceRecord.namespace !== "string") {
    throw new Error("tool_choice namespace must be a string");
  }
  if (choiceRecord?.function != null) {
    const functionChoice = asRecord(choiceRecord.function);
    if (!functionChoice) {
      throw new Error("tool_choice function must be an object");
    }
    if (
      functionChoice.namespace != null &&
      typeof functionChoice.namespace !== "string"
    ) {
      throw new Error("tool_choice namespace must be a string");
    }
  }
  let resolved = [...tools];
  let instruction: string | undefined;

  if (selected) {
    if (!selected.name.trim()) {
      throw new Error("tool_choice function is missing name");
    }
    if (selected.namespace !== undefined && !selected.namespace.trim()) {
      throw new Error("tool_choice namespace must not be empty");
    }
    const matches = tools.filter(
      (item) =>
        item.name === selected.name &&
        (selected.namespace === undefined ||
          item.namespace === selected.namespace),
    );
    if (matches.length === 0) {
      const qualifiedName = selected.namespace
        ? `${selected.namespace}.${selected.name}`
        : selected.name;
      throw new Error(`Unknown tool_choice function: ${qualifiedName}`);
    }
    if (matches.length > 1) {
      throw new Error(
        `Ambiguous tool_choice function: ${selected.name}; specify namespace`,
      );
    }
    const [tool] = matches;
    resolved = [tool];
    const qualifiedName = tool.namespace
      ? `${tool.namespace}.${tool.name}`
      : tool.name;
    instruction = `You must call the ${qualifiedName} tool.`;
  } else if (required) {
    instruction = "You must call at least one available tool.";
  }

  if (opts.parallelToolCalls === false && resolved.length > 0) {
    instruction = [instruction, "Call at most one tool at a time."]
      .filter(Boolean)
      .join(" ");
  }
  return {
    tools: resolved,
    required: required || selected !== undefined,
    ...(opts.parallelToolCalls === false ? { maxParallelToolCalls: 1 } : {}),
    ...(instruction ? { instruction } : {}),
  };
}

function stringifyOutput(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const text = value
      .map((part) => {
        if (typeof part === "string") return part;
        const rec = asRecord(part);
        return (rec?.type === "text" ||
          rec?.type === "input_text" ||
          rec?.type === "output_text") &&
          typeof rec.text === "string"
          ? rec.text
          : "";
      })
      .filter(Boolean)
      .join("\n");
    if (text) return text;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function chatToolOutputs(
  messages: readonly unknown[],
): ClientToolOutput[] {
  const outputs: ClientToolOutput[] = [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = asRecord(messages[i]);
    if (!message || message.role !== "tool") break;
    if (typeof message.tool_call_id !== "string" || !message.tool_call_id) {
      throw new Error("Tool message is missing tool_call_id");
    }
    outputs.unshift({
      callId: message.tool_call_id,
      output: stringifyOutput(message.content),
    });
  }
  return outputs;
}

export function responsesToolOutputs(input: unknown): ClientToolOutput[] {
  if (!Array.isArray(input)) return [];
  const outputs: ClientToolOutput[] = [];
  for (const value of input) {
    const item = asRecord(value);
    if (!item || item.type !== "function_call_output") continue;
    if (typeof item.call_id !== "string" || !item.call_id) {
      throw new Error("function_call_output is missing call_id");
    }
    const hasName = Object.hasOwn(item, "name");
    const hasNamespace = Object.hasOwn(item, "namespace");
    if (hasName && (typeof item.name !== "string" || !item.name.trim())) {
      throw new Error("function_call_output has invalid name");
    }
    if (
      hasNamespace &&
      (typeof item.namespace !== "string" || !item.namespace.trim())
    ) {
      throw new Error("function_call_output has invalid namespace");
    }
    if (hasNamespace && !hasName) {
      throw new Error("function_call_output namespace is missing name");
    }
    outputs.push({
      callId: item.call_id,
      ...(hasName ? { name: item.name as string } : {}),
      ...(hasNamespace ? { namespace: item.namespace as string } : {}),
      output: stringifyOutput(item.output),
    });
  }
  return outputs;
}

export function anthropicToolOutputs(
  messages: readonly unknown[],
): ClientToolOutput[] {
  const last = asRecord(messages[messages.length - 1]);
  if (!last || last.role !== "user" || !Array.isArray(last.content)) return [];
  const outputs: ClientToolOutput[] = [];
  for (const value of last.content) {
    const block = asRecord(value);
    if (!block || block.type !== "tool_result") continue;
    if (typeof block.tool_use_id !== "string" || !block.tool_use_id) {
      throw new Error("tool_result is missing tool_use_id");
    }
    outputs.push({
      callId: block.tool_use_id,
      output: stringifyOutput(block.content),
      isError: block.is_error === true,
    });
  }
  return outputs;
}
