import { createHash } from "node:crypto";

import type { ToolTurnResult } from "./acp-tool-session.js";
import {
  responsesToolOutputs,
  type ClientToolOutput,
} from "./tool-types.js";

export type ResponsesToolControlKey =
  | "model"
  | "mode"
  | "instructions"
  | "tools"
  | "tool_choice"
  | "parallel_tool_calls"
  | "reasoning";

export type ResponsesToolReplayState = {
  initialInput: string[];
  history: string[];
  controls: Record<ResponsesToolControlKey, string>;
  acceptedOutputs: Record<string, string>;
  callIds: string[];
};

export type ResponsesToolControlValues = Record<
  ResponsesToolControlKey,
  unknown
>;

export type ResponsesToolResumePlan = {
  kind: "outputs" | "replay";
  outputs: ClientToolOutput[];
};

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, stableValue(child)]),
  );
}

function fingerprint(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(stableValue(value)))
    .digest("hex");
}

function canonicalArguments(value: unknown): unknown {
  if (typeof value !== "string") return value ?? {};
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function canonicalContent(value: unknown): unknown {
  if (typeof value === "string") return [{ type: "text", text: value }];
  if (!Array.isArray(value)) return stableValue(value);
  return value.map((part) => {
    if (typeof part === "string") return { type: "text", text: part };
    if (!isRecord(part)) return stableValue(part);
    if (
      (part.type === "input_text" ||
        part.type === "output_text" ||
        part.type === "text") &&
      typeof part.text === "string"
    ) {
      return { type: "text", text: part.text };
    }
    const { annotations: _annotations, ...rest } = part;
    return stableValue(rest);
  });
}

function canonicalInputItem(value: unknown): unknown {
  if (typeof value === "string") {
    return {
      type: "message",
      role: "user",
      content: canonicalContent(value),
    };
  }
  if (!isRecord(value)) return stableValue(value);

  if (value.type === "function_call") {
    return {
      type: "function_call",
      call_id: value.call_id,
      ...(typeof value.namespace === "string"
        ? { namespace: value.namespace }
        : {}),
      name: value.name,
      arguments: canonicalArguments(value.arguments),
    };
  }

  if (value.type === "function_call_output") {
    const output = responsesToolOutputs([value])[0];
    if (!output) return stableValue(value);
    return canonicalOutput(output);
  }

  if (value.type === undefined || value.type === "message") {
    return {
      type: "message",
      role: typeof value.role === "string" ? value.role : "user",
      content: canonicalContent(value.content ?? value.text),
    };
  }

  const { id: _id, status: _status, annotations: _annotations, ...rest } = value;
  return stableValue(rest);
}

function canonicalInput(input: unknown): unknown[] {
  if (typeof input === "string") return [canonicalInputItem(input)];
  if (Array.isArray(input)) return input.map(canonicalInputItem);
  return input == null ? [] : [canonicalInputItem(input)];
}

function canonicalOutput(output: ClientToolOutput): unknown {
  return {
    type: "function_call_output",
    call_id: output.callId,
    ...(output.namespace !== undefined ? { namespace: output.namespace } : {}),
    ...(output.name !== undefined ? { name: output.name } : {}),
    output: output.output,
  };
}

function itemFingerprint(value: unknown): string {
  return fingerprint(canonicalInputItem(value));
}

function isFunctionCallOutput(value: unknown): boolean {
  return isRecord(value) && value.type === "function_call_output";
}

function inputItems(input: unknown): unknown[] {
  if (typeof input === "string") return [input];
  if (Array.isArray(input)) return input;
  return input == null ? [] : [input];
}

export function createResponsesToolReplayState(opts: {
  input: unknown;
  controls: ResponsesToolControlValues;
}): ResponsesToolReplayState {
  return {
    initialInput: canonicalInput(opts.input).map(fingerprint),
    history: [],
    controls: Object.fromEntries(
      Object.entries(opts.controls).map(([key, value]) => [key, fingerprint(value)]),
    ) as Record<ResponsesToolControlKey, string>,
    acceptedOutputs: {},
    callIds: [],
  };
}

export function fingerprintResponsesToolControls(
  values: ResponsesToolControlValues,
): Record<ResponsesToolControlKey, string> {
  return Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key, fingerprint(value)]),
  ) as Record<ResponsesToolControlKey, string>;
}

export function assertResponsesToolControls(
  state: ResponsesToolReplayState,
  supplied: Partial<Record<ResponsesToolControlKey, unknown>>,
): void {
  for (const [key, value] of Object.entries(supplied) as Array<
    [ResponsesToolControlKey, unknown]
  >) {
    if (fingerprint(value) !== state.controls[key]) {
      throw new Error(`Responses ${key} changed while an ACP tool turn is active`);
    }
  }
}

export function responsesTrailingOutputCallIds(input: unknown): string[] {
  const items = inputItems(input);
  const outputs: unknown[] = [];
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (!isFunctionCallOutput(items[i])) break;
    outputs.unshift(items[i]);
  }
  return responsesToolOutputs(outputs).map((output) => output.callId);
}

export function responsesReferencedCallIds(input: unknown): string[] {
  const ids = new Set<string>();
  for (const item of inputItems(input)) {
    if (!isRecord(item)) continue;
    if (
      (item.type === "function_call" ||
        item.type === "function_call_output") &&
      typeof item.call_id === "string" &&
      item.call_id
    ) {
      ids.add(item.call_id);
    }
  }
  return [...ids];
}

export function prepareResponsesToolResume(opts: {
  input: unknown;
  state: ResponsesToolReplayState;
}): ResponsesToolResumePlan {
  const items = inputItems(opts.input);
  const outputOnly = items.length > 0 && items.every(isFunctionCallOutput);
  if (outputOnly) {
    const outputs = responsesToolOutputs(items);
    const fresh: ClientToolOutput[] = [];
    for (const output of outputs) {
      const accepted = opts.state.acceptedOutputs[output.callId];
      if (accepted !== undefined) {
        if (accepted !== fingerprint(canonicalOutput(output))) {
          throw new Error(
            `Previously resolved function_call_output changed for ${output.callId}`,
          );
        }
        continue;
      }
      fresh.push(output);
    }
    assertUniqueOutputIds(fresh);
    if (fresh.length === 0) {
      throw new Error("No new pending function_call_output items were supplied");
    }
    return { kind: "outputs", outputs: fresh };
  }

  const incoming = canonicalInput(opts.input).map(fingerprint);
  const expected = [...opts.state.initialInput, ...opts.state.history];
  if (
    incoming.length < expected.length ||
    expected.some((item, index) => incoming[index] !== item)
  ) {
    throw new Error(
      "Responses input does not match the original input and active ACP tool history",
    );
  }

  const freshItems = items.slice(expected.length);
  if (
    freshItems.length === 0 ||
    freshItems.some((item) => !isFunctionCallOutput(item))
  ) {
    throw new Error(
      "An active ACP tool replay may append only new function_call_output items",
    );
  }
  const outputs = responsesToolOutputs(freshItems);
  assertUniqueOutputIds(outputs);
  if (outputs.length !== freshItems.length) {
    throw new Error("Responses replay contains malformed function_call_output items");
  }
  return { kind: "replay", outputs };
}

function assertUniqueOutputIds(outputs: readonly ClientToolOutput[]): void {
  const seen = new Set<string>();
  for (const output of outputs) {
    if (seen.has(output.callId)) {
      throw new Error(`Duplicate function_call_output for ${output.callId}`);
    }
    seen.add(output.callId);
  }
}

export function appendResponsesToolOutputs(
  state: ResponsesToolReplayState,
  outputs: readonly ClientToolOutput[],
): void {
  for (const output of outputs) {
    const item = canonicalOutput(output);
    const value = fingerprint(item);
    const prior = state.acceptedOutputs[output.callId];
    if (prior !== undefined && prior !== value) {
      throw new Error(
        `Previously resolved function_call_output changed for ${output.callId}`,
      );
    }
    state.acceptedOutputs[output.callId] = value;
    state.history.push(value);
  }
}

export function appendResponsesToolResult(
  state: ResponsesToolReplayState,
  result: ToolTurnResult,
): void {
  if (result.text) {
    state.history.push(
      itemFingerprint({
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: result.text, annotations: [] }],
      }),
    );
  }
  if (result.status === "tool_calls") {
    for (const call of result.toolCalls) {
      state.history.push(
        itemFingerprint({
          id: call.itemId,
          type: "function_call",
          status: "completed",
          call_id: call.callId,
          ...(call.namespace ? { namespace: call.namespace } : {}),
          name: call.name,
          arguments: call.arguments,
        }),
      );
      state.callIds.push(call.callId);
    }
  }
}
