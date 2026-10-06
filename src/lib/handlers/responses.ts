import { randomUUID } from "node:crypto";
import * as http from "node:http";

import { buildAgentFixedArgs } from "../agent-cmd-args.js";
import {
  getAccountStats,
  getNextAccountConfigDir,
  reportRateLimit,
  reportRequestEnd,
  reportRequestError,
  reportRequestStart,
  reportRequestSuccess,
} from "../account-pool.js";
import {
  BRIDGE_AGENT_PROMPT_SEPARATOR,
  buildBridgeContextPreamble,
} from "../bridge-context-preamble.js";
import type { BridgeConfig } from "../config.js";
import type { CursorExecutionMode } from "../execution-mode.js";
import { json, startSseKeepalive, writeSseHeaders } from "../http.js";
import {
  runAgentStream,
  runAgentSync,
  startAgentToolSession,
} from "../agent-runner.js";
import type { ToolTurnEvent, ToolTurnResult } from "../acp-tool-session.js";
import { createStreamParser } from "../cli-stream-parser.js";
import {
  resolveModelForExecution,
  UnsupportedReasoningEffortError,
} from "../model-map.js";
import {
  buildPromptFromMessages,
  normalizeModelId,
  responsesInputToMessages,
  toolsToSystemText,
  type OpenAiResponsesRequest,
} from "../openai.js";
import {
  logAccountAssigned,
  logAccountStats,
  logAgentError,
  logModelResolution,
  logTrafficRequest,
  logTrafficResponse,
  type TrafficMessage,
} from "../request-log.js";
import { rememberResolvedModel, resolveModel } from "../resolve-model.js";
import { resolveRequestMode } from "../resolve-mode.js";
import { sanitizeMessages } from "../sanitize.js";
import { resolveWorkspace } from "../workspace.js";
import {
  fitPromptToWinCmdline,
  warnPromptTruncated,
} from "../win-cmdline-limit.js";
import { abortOnClientDisconnect } from "../client-disconnect.js";
import {
  parseOpenAiFunctionTools,
  resolveToolChoice,
  responsesToolOutputs,
  type PendingClientToolCall,
  type ClientToolDefinition,
} from "../tool-types.js";
import {
  ToolSessionError,
  toolSessionOwnerKey,
  type ToolSessionRecord,
  type ToolSessionRegistry,
} from "../tool-session-registry.js";
import { getCachedCursorModels, type ModelCacheRef } from "./models.js";
import {
  appendResponsesToolOutputs,
  appendResponsesToolResult,
  assertResponsesToolControls,
  createResponsesToolReplayState,
  prepareResponsesToolResume,
  responsesReferencedCallIds,
  responsesTrailingOutputCallIds,
  type ResponsesToolControlKey,
  type ResponsesToolControlValues,
} from "../responses-tool-replay.js";

function isRateLimited(stderr: string): boolean {
  return /\b429\b|rate.?limit|too many requests/i.test(stderr);
}

export type ResponsesCtx = {
  config: BridgeConfig;
  lastRequestedModelRef: { current?: string };
  modelCacheRef: ModelCacheRef;
  toolSessions: ToolSessionRegistry;
};

type ResponseStatus = "in_progress" | "completed" | "failed";

function functionCallItem(call: PendingClientToolCall) {
  return {
    id: call.itemId,
    type: "function_call",
    status: "completed",
    call_id: call.callId,
    ...(call.namespace ? { namespace: call.namespace } : {}),
    name: call.name,
    arguments: call.arguments,
  };
}

function responsesToolControlValues(opts: {
  model: string;
  mode: CursorExecutionMode;
  body: OpenAiResponsesRequest;
  tools: ClientToolDefinition[];
}): ResponsesToolControlValues {
  return {
    model: opts.model,
    mode: opts.mode,
    instructions:
      typeof opts.body.instructions === "string"
        ? opts.body.instructions.trim()
        : null,
    tools: [...opts.tools].sort((left, right) =>
      `${left.namespace ?? ""}\u0000${left.name}`.localeCompare(
        `${right.namespace ?? ""}\u0000${right.name}`,
      ),
    ),
    tool_choice: opts.body.tool_choice ?? "auto",
    parallel_tool_calls: opts.body.parallel_tool_calls ?? true,
    reasoning: opts.body.reasoning ?? null,
  };
}

function suppliedResponseToolControls(
  body: OpenAiResponsesRequest,
  values: ResponsesToolControlValues,
  includeInstructions: boolean,
  headerMode: string | string[] | undefined,
): Partial<Record<ResponsesToolControlKey, unknown>> {
  const supplied: Partial<Record<ResponsesToolControlKey, unknown>> = {};
  if (Object.hasOwn(body, "model")) supplied.model = values.model;
  if (
    Object.hasOwn(body, "mode") ||
    (typeof headerMode === "string" && headerMode.trim()) ||
    (Array.isArray(headerMode) && headerMode.some((value) => value.trim()))
  ) {
    supplied.mode = values.mode;
  }
  if (includeInstructions || Object.hasOwn(body, "instructions")) {
    supplied.instructions = values.instructions;
  }
  if (Object.hasOwn(body, "tools")) supplied.tools = values.tools;
  if (Object.hasOwn(body, "tool_choice")) {
    supplied.tool_choice = values.tool_choice;
  }
  if (Object.hasOwn(body, "parallel_tool_calls")) {
    supplied.parallel_tool_calls = values.parallel_tool_calls;
  }
  if (Object.hasOwn(body, "reasoning")) supplied.reasoning = values.reasoning;
  return supplied;
}

function structuredResponseObject(opts: {
  body: OpenAiResponsesRequest;
  id: string;
  createdAt: number;
  model: string | undefined;
  result: ToolTurnResult;
  previousResponseId?: string | null;
  promptLength: number;
  messageId?: string;
}) {
  const output: Array<Record<string, unknown>> = [];
  if (opts.result.text) {
    output.push({
      id: opts.messageId ?? `msg_${randomUUID().replace(/-/g, "")}`,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: opts.result.text,
          annotations: [],
        },
      ],
    });
  }
  if (opts.result.status === "tool_calls") {
    output.push(...opts.result.toolCalls.map(functionCallItem));
  }
  const inputTokens = Math.max(1, Math.round(opts.promptLength / 4));
  const outputTokens = Math.max(1, Math.round(opts.result.text.length / 4));
  return {
    id: opts.id,
    object: "response",
    created_at: opts.createdAt,
    status: "completed",
    background: false,
    error: null,
    incomplete_details: null,
    instructions: opts.body.instructions ?? null,
    max_output_tokens: opts.body.max_output_tokens ?? null,
    model: opts.model,
    output,
    output_text: opts.result.text,
    parallel_tool_calls: opts.body.parallel_tool_calls ?? true,
    previous_response_id:
      opts.previousResponseId ?? opts.body.previous_response_id ?? null,
    reasoning: opts.body.reasoning ?? null,
    service_tier: opts.body.service_tier ?? "default",
    store: opts.body.store ?? true,
    temperature: opts.body.temperature ?? null,
    text: opts.body.text ?? { format: { type: "text" } },
    tool_choice: opts.body.tool_choice ?? "auto",
    tools: opts.body.tools ?? [],
    top_p: opts.body.top_p ?? null,
    truncation: opts.body.truncation ?? "disabled",
    usage: {
      input_tokens: inputTokens,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: outputTokens,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: inputTokens + outputTokens,
    },
    user: opts.body.user ?? null,
    metadata: opts.body.metadata ?? null,
  };
}

async function writeStructuredResponseTurn(opts: {
  res: http.ServerResponse;
  body: OpenAiResponsesRequest;
  stream: boolean;
  id: string;
  createdAt: number;
  model: string | undefined;
  previousResponseId?: string | null;
  promptLength: number;
  onResult?: (result: ToolTurnResult) => void;
  run: (
    listener?: (event: ToolTurnEvent) => void,
  ) => Promise<ToolTurnResult>;
}): Promise<ToolTurnResult> {
  const messageId = `msg_${randomUUID().replace(/-/g, "")}`;
  if (!opts.stream) {
    const result = await opts.run();
    opts.onResult?.(result);
    json(
      opts.res,
      200,
      structuredResponseObject({
        body: opts.body,
        id: opts.id,
        createdAt: opts.createdAt,
        model: opts.model,
        result,
        previousResponseId: opts.previousResponseId,
        promptLength: opts.promptLength,
        messageId,
      }),
    );
    return result;
  }

  writeSseHeaders(opts.res);
  const stopKeepalive = startSseKeepalive(opts.res);
  try {
    const initial = {
      id: opts.id,
      object: "response",
      created_at: opts.createdAt,
      status: "in_progress",
      model: opts.model,
      output: [],
    };
    writeResponseEvent(opts.res, "response.created", { response: initial });
    let textStarted = false;
    let emittedText = "";
    const emitText = (text: string) => {
      if (!textStarted) {
        textStarted = true;
        writeResponseEvent(opts.res, "response.output_item.added", {
          response_id: opts.id,
          output_index: 0,
          item: {
            id: messageId,
            type: "message",
            status: "in_progress",
            role: "assistant",
            content: [],
          },
        });
        writeResponseEvent(opts.res, "response.content_part.added", {
          response_id: opts.id,
          item_id: messageId,
          output_index: 0,
          content_index: 0,
          part: { type: "output_text", text: "", annotations: [] },
        });
      }
      emittedText += text;
      writeResponseEvent(opts.res, "response.output_text.delta", {
        response_id: opts.id,
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        delta: text,
      });
    };
    const result = await opts.run((event) => {
      if (event.type === "text") emitText(event.text);
    });
    opts.onResult?.(result);
    if (result.text.length > emittedText.length) {
      emitText(result.text.slice(emittedText.length));
    }
    let outputIndex = textStarted ? 1 : 0;
    if (textStarted) {
      writeResponseEvent(opts.res, "response.output_text.done", {
        response_id: opts.id,
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        text: result.text,
      });
      writeResponseEvent(opts.res, "response.content_part.done", {
        response_id: opts.id,
        item_id: messageId,
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: result.text, annotations: [] },
      });
      writeResponseEvent(opts.res, "response.output_item.done", {
        response_id: opts.id,
        output_index: 0,
        item: {
          id: messageId,
          type: "message",
          status: "completed",
          role: "assistant",
          content: [
            { type: "output_text", text: result.text, annotations: [] },
          ],
        },
      });
    }
    if (result.status === "tool_calls") {
      for (const call of result.toolCalls) {
        writeResponseEvent(opts.res, "response.output_item.added", {
          response_id: opts.id,
          output_index: outputIndex,
          item: {
            id: call.itemId,
            type: "function_call",
            status: "in_progress",
            call_id: call.callId,
            ...(call.namespace ? { namespace: call.namespace } : {}),
            name: call.name,
            arguments: "",
          },
        });
        writeResponseEvent(
          opts.res,
          "response.function_call_arguments.delta",
          {
            response_id: opts.id,
            item_id: call.itemId,
            output_index: outputIndex,
            delta: call.arguments,
          },
        );
        writeResponseEvent(
          opts.res,
          "response.function_call_arguments.done",
          {
            response_id: opts.id,
            item_id: call.itemId,
            output_index: outputIndex,
            arguments: call.arguments,
          },
        );
        writeResponseEvent(opts.res, "response.output_item.done", {
          response_id: opts.id,
          output_index: outputIndex,
          item: functionCallItem(call),
        });
        outputIndex += 1;
      }
    }
    writeResponseEvent(opts.res, "response.completed", {
      response: structuredResponseObject({
        body: opts.body,
        id: opts.id,
        createdAt: opts.createdAt,
        model: opts.model,
        result,
        previousResponseId: opts.previousResponseId,
        promptLength: opts.promptLength,
        messageId,
      }),
    });
    opts.res.write("data: [DONE]\n\n");
    opts.res.end();
    return result;
  } finally {
    stopKeepalive();
  }
}

function createResponseObject(opts: {
  body: OpenAiResponsesRequest;
  id: string;
  itemId: string;
  createdAt: number;
  model: string | undefined;
  status: ResponseStatus;
  text: string;
  promptTokens: number;
  completionTokens: number;
  error?: { message: string; code: string } | null;
}) {
  const output =
    opts.status === "completed"
      ? [
          {
            id: opts.itemId,
            type: "message",
            status: "completed",
            role: "assistant",
            content: [
              {
                type: "output_text",
                text: opts.text,
                annotations: [],
              },
            ],
          },
        ]
      : [];
  const totalTokens = opts.promptTokens + opts.completionTokens;

  return {
    id: opts.id,
    object: "response",
    created_at: opts.createdAt,
    status: opts.status,
    background: false,
    error: opts.error ?? null,
    incomplete_details: null,
    instructions: opts.body.instructions ?? null,
    max_output_tokens: opts.body.max_output_tokens ?? null,
    model: opts.model,
    output,
    output_text: opts.text,
    parallel_tool_calls: opts.body.parallel_tool_calls ?? true,
    previous_response_id: opts.body.previous_response_id ?? null,
    reasoning: opts.body.reasoning ?? null,
    service_tier: opts.body.service_tier ?? "default",
    store: opts.body.store ?? false,
    temperature: opts.body.temperature ?? null,
    text: opts.body.text ?? { format: { type: "text" } },
    tool_choice: opts.body.tool_choice ?? "auto",
    tools: opts.body.tools ?? [],
    top_p: opts.body.top_p ?? null,
    truncation: opts.body.truncation ?? "disabled",
    usage: {
      input_tokens: opts.promptTokens,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: opts.completionTokens,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: totalTokens,
    },
    user: opts.body.user ?? null,
    metadata: opts.body.metadata ?? null,
  };
}

function createOutputItem(itemId: string, status: ResponseStatus, text: string) {
  return {
    id: itemId,
    type: "message",
    status,
    role: "assistant",
    content: [
      {
        type: "output_text",
        text,
        annotations: [],
      },
    ],
  };
}

function writeResponseEvent(
  res: http.ServerResponse,
  type: string,
  data: Record<string, unknown>,
): void {
  res.write(`event: ${type}\n`);
  res.write(`data: ${JSON.stringify({ type, ...data })}\n\n`);
}

function responseContentText(message: any): string {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter(
        (p) =>
          p?.type === "text" ||
          p?.type === "input_text" ||
          p?.type === "output_text",
      )
      .map((p) => p.text ?? "")
      .join("");
  }
  return "";
}

export async function handleResponses(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  ctx: ResponsesCtx,
  rawBody: string,
  method: string,
  pathname: string,
  remoteAddress: string,
): Promise<void> {
  const { config, lastRequestedModelRef, modelCacheRef } = ctx;
  const body = JSON.parse(rawBody || "{}") as OpenAiResponsesRequest;
  let selectedTools;
  let parsedTools = [] as ReturnType<typeof parseOpenAiFunctionTools>;
  let toolInstruction: string | undefined;
  let requireToolCall = false;
  let maxParallelToolCalls: number | undefined;
  let submittedToolOutputs;
  try {
    parsedTools = parseOpenAiFunctionTools(body.tools, undefined, {
      allowNamespaces: true,
    });
    const choice = resolveToolChoice(parsedTools, body.tool_choice, {
      parallelToolCalls: body.parallel_tool_calls,
    });
    selectedTools = choice.tools;
    toolInstruction = choice.instruction;
    requireToolCall = choice.required;
    maxParallelToolCalls = choice.maxParallelToolCalls;
    submittedToolOutputs = responsesToolOutputs(body.input);
  } catch (error) {
    json(res, 400, {
      error: {
        message: error instanceof Error ? error.message : String(error),
        code: "invalid_tools",
        type: "invalid_request_error",
      },
    });
    return;
  }
  const ownerKey = toolSessionOwnerKey(req, remoteAddress);
  const requested = normalizeModelId(body.model);
  const model = resolveModel(requested, lastRequestedModelRef, config);
  const models = await getCachedCursorModels(config, modelCacheRef);
  let decision;
  try {
    decision = resolveModelForExecution({
      requested: model,
      defaultModel: config.defaultModel,
      availableCursorIds: models.map((m) => m.id),
      reasoningEffort:
        typeof body.reasoning?.effort === "string"
          ? body.reasoning.effort
          : undefined,
    });
  } catch (error) {
    if (!(error instanceof UnsupportedReasoningEffortError)) throw error;
    json(res, 400, {
      error: {
        message: error.message,
        code: error.code,
        type: "invalid_request_error",
      },
    });
    return;
  }
  const cursorModel = decision.final;
  rememberResolvedModel(cursorModel, lastRequestedModelRef);
  logModelResolution(config.verbose, decision);
  const displayModel =
    decision.requestedWasDefault && config.defaultModel !== "default"
      ? config.defaultModel
      : model;
  const modelCatalogName = models.find(
    (item) => item.id === cursorModel,
  )?.name;
  const id = `resp_${randomUUID().replace(/-/g, "")}`;
  const createdAt = Math.floor(Date.now() / 1000);

  let mode: CursorExecutionMode;
  try {
    mode = resolveRequestMode(
      config,
      req.headers["x-cursor-mode"],
      body.mode,
      responsesInputToMessages(body),
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Invalid mode";
    json(res, 400, { error: { message: msg, code: "invalid_mode" } });
    return;
  }

  const controlValues = responsesToolControlValues({
    model: cursorModel,
    mode,
    body,
    tools: parsedTools,
  });
  const previousResponseId = body.previous_response_id;
  const trailingOutputCallIds = responsesTrailingOutputCallIds(body.input);
  let resumeRecord: ToolSessionRecord | undefined;
  if (config.useAcp) {
    if (submittedToolOutputs.length > 0 && typeof previousResponseId === "string") {
      resumeRecord = ctx.toolSessions.findByResponseId(
        ownerKey,
        previousResponseId,
      );
      if (!resumeRecord) {
        json(res, 409, {
          error: {
            message:
              "Response tool session is missing or expired; resend the original input",
            code: "tool_session_expired",
            type: "invalid_request_error",
          },
        });
        return;
      }
    } else if (
      submittedToolOutputs.length > 0 &&
      trailingOutputCallIds.length > 0
    ) {
      resumeRecord = ctx.toolSessions.findByAnyCallIds(
        "responses",
        ownerKey,
        trailingOutputCallIds,
      ) ?? ctx.toolSessions.findByReferencedCallIds(
        "responses",
        ownerKey,
        responsesReferencedCallIds(body.input),
      );
      if (!resumeRecord) {
        json(res, 409, {
          error: {
            message:
              "Response tool session is missing or expired; resend the original input",
            code: "tool_session_expired",
            type: "invalid_request_error",
          },
        });
        return;
      }
    } else {
      const referenced = responsesReferencedCallIds(body.input);
      const activeRecord = ctx.toolSessions.findByReferencedCallIds(
        "responses",
        ownerKey,
        referenced,
      );
      if (activeRecord) {
        json(res, 400, {
          error: {
            message:
              "An active ACP tool replay cannot add user or other input before pending function_call_output items",
            code: "invalid_tool_resume",
            type: "invalid_request_error",
          },
        });
        return;
      }
    }
  }
  const structuredToolStart =
    config.useAcp && selectedTools.length > 0 && !resumeRecord;
  const cleanMessages = sanitizeMessages(responsesInputToMessages(body));
  const toolsText = structuredToolStart
    ? undefined
    : body.tool_choice === "none"
      ? undefined
      : toolsToSystemText(body.tools);
  const messagesWithTools = [
    ...(toolInstruction && structuredToolStart
      ? [{ role: "system", content: toolInstruction }]
      : []),
    ...(toolsText ? [{ role: "system", content: toolsText }] : []),
    ...cleanMessages,
  ];
  const prompt = buildPromptFromMessages(messagesWithTools);

  const trafficMessages: TrafficMessage[] = cleanMessages.map((m: any) => ({
    role: String(m?.role ?? "user"),
    content: responseContentText(m),
  }));
  logTrafficRequest(
    config.verbose,
    model ?? cursorModel,
    trafficMessages,
    !!body.stream,
  );

  if (config.useAcp && resumeRecord) {
    const record = resumeRecord;
    if (
      record.api !== "responses" ||
      !record.responsesReplay
    ) {
      json(res, 409, {
        error: {
          message:
            "Response tool session is missing or expired; resend the original input",
          code: "tool_session_expired",
          type: "invalid_request_error",
        },
      });
      return;
    }
    let resumePlan;
    try {
      resumePlan = prepareResponsesToolResume({
        input: body.input,
        state: record.responsesReplay,
      });
      assertResponsesToolControls(
        record.responsesReplay,
        suppliedResponseToolControls(
          body,
          controlValues,
          resumePlan.kind === "replay",
          req.headers["x-cursor-mode"],
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      json(res, 400, {
        error: {
          message,
          code: message.startsWith("Responses tools changed")
            ? "invalid_tool_definition"
            : "invalid_tool_resume",
          type: "invalid_request_error",
        },
      });
      return;
    }
    const outputs = resumePlan.outputs;
    if (!outputs.every((output) => record.session.hasCall(output.callId))) {
      json(res, 409, {
        error: {
          message:
            "Response tool session is missing or expired; resend the original input",
          code: "tool_session_expired",
          type: "invalid_request_error",
        },
      });
      return;
    }
    if (
      !outputs.every((output) =>
        record.session.matchesOutput(output),
      )
    ) {
      json(res, 400, {
        error: {
          message:
            "function_call_output namespace/name does not match the pending tool call",
          code: "invalid_tool_identity",
          type: "invalid_request_error",
        },
      });
      return;
    }
    if (
      Object.hasOwn(body, "tools") &&
      !outputs.every((output) =>
        record.session.matchesDefinition(output.callId, parsedTools),
      )
    ) {
      json(res, 400, {
        error: {
          message:
            "Responses tool definitions do not match the pending tool call identity",
          code: "invalid_tool_definition",
          type: "invalid_request_error",
        },
      });
      return;
    }
    const configDir = record.configDir;
    logAccountAssigned(configDir);
    reportRequestStart(configDir);
    const startedAt = Date.now();
    const abortController = new AbortController();
    abortOnClientDisconnect(res, abortController);
    abortController.signal.addEventListener(
      "abort",
      () => void record.session.close(),
      { once: true },
    );
    try {
      const result = await writeStructuredResponseTurn({
        res,
        body,
        stream: !!body.stream,
        id,
        createdAt,
        model: displayModel,
        previousResponseId,
        promptLength: prompt.length,
        run: async (listener) => {
          const turn = await ctx.toolSessions.resume(
            record,
            outputs,
            listener,
            () =>
              appendResponsesToolOutputs(
                record.responsesReplay!,
                outputs,
              ),
          );
          if (turn.status === "tool_calls") {
            ctx.toolSessions.aliasResponse(record, id);
          }
          return turn;
        },
        onResult: (turn) =>
          appendResponsesToolResult(record.responsesReplay!, turn),
      });
      reportRequestSuccess(configDir, Date.now() - startedAt);
      logTrafficResponse(
        config.verbose,
        model ?? cursorModel,
        result.text,
        !!body.stream,
      );
    } catch (error) {
      reportRequestError(configDir, Date.now() - startedAt);
      if (!res.headersSent) {
        json(res, error instanceof ToolSessionError ? error.status : 500, {
          error: {
            message: error instanceof Error ? error.message : String(error),
            code:
              error instanceof ToolSessionError
                ? error.code
                : "tool_session_error",
          },
        });
      } else if (!res.writableEnded) {
        writeResponseEvent(res, "response.failed", {
          response: {
            id,
            object: "response",
            status: "failed",
            error: {
              message: error instanceof Error ? error.message : String(error),
              code: "tool_session_error",
            },
          },
        });
        res.end();
      }
    } finally {
      reportRequestEnd(configDir);
      logAccountStats(config.verbose, getAccountStats());
    }
    return;
  }

  const effectiveChatOnly =
    mode === "ask"
      ? config.chatOnlyWorkspace
      : config.chatOnlyWorkspaceExplicit && config.chatOnlyWorkspace;

  const headerWs = req.headers["x-cursor-workspace"];
  let workspaceDir: string;
  let tempDir: string | undefined;
  try {
    const ws = resolveWorkspace(config, headerWs, effectiveChatOnly);
    workspaceDir = ws.workspaceDir;
    tempDir = ws.tempDir;
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Invalid workspace";
    json(res, 400, { error: { message: msg, code: "invalid_workspace" } });
    return;
  }

  const agentPrompt = config.contextPreamble
    ? `${buildBridgeContextPreamble({
        headers: req.headers,
        bridgeWorkspaceBase: config.workspace,
        agentWorkspaceDir: workspaceDir,
        isolatedChatOnly: tempDir !== undefined,
        cursorMode: mode,
        contextExtra: config.contextExtra,
      })}${BRIDGE_AGENT_PROMPT_SEPARATOR}${prompt}`
    : prompt;

  const fixedArgs = buildAgentFixedArgs(
    config,
    workspaceDir,
    cursorModel,
    !!body.stream,
    mode,
    effectiveChatOnly,
  );
  const promptViaStdin = config.promptViaStdin || config.useAcp;
  const fit = promptViaStdin
    ? {
        ok: true as const,
        args: fixedArgs,
        truncated: false,
        originalLength: agentPrompt.length,
        finalPromptLength: agentPrompt.length,
      }
    : fitPromptToWinCmdline(config.agentBin, fixedArgs, agentPrompt, {
        maxCmdline: config.winCmdlineMax,
        platform: process.platform,
        cwd: workspaceDir,
      });
  if (!fit.ok) {
    json(res, 500, {
      error: {
        message: fit.error,
        code: "windows_cmdline_limit",
        type: "api_error",
      },
    });
    return;
  }
  if (fit.truncated) {
    warnPromptTruncated(fit.originalLength, fit.finalPromptLength);
  }
  // When the prompt is delivered via stdin (or ACP), keep it OUT of argv,
  // otherwise a long prompt still blows past the kernel ARG_MAX (spawn E2BIG
  // on Linux). fit.args appends the full prompt for the argv path only.
  const cmdArgs = promptViaStdin ? fixedArgs : fit.args;

  const itemId = `msg_${randomUUID().replace(/-/g, "")}`;
  const promptForAgent =
    promptViaStdin ? agentPrompt : undefined;
  const truncatedHeaders = fit.truncated
    ? { "X-Cursor-Proxy-Prompt-Truncated": "true" }
    : undefined;

  if (structuredToolStart) {
    const configDir = getNextAccountConfigDir();
    logAccountAssigned(configDir);
    reportRequestStart(configDir);
    const startedAt = Date.now();
    const abortController = new AbortController();
    abortOnClientDisconnect(res, abortController);
    let record: ToolSessionRecord | undefined;
    try {
      const session = await startAgentToolSession({
        config,
        workspaceDir,
        effectiveChatOnly,
        cmdArgs,
        prompt: agentPrompt,
        tools: selectedTools,
        tempDir,
        configDir,
        signal: abortController.signal,
        modelDisplayName: modelCatalogName,
        requireToolCall,
        maxParallelToolCalls,
      });
      record = ctx.toolSessions.createRecord({
        api: "responses",
        ownerKey,
        model: displayModel ?? cursorModel,
        configDir,
        session,
      });
      record.responsesReplay = createResponsesToolReplayState({
        input: body.input,
        controls: controlValues,
      });
      abortController.signal.addEventListener(
        "abort",
        () => void session.close(),
        { once: true },
      );
      const result = await writeStructuredResponseTurn({
        res,
        body,
        stream: !!body.stream,
        id,
        createdAt,
        model: displayModel,
        previousResponseId: body.previous_response_id,
        promptLength: agentPrompt.length,
        run: async (listener) => {
          const turn = await ctx.toolSessions.collect(record!, listener);
          if (turn.status === "tool_calls") {
            ctx.toolSessions.aliasResponse(record!, id);
          }
          return turn;
        },
        onResult: (turn) =>
          appendResponsesToolResult(record!.responsesReplay!, turn),
      });
      reportRequestSuccess(configDir, Date.now() - startedAt);
      if (
        result.status === "completed" &&
        result.stderr &&
        isRateLimited(result.stderr)
      ) {
        reportRateLimit(configDir, 60_000);
      }
      logTrafficResponse(
        config.verbose,
        model ?? cursorModel,
        result.text,
        !!body.stream,
      );
    } catch (error) {
      if (record) {
        ctx.toolSessions.remove(record);
        await record.session.close().catch(() => undefined);
      }
      reportRequestError(configDir, Date.now() - startedAt);
      if (!res.headersSent) {
        json(res, 500, {
          error: {
            message: error instanceof Error ? error.message : String(error),
            code: "tool_session_error",
            type: "api_error",
          },
        });
      } else if (!res.writableEnded) {
        writeResponseEvent(res, "response.failed", {
          response: {
            id,
            object: "response",
            status: "failed",
            error: {
              message: error instanceof Error ? error.message : String(error),
              code: "tool_session_error",
            },
          },
        });
        res.end();
      }
    } finally {
      reportRequestEnd(configDir);
      logAccountStats(config.verbose, getAccountStats());
    }
    return;
  }

  if (body.stream) {
    const configDir = getNextAccountConfigDir();
    logAccountAssigned(configDir);
    reportRequestStart(configDir);
    const streamStart = Date.now();

    const abortController = new AbortController();
    abortOnClientDisconnect(res, abortController);

    writeSseHeaders(res, truncatedHeaders);
    const stopKeepalive = startSseKeepalive(res);
    res.on("error", () => {
      /* client disconnected mid-stream */
    });

    const initialResponse = createResponseObject({
      body,
      id,
      itemId,
      createdAt,
      model: displayModel,
      status: "in_progress",
      text: "",
      promptTokens: Math.max(1, Math.round(agentPrompt.length / 4)),
      completionTokens: 0,
    });
    writeResponseEvent(res, "response.created", { response: initialResponse });
    writeResponseEvent(res, "response.output_item.added", {
      response_id: id,
      output_index: 0,
      item: createOutputItem(itemId, "in_progress", ""),
    });
    writeResponseEvent(res, "response.content_part.added", {
      response_id: id,
      item_id: itemId,
      output_index: 0,
      content_index: 0,
      part: { type: "output_text", text: "", annotations: [] },
    });

    const writeChunk = (chunk: string, accumulated: string) => {
      writeResponseEvent(res, "response.output_text.delta", {
        response_id: id,
        item_id: itemId,
        output_index: 0,
        content_index: 0,
        delta: chunk,
      });
      return accumulated + chunk;
    };

    const finishStream = (accumulated: string) => {
      logTrafficResponse(config.verbose, model ?? cursorModel, accumulated, true);
      const promptTokens = Math.max(1, Math.round(agentPrompt.length / 4));
      const completionTokens = Math.max(1, Math.round(accumulated.length / 4));
      const completedItem = createOutputItem(itemId, "completed", accumulated);
      writeResponseEvent(res, "response.output_text.done", {
        response_id: id,
        item_id: itemId,
        output_index: 0,
        content_index: 0,
        text: accumulated,
      });
      writeResponseEvent(res, "response.content_part.done", {
        response_id: id,
        item_id: itemId,
        output_index: 0,
        content_index: 0,
        part: completedItem.content[0],
      });
      writeResponseEvent(res, "response.output_item.done", {
        response_id: id,
        output_index: 0,
        item: completedItem,
      });
      writeResponseEvent(res, "response.completed", {
        response: createResponseObject({
          body,
          id,
          itemId,
          createdAt,
          model: displayModel,
          status: "completed",
          text: accumulated,
          promptTokens,
          completionTokens,
        }),
      });
      res.write("data: [DONE]\n\n");
    };

    if (config.useAcp && typeof promptForAgent === "string") {
      let accumulated = "";
      runAgentStream(
        config,
        workspaceDir,
        effectiveChatOnly,
        cmdArgs,
        (chunk) => {
          accumulated = writeChunk(chunk, accumulated);
        },
        tempDir,
        promptForAgent,
        configDir,
        abortController.signal,
        modelCatalogName,
      )
        .then(({ code, stderr: stderrOut }) => {
          stopKeepalive();
          const latencyMs = Date.now() - streamStart;
          reportRequestEnd(configDir);

          if (stderrOut && isRateLimited(stderrOut)) {
            reportRateLimit(configDir, 60000);
          }

          if (abortController.signal.aborted) {
            /* client disconnected — do not count as success or failure */
          } else if (code !== 0) {
            reportRequestError(configDir, latencyMs);
            const publicMsg = logAgentError(
              config.sessionsLogPath,
              method,
              pathname,
              remoteAddress,
              code,
              stderrOut,
            );
            writeResponseEvent(res, "error", {
              error: { message: publicMsg, code: "cursor_cli_error" },
            });
            res.write("data: [DONE]\n\n");
            logAccountStats(config.verbose, getAccountStats());
            res.end();
            return;
          } else {
            reportRequestSuccess(configDir, latencyMs);
          }
          logAccountStats(config.verbose, getAccountStats());
          finishStream(accumulated);
          res.end();
        })
        .catch((err) => {
          stopKeepalive();
          reportRequestEnd(configDir);
          if (!abortController.signal.aborted) {
            reportRequestError(configDir, Date.now() - streamStart);
            writeResponseEvent(res, "error", {
              error: {
                message:
                  "The Cursor agent stream failed. See server logs for details.",
                code: "cursor_cli_error",
              },
            });
            res.write("data: [DONE]\n\n");
          }
          console.error(
            `[${new Date().toISOString()}] Agent stream error:`,
            err,
          );
          res.end();
        });
      return;
    }

    let accumulated = "";
    const parseLine = createStreamParser(
      (text) => {
        accumulated = writeChunk(text, accumulated);
      },
      () => {
        finishStream(accumulated);
      },
    );

    runAgentStream(
      config,
      workspaceDir,
      effectiveChatOnly,
      cmdArgs,
      parseLine,
      tempDir,
      promptForAgent,
      configDir,
      abortController.signal,
      modelCatalogName,
    )
      .then(({ code, stderr: stderrOut }) => {
        stopKeepalive();
        const latencyMs = Date.now() - streamStart;
        reportRequestEnd(configDir);

        if (stderrOut && isRateLimited(stderrOut)) {
          reportRateLimit(configDir, 60000);
        }

        if (abortController.signal.aborted) {
          /* client disconnected — do not count as success or failure */
        } else if (code !== 0) {
          reportRequestError(configDir, latencyMs);
          logAgentError(
            config.sessionsLogPath,
            method,
            pathname,
            remoteAddress,
            code,
            stderrOut,
          );
        } else {
          reportRequestSuccess(configDir, latencyMs);
        }
        logAccountStats(config.verbose, getAccountStats());
        res.end();
      })
      .catch((err) => {
        stopKeepalive();
        reportRequestEnd(configDir);
        if (!abortController.signal.aborted) {
          reportRequestError(configDir, Date.now() - streamStart);
        }
        console.error(
          `[${new Date().toISOString()}] Agent stream error:`,
          err,
        );
        res.end();
      });
    return;
  }

  const configDir = getNextAccountConfigDir();
  logAccountAssigned(configDir);
  reportRequestStart(configDir);
  const syncStart = Date.now();

  const abortController = new AbortController();
  abortOnClientDisconnect(res, abortController);

  const out = await runAgentSync(
    config,
    workspaceDir,
    effectiveChatOnly,
    cmdArgs,
    tempDir,
    promptForAgent,
    configDir,
    abortController.signal,
    modelCatalogName,
  );
  const syncLatency = Date.now() - syncStart;
  reportRequestEnd(configDir);

  if (out.stderr && isRateLimited(out.stderr)) {
    reportRateLimit(configDir, 60000);
  }

  if (out.code !== 0) {
    reportRequestError(configDir, syncLatency);
    logAccountStats(config.verbose, getAccountStats());
    const errMsg = logAgentError(
      config.sessionsLogPath,
      method,
      pathname,
      remoteAddress,
      out.code,
      out.stderr,
    );
    json(res, 500, {
      error: { message: errMsg, code: "cursor_cli_error" },
    });
    return;
  }

  reportRequestSuccess(configDir, syncLatency);
  const content = out.stdout.trim();
  logTrafficResponse(config.verbose, model ?? cursorModel, content, false);

  const promptTokens = Math.max(1, Math.round(agentPrompt.length / 4));
  const completionTokens = Math.max(1, Math.round(content.length / 4));

  logAccountStats(config.verbose, getAccountStats());
  json(
    res,
    200,
    createResponseObject({
      body,
      id,
      itemId,
      createdAt,
      model: displayModel,
      status: "completed",
      text: content,
      promptTokens,
      completionTokens,
    }),
    truncatedHeaders,
  );
}
