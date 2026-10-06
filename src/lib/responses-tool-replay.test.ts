import { describe, expect, it } from "vitest";

import {
  appendResponsesToolOutputs,
  appendResponsesToolResult,
  assertResponsesToolControls,
  createResponsesToolReplayState,
  prepareResponsesToolResume,
  type ResponsesToolControlKey,
  type ResponsesToolControlValues,
} from "./responses-tool-replay.js";
import type { ToolTurnResult } from "./acp-tool-session.js";
import type { ClientToolOutput, PendingClientToolCall } from "./tool-types.js";

const originalInput = [
  {
    type: "message",
    role: "developer",
    content: [{ type: "input_text", text: "Use the available lookup tools." }],
  },
  {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: "Look up both Alpha and Beta." }],
  },
];

const controls: ResponsesToolControlValues = {
  model: "composer-2.5",
  mode: "ask",
  instructions: "Keep the original request and tool definitions active.",
  tools: [
    {
      type: "namespace",
      name: "alpha",
      tools: [{ type: "function", name: "lookup", parameters: { type: "object" } }],
    },
    {
      type: "namespace",
      name: "beta",
      tools: [{ type: "function", name: "lookup", parameters: { type: "object" } }],
    },
  ],
  tool_choice: "auto",
  parallel_tool_calls: true,
  reasoning: { effort: "medium" },
};

const alphaCall: PendingClientToolCall = {
  callId: "call_alpha",
  itemId: "item_alpha",
  name: "lookup",
  namespace: "alpha",
  arguments: '{"query":"Alpha"}',
};

const betaCall: PendingClientToolCall = {
  callId: "call_beta",
  itemId: "item_beta",
  name: "lookup",
  namespace: "beta",
  arguments: '{"query":"Beta"}',
};

const alphaOutput: ClientToolOutput = {
  callId: alphaCall.callId,
  name: alphaCall.name,
  namespace: alphaCall.namespace,
  output: "Alpha result",
};

const betaOutput: ClientToolOutput = {
  callId: betaCall.callId,
  name: betaCall.name,
  namespace: betaCall.namespace,
  output: "Beta result",
};

const toolTurn: ToolTurnResult = {
  status: "tool_calls",
  text: "Starting both lookups.",
  reasoning: "",
  toolCalls: [alphaCall, betaCall],
};

function functionCallItem(call: PendingClientToolCall) {
  return {
    id: call.itemId,
    type: "function_call",
    status: "completed",
    call_id: call.callId,
    namespace: call.namespace,
    name: call.name,
    arguments: call.arguments,
    annotations: [{ type: "call_annotation" }],
  };
}

function functionOutputItem(output: ClientToolOutput) {
  return {
    id: `wire_${output.callId}`,
    type: "function_call_output",
    status: "completed",
    call_id: output.callId,
    name: output.name,
    namespace: output.namespace,
    output: output.output,
    annotations: [{ type: "output_annotation" }],
  };
}

function createReplayState() {
  const state = createResponsesToolReplayState({
    input: originalInput,
    controls,
  });
  appendResponsesToolResult(state, toolTurn);
  appendResponsesToolOutputs(state, [alphaOutput]);
  return state;
}

function replayInput(opts: {
  input?: unknown[];
  calls?: ReturnType<typeof functionCallItem>[];
  outputs?: ReturnType<typeof functionOutputItem>[];
  extra?: unknown[];
} = {}) {
  const calls = opts.calls ?? [functionCallItem(alphaCall), functionCallItem(betaCall)];
  const outputs = opts.outputs ?? [functionOutputItem(alphaOutput), functionOutputItem(betaOutput)];
  return [
    ...(opts.input ?? originalInput),
    {
      id: "wire_assistant_message",
      type: "message",
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: "Starting both lookups.",
          annotations: [{ type: "text_annotation" }],
        },
      ],
    },
    ...calls,
    ...outputs,
    ...(opts.extra ?? []),
  ];
}

describe("Responses ACP tool replay", () => {
  it("preserves the original developer/user prefix and delivers only the pending namespaced call", () => {
    const state = createReplayState();

    const plan = prepareResponsesToolResume({
      input: replayInput(),
      state,
    });

    expect(plan.kind).toBe("replay");
    expect(plan.outputs).toEqual([betaOutput]);
  });

  it("ignores replayed wire ids, status, and annotations while matching resolved history", () => {
    const state = createReplayState();
    const calls = [
      { ...functionCallItem(alphaCall), id: "new_alpha_item", status: "in_progress", annotations: [{ type: "new" }] },
      { ...functionCallItem(betaCall), id: "new_beta_item", status: "in_progress", annotations: [{ type: "new" }] },
    ];
    const outputs = [
      { ...functionOutputItem(alphaOutput), id: "new_alpha_output", status: "in_progress", annotations: [{ type: "new" }] },
      { ...functionOutputItem(betaOutput), id: "new_beta_output", status: "in_progress", annotations: [{ type: "new" }] },
    ];

    const plan = prepareResponsesToolResume({
      input: replayInput({ calls, outputs }),
      state,
    });

    expect(plan).toEqual({ kind: "replay", outputs: [betaOutput] });
  });

  it("supports outputs-only continuation after ignoring an exact previously accepted result", () => {
    const state = createReplayState();

    const plan = prepareResponsesToolResume({
      input: [functionOutputItem(alphaOutput), functionOutputItem(betaOutput)],
      state,
    });

    expect(plan).toEqual({ kind: "outputs", outputs: [betaOutput] });
  });

  it.each([
    ["developer instructions", [
      originalInput[0],
      { ...originalInput[1], content: [{ type: "input_text", text: "A different request." }] },
    ]],
    ["user content", [
      originalInput[0],
      { ...originalInput[1], content: [{ type: "input_text", text: "Only look up Alpha." }] },
    ]],
  ] as Array<[string, unknown[]]>)('rejects changed original %s', (_description, changedInput) => {
    expect(() =>
      prepareResponsesToolResume({
        input: replayInput({ input: changedInput }),
        state: createReplayState(),
      }),
    ).toThrow(/original input and active ACP tool history/);
  });

  it("rejects new user content appended after the original request and tool history", () => {
    expect(() =>
      prepareResponsesToolResume({
        input: replayInput({
          extra: [
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "Also check Gamma." }],
            },
          ],
        }),
        state: createReplayState(),
      }),
    ).toThrow(/append only new function_call_output/);
  });

  it("rejects a previously resolved output whose value changes", () => {
    const changedAlpha = { ...alphaOutput, output: "Forged Alpha result" };

    expect(() =>
      prepareResponsesToolResume({
        input: replayInput({
          outputs: [functionOutputItem(changedAlpha), functionOutputItem(betaOutput)],
        }),
        state: createReplayState(),
      }),
    ).toThrow(/original input and active ACP tool history/);
  });

  it.each([
    ["call_id", { call_id: "different_call_id" }],
    ["namespace", { namespace: "beta" }],
    ["name", { name: "search" }],
    ["arguments", { arguments: '{"query":"Elsewhere"}' }],
  ] as Array<[string, Partial<ReturnType<typeof functionCallItem>>]>)('rejects a replayed function call with changed %s', (_field, change) => {
    const changedAlphaCall = { ...functionCallItem(alphaCall), ...change };

    expect(() =>
      prepareResponsesToolResume({
        input: replayInput({
          calls: [changedAlphaCall, functionCallItem(betaCall)],
        }),
        state: createReplayState(),
      }),
    ).toThrow(/original input and active ACP tool history/);
  });

  it.each([
    ["model", "composer-2"],
    ["mode", "agent"],
    ["instructions", "Changed instructions."],
    ["tools", [{ type: "function", name: "lookup" }]],
    ["tool_choice", "none"],
    ["parallel_tool_calls", false],
    ["reasoning", { effort: "high" }],
  ] as Array<[ResponsesToolControlKey, unknown]>)('rejects a changed %s control fingerprint', (key, value) => {
    expect(() =>
      assertResponsesToolControls(createReplayState(), { [key]: value }),
    ).toThrow(new RegExp(`Responses ${key} changed`));
  });

  it("rejects changed resolved output in outputs-only continuation", () => {
    const changedAlpha = { ...functionOutputItem(alphaOutput), output: "tampered" };

    expect(() =>
      prepareResponsesToolResume({
        input: [changedAlpha, functionOutputItem(betaOutput)],
        state: createReplayState(),
      }),
    ).toThrow(/Previously resolved function_call_output changed/);
  });
});
