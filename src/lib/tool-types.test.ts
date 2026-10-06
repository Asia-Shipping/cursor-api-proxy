import { describe, expect, it } from "vitest";

import {
  anthropicToolOutputs,
  chatToolOutputs,
  parseAnthropicFunctionTools,
  parseOpenAiFunctionTools,
  resolveToolChoice,
  responsesToolOutputs,
} from "./tool-types.js";

describe("tool normalization", () => {
  it("accepts Chat, Responses, legacy, and Anthropic function schemas", () => {
    expect(
      parseOpenAiFunctionTools([
        {
          type: "function",
          function: {
            name: "chat",
            parameters: { type: "object", properties: { x: {} } },
          },
        },
        {
          type: "function",
          name: "response",
          parameters: { type: "object", properties: { y: {} } },
        },
      ]),
    ).toEqual([
      {
        name: "chat",
        description: undefined,
        inputSchema: { type: "object", properties: { x: {} } },
      },
      {
        name: "response",
        description: undefined,
        inputSchema: { type: "object", properties: { y: {} } },
      },
    ]);
    expect(
      parseOpenAiFunctionTools(undefined, [
        {
          name: "legacy",
          parameters: { type: "object", properties: {} },
        },
      ])[0].name,
    ).toBe("legacy");
    expect(
      parseAnthropicFunctionTools([
        {
          name: "anthropic",
          input_schema: { type: "object", properties: {} },
        },
      ])[0].inputSchema,
    ).toEqual({ type: "object", properties: {} });
  });

  it("rejects unsupported and duplicate tools", () => {
    expect(() =>
      parseOpenAiFunctionTools([{ type: "web_search" }]),
    ).toThrow(/Unsupported tool type/);
    expect(() =>
      parseOpenAiFunctionTools([
        { type: "function", name: "same" },
        { type: "function", function: { name: "same" } },
      ]),
    ).toThrow(/Duplicate/);
  });

  it("preserves namespace identity when bare names collide", () => {
    const tools = parseOpenAiFunctionTools(
      [
        { type: "function", name: "search" },
        {
          type: "namespace",
          name: "mcp__docs",
          description: "Search documentation.",
          tools: [{ type: "function", name: "search" }],
        },
        {
          type: "namespace",
          name: "mcp__wiki",
          tools: [{ type: "function", name: "search" }],
        },
      ],
      undefined,
      { allowNamespaces: true },
    );

    expect(tools.map(({ namespace, name }) => [namespace, name])).toEqual([
      [undefined, "search"],
      ["mcp__docs", "search"],
      ["mcp__wiki", "search"],
    ]);
    expect(tools[1].namespaceDescription).toBe("Search documentation.");
  });

  it("rejects malformed namespace groups and unsupported nested tools", () => {
    expect(() =>
      parseOpenAiFunctionTools(
        [{ type: "namespace", name: "empty" }],
        undefined,
        { allowNamespaces: true },
      ),
    ).toThrow(/missing tools array/);
    expect(() =>
      parseOpenAiFunctionTools(
        [
          {
            type: "namespace",
            name: "custom",
            tools: [{ type: "custom", name: "patch" }],
          },
        ],
        undefined,
        { allowNamespaces: true },
      ),
    ).toThrow(/Unsupported tool type in namespace/);
    expect(() =>
      parseOpenAiFunctionTools([
        { type: "namespace", name: "chat-only", tools: [] },
      ]),
    ).toThrow(/only supported by the Responses API/);
  });

  it("implements none, required, named, and no-parallel choices", () => {
    const tools = [
      {
        name: "one",
        inputSchema: { type: "object", properties: {} },
      },
      {
        name: "two",
        inputSchema: { type: "object", properties: {} },
      },
    ];
    expect(resolveToolChoice(tools, "none").tools).toEqual([]);
    expect(resolveToolChoice(tools, "required").instruction).toMatch(/must/);
    expect(
      resolveToolChoice(tools, {
        type: "function",
        function: { name: "two" },
      }).tools.map((tool) => tool.name),
    ).toEqual(["two"]);
    expect(
      resolveToolChoice(tools, "auto", { parallelToolCalls: false })
        .instruction,
    ).toMatch(/at most one/i);
    expect(() =>
      resolveToolChoice(tools, { type: "tool", name: "missing" }),
    ).toThrow(/Unknown/);
  });

  it("requires namespace-qualified choices when function names collide", () => {
    const tools = [
      { name: "lookup", inputSchema: {}, namespace: "alpha" },
      { name: "lookup", inputSchema: {}, namespace: "beta" },
    ];
    expect(() => resolveToolChoice(tools, { type: "function", name: "lookup" }))
      .toThrow(/Ambiguous.*specify namespace/);
    expect(
      resolveToolChoice(tools, {
        type: "function",
        namespace: "beta",
        name: "lookup",
      }).tools,
    ).toEqual([tools[1]]);
    expect(
      resolveToolChoice(tools, {
        type: "function",
        function: { namespace: "alpha", name: "lookup" },
      }).tools,
    ).toEqual([tools[0]]);
  });
});

describe("tool output correlation", () => {
  it("reads Chat tool_call_id", () => {
    expect(
      chatToolOutputs([
        { role: "user", content: "x" },
        { role: "tool", tool_call_id: "call_1", content: { ok: true } },
      ]),
    ).toEqual([{ callId: "call_1", output: '{"ok":true}' }]);
  });

  it("reads Responses function_call_output", () => {
    expect(
      responsesToolOutputs([
        {
          type: "function_call_output",
          call_id: "call_2",
          output: "done",
        },
      ]),
    ).toEqual([{ callId: "call_2", output: "done" }]);
    expect(
      responsesToolOutputs([
        {
          type: "function_call_output",
          call_id: "call_3",
          namespace: "mcp__docs",
          name: "search",
          output: "done",
        },
      ]),
    ).toEqual([
      {
        callId: "call_3",
        namespace: "mcp__docs",
        name: "search",
        output: "done",
      },
    ]);
    expect(() =>
      responsesToolOutputs([
        {
          type: "function_call_output",
          call_id: "call_4",
          namespace: "mcp__docs",
          output: "done",
        },
      ]),
    ).toThrow(/namespace is missing name/);
  });

  it("reads Anthropic tool_result and error state", () => {
    expect(
      anthropicToolOutputs([
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "call_3",
              content: [{ type: "text", text: "failed" }],
              is_error: true,
            },
          ],
        },
      ]),
    ).toEqual([
      { callId: "call_3", output: "failed", isError: true },
    ]);
  });
});
