import { afterEach, describe, expect, it } from "vitest";

import { ClientToolBridge } from "./client-tool-bridge.js";
import type { ClientToolDefinition } from "./tool-types.js";

const definitions: ClientToolDefinition[] = [
  { name: "lookup", inputSchema: { type: "object", properties: {} } },
  {
    name: "lookup",
    namespace: "alpha",
    namespaceDescription: "Alpha lookup service.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "lookup",
    namespace: "beta",
    namespaceDescription: "Beta lookup service.",
    inputSchema: { type: "object", properties: {} },
  },
];

const bridges: ClientToolBridge[] = [];

async function connect(bridge: ClientToolBridge) {
  await bridge.start();
  const server = bridge.mcpServer;
  let sessionId: string | undefined;
  let nextRequestId = 1;

  const request = async (method: string, params: unknown) => {
    const response = await fetch(server.url, {
      method: "POST",
      headers: {
        accept: "application/json, text/event-stream",
        authorization: server.headers[0].value,
        "content-type": "application/json",
        "mcp-protocol-version": "2025-03-26",
        ...(sessionId ? { "mcp-session-id": sessionId } : {}),
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: nextRequestId++,
        method,
        params,
      }),
    });
    sessionId = response.headers.get("mcp-session-id") ?? sessionId;
    const payload = (await response.json()) as Record<string, any>;
    if (!response.ok || payload.error) {
      throw new Error(payload.error?.message ?? `MCP request failed: ${method}`);
    }
    return payload.result;
  };

  await request("initialize", {
    protocolVersion: "2025-03-26",
    capabilities: {},
    clientInfo: { name: "bridge-test", version: "1" },
  });
  const initialized = await fetch(server.url, {
    method: "POST",
    headers: {
      accept: "application/json, text/event-stream",
      authorization: server.headers[0].value,
      "content-type": "application/json",
      "mcp-protocol-version": "2025-03-26",
      ...(sessionId ? { "mcp-session-id": sessionId } : {}),
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "notifications/initialized",
      params: {},
    }),
  });
  if (!initialized.ok) throw new Error("MCP initialized notification failed");

  return { request, server };
}

afterEach(async () => {
  await Promise.all(bridges.splice(0).map((bridge) => bridge.close()));
});

describe("ClientToolBridge namespace identities", () => {
  it("uses stable safe aliases and maps calls back to namespace/name", async () => {
    const bridge = new ClientToolBridge(definitions);
    bridges.push(bridge);
    const { request } = await connect(bridge);
    const firstList = await request("tools/list", {});
    const listedTools = firstList.tools as Array<{
      name: string;
      description?: string;
    }>;

    expect(listedTools.map((tool) => tool.name)).toHaveLength(3);
    expect(new Set(listedTools.map((tool) => tool.name)).size).toBe(3);
    expect(listedTools[0].name).toBe("lookup");
    expect(listedTools[1].name).toMatch(/^tool_[a-f0-9]{40}$/);
    expect(listedTools[2].name).toMatch(/^tool_[a-f0-9]{40}$/);
    expect(listedTools[1].description).toContain("Namespace: alpha");
    expect(listedTools[2].description).toContain("Function: lookup");

    const duplicate = new ClientToolBridge(definitions);
    bridges.push(duplicate);
    const { request: duplicateRequest } = await connect(duplicate);
    const secondList = await duplicateRequest("tools/list", {});
    expect((secondList.tools as Array<{ name: string }>).map((tool) => tool.name))
      .toEqual(listedTools.map((tool) => tool.name));

    let notifyCalls!: () => void;
    const callsObserved = new Promise<void>((resolve) => {
      let count = 0;
      notifyCalls = () => {
        count += 1;
        if (count === 2) resolve();
      };
    });
    bridge.onCall(notifyCalls);
    const alphaPromise = request("tools/call", {
      name: listedTools[1].name,
      arguments: { query: "value" },
    });
    const betaPromise = request("tools/call", {
      name: listedTools[2].name,
      arguments: { query: "value" },
    });
    await callsObserved;
    const pending = bridge.pendingCalls();
    const alpha = pending.find((call) => call.namespace === "alpha");
    const beta = pending.find((call) => call.namespace === "beta");
    expect(alpha).toMatchObject({ namespace: "alpha", name: "lookup" });
    expect(beta).toMatchObject({ namespace: "beta", name: "lookup" });
    expect(JSON.parse(alpha!.arguments)).toEqual({ query: "value" });
    expect(() =>
      bridge.resolveOutputs([
        {
          callId: alpha!.callId,
          namespace: "beta",
          name: "lookup",
          output: "wrong target",
        },
      ]),
    ).toThrow(/identity does not match/);
    expect(bridge.has(alpha!.callId)).toBe(true);

    bridge.resolveOutputs([
      {
        callId: beta!.callId,
        namespace: "beta",
        name: "lookup",
        output: "beta result",
      },
      {
        callId: alpha!.callId,
        namespace: "alpha",
        name: "lookup",
        output: "alpha result",
      },
    ]);
    const [alphaResult, betaResult] = await Promise.all([
      alphaPromise,
      betaPromise,
    ]);
    expect(alphaResult.content[0].text).toBe("alpha result");
    expect(betaResult.content[0].text).toBe("beta result");
  });
});
