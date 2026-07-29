export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
}

export interface ToolResult {
  tool: string;
  result: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
  timestamp: number;
  isStreaming?: boolean;
}

export interface ChatSSEEvent {
  event: "token" | "tool_call" | "tool_result" | "error" | "done";
  data: Record<string, unknown>;
}
