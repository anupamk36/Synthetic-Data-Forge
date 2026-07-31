import { create } from "zustand";
import type { ChatMessage, ToolCall, ToolResult } from "./chat-types";

interface ChatStore {
  chatOpen: boolean;
  sessionId: string;
  messages: ChatMessage[];
  isStreaming: boolean;
  panelWidth: number;

  toggleChat: () => void;
  openChat: () => void;
  closeChat: () => void;
  setSessionId: (id: string) => void;
  addMessage: (msg: ChatMessage) => void;
  updateLastAssistantContent: (content: string) => void;
  appendToolCall: (messageId: string, toolCall: ToolCall) => void;
  appendToolResult: (messageId: string, toolResult: ToolResult) => void;
  setStreaming: (s: boolean) => void;
  setPanelWidth: (w: number) => void;
  clearChat: () => void;
}

function generateId(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

export const useChatStore = create<ChatStore>((set) => ({
  chatOpen: false,
  sessionId: generateId(),
  messages: [],
  isStreaming: false,
  panelWidth: 400,

  toggleChat: () => set((s) => ({ chatOpen: !s.chatOpen })),
  openChat: () => set({ chatOpen: true }),
  closeChat: () => set({ chatOpen: false }),
  setSessionId: (sessionId) => set({ sessionId }),

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  updateLastAssistantContent: (content) =>
    set((s) => {
      const msgs = [...s.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content, isStreaming: true };
      }
      return { messages: msgs };
    }),

  appendToolCall: (messageId, toolCall) =>
    set((s) => {
      const msgs = s.messages.map((m) =>
        m.id === messageId
          ? { ...m, toolCalls: [...(m.toolCalls || []), toolCall] }
          : m
      );
      return { messages: msgs };
    }),

  appendToolResult: (messageId, toolResult) =>
    set((s) => {
      const msgs = s.messages.map((m) =>
        m.id === messageId
          ? { ...m, toolResults: [...(m.toolResults || []), toolResult] }
          : m
      );
      return { messages: msgs };
    }),

  setStreaming: (isStreaming) => set({ isStreaming }),
  setPanelWidth: (panelWidth) => set({ panelWidth: Math.max(320, Math.min(600, panelWidth)) }),

  clearChat: () =>
    set({ messages: [], sessionId: generateId() }),
}));
