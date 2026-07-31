"use client";

import { useCallback, useRef } from "react";
import { useChatStore } from "@/lib/chat-store";
import type { ChatMessage } from "@/lib/chat-types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8100";

export function useChatStream() {
  const abortRef = useRef<AbortController | null>(null);
  const store = useChatStore;

  const sendMessage = useCallback(
    async (
      content: string,
      opts?: { provider?: string; model?: string }
    ) => {
      const {
        sessionId,
        addMessage,
        updateLastAssistantContent,
        appendToolCall,
        appendToolResult,
        setStreaming,
      } = store.getState();

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: Date.now(),
      };
      addMessage(userMsg);

      const assistantId = crypto.randomUUID();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        isStreaming: true,
      };
      addMessage(assistantMsg);
      setStreaming(true);

      abortRef.current = new AbortController();
      let accumulated = "";

      try {
        const res = await fetch(`${API_URL}/api/v1/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            message: content,
            provider: opts?.provider,
            model: opts?.model,
          }),
          signal: abortRef.current.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`Chat request failed: ${res.statusText}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentEventType = "token";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            if (trimmed.startsWith("event:")) {
              currentEventType = trimmed.slice(6).trim();
              continue;
            }

            if (trimmed.startsWith("data:")) {
              const rawData = trimmed.slice(5).trim();
              if (!rawData) continue;

              try {
                const parsed = JSON.parse(rawData);

                switch (currentEventType) {
                  case "token":
                    accumulated += parsed.content || "";
                    updateLastAssistantContent(accumulated);
                    break;

                  case "tool_call":
                    appendToolCall(assistantId, {
                      tool: parsed.tool,
                      args: parsed.args,
                    });
                    break;

                  case "tool_result":
                    appendToolResult(assistantId, {
                      tool: parsed.tool,
                      result: parsed.result,
                    });
                    break;

                  case "error": {
                    const errorMsg = parsed.message || "An error occurred";
                    accumulated += `\n\n**Error:** ${errorMsg}`;
                    updateLastAssistantContent(accumulated);
                    break;
                  }

                  case "done":
                    break;
                }
              } catch {
                // Skip malformed JSON lines
              }
            }
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          const { updateLastAssistantContent: update } = store.getState();
          update(
            (accumulated || "") +
              "\n\n**Error:** Failed to connect to the AI assistant."
          );
        }
      } finally {
        const { setStreaming: finish } = store.getState();
        finish(false);
        // Mark the message as no longer streaming
        const msgs = store.getState().messages;
        const lastMsg = msgs[msgs.length - 1];
        if (lastMsg?.role === "assistant" && lastMsg.isStreaming) {
          store.setState({
            messages: msgs.map((m) =>
              m.id === lastMsg.id ? { ...m, isStreaming: false } : m
            ),
          });
        }
      }
    },
    []
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
    store.getState().setStreaming(false);
  }, []);

  return { sendMessage, abort };
}
