"use client";

import { useEffect, useRef, useState } from "react";
import { X, Trash2 } from "lucide-react";
import { useChatStore } from "@/lib/chat-store";
import { useChatStream } from "@/hooks/use-chat-stream";
import { getChatModels } from "@/lib/api";
import { ChatInput } from "@/components/chat/chat-input";
import { MessageBubble } from "@/components/chat/message-bubble";
import { ToolResultCard } from "@/components/chat/tool-result-card";
import { SuggestedActions } from "@/components/chat/suggested-actions";
import { cn } from "@/lib/utils";

export function ChatPanel() {
  const { chatOpen, messages, isStreaming, panelWidth } = useChatStore();
  const clearChat = useChatStore((s) => s.clearChat);
  const closeChat = useChatStore((s) => s.closeChat);
  const { sendMessage, abort } = useChatStream();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [chatModel, setChatModel] = useState<{ provider: string; model: string } | null>(null);

  useEffect(() => {
    if (!chatOpen) return;
    getChatModels()
      .then((res) => setChatModel({ provider: res.provider, model: res.default }))
      .catch(() => {});
  }, [chatOpen]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === ".") {
        e.preventDefault();
        useChatStore.getState().toggleChat();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  return (
    <div
      className={cn(
        "fixed top-0 right-0 h-full z-40 flex flex-col transition-transform duration-300 ease-out",
        chatOpen ? "translate-x-0" : "translate-x-full"
      )}
      style={{ width: panelWidth }}
    >
      {/* Animated gradient left border */}
      <div
        className="absolute left-0 top-0 bottom-0 w-[2px] z-20"
        style={{
          background:
            "linear-gradient(180deg, #007AFF, #34C759, #AF82FF, #007AFF)",
          backgroundSize: "100% 300%",
          animation: "border-flow 4s ease infinite",
        }}
      />

      {/* Glass background */}
      <div className="absolute inset-0 bg-white/55 backdrop-blur-[40px] saturate-[180%] border-l border-black/[0.06]" />

      {/* Content */}
      <div className="relative z-10 flex flex-col h-full">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-black/[0.06]">
          <div className="flex items-center gap-2">
            <h2 className="text-[14px] font-semibold text-[#1D1D1F]">
              Forge AI
            </h2>
            {chatModel && (
              <span className="text-[10px] font-medium text-[#86868B] bg-black/[0.04] px-1.5 py-0.5 rounded-md">
                {chatModel.model}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={clearChat}
              className="p-1.5 rounded-lg text-[#86868B] hover:text-[#1D1D1F] hover:bg-black/[0.04] transition-colors"
              aria-label="Clear chat"
              title="Clear conversation"
            >
              <Trash2 className="size-3.5" />
            </button>
            <button
              onClick={closeChat}
              className="p-1.5 rounded-lg text-[#86868B] hover:text-[#1D1D1F] hover:bg-black/[0.04] transition-colors"
              aria-label="Close panel"
            >
              <X className="size-4" />
            </button>
          </div>
        </header>

        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-4 py-4 space-y-3"
        >
          {messages.length === 0 ? (
            <SuggestedActions onSelect={sendMessage} />
          ) : (
            messages.map((msg) => (
              <div key={msg.id}>
                <MessageBubble message={msg} />
                {msg.toolResults?.map((tr, i) => (
                  <ToolResultCard key={`${msg.id}-tr-${i}`} toolResult={tr} />
                ))}
              </div>
            ))
          )}
        </div>

        {/* Input */}
        <ChatInput
          onSend={sendMessage}
          isStreaming={isStreaming}
          onAbort={abort}
        />
      </div>
    </div>
  );
}
