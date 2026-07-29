"use client";

import { useState, useRef, useCallback } from "react";
import { Send, Square, Paperclip } from "lucide-react";
import { chatUpload } from "@/lib/api";
import { useChatStore } from "@/lib/chat-store";

interface ChatInputProps {
  onSend: (message: string) => void;
  isStreaming: boolean;
  onAbort: () => void;
}

export function ChatInput({ onSend, isStreaming, onAbort }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const sessionId = useChatStore((s) => s.sessionId);

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, isStreaming, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const result = await chatUpload(sessionId, file);
      onSend(
        `I've uploaded "${file.name}" (${result.rows} rows, columns: ${result.columns.join(", ")})`
      );
    } catch {
      onSend(`Failed to upload "${file.name}". Please try again.`);
    }
    e.target.value = "";
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  return (
    <div className="border-t border-black/[0.06] p-3">
      <div className="flex items-end gap-2 rounded-xl bg-white/60 border border-black/[0.06] px-3 py-2">
        <button
          onClick={() => fileRef.current?.click()}
          className="shrink-0 p-1.5 rounded-lg text-[#86868B] hover:text-[#1D1D1F] hover:bg-black/[0.04] transition-colors"
          aria-label="Upload file"
        >
          <Paperclip className="size-4" />
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.json,.jsonl,.parquet"
          onChange={handleFileChange}
          className="hidden"
        />

        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Ask Forge AI..."
          rows={1}
          className="flex-1 resize-none bg-transparent text-[13px] text-[#1D1D1F] placeholder:text-[#86868B] focus:outline-none"
          style={{ maxHeight: 120 }}
        />

        {isStreaming ? (
          <button
            onClick={onAbort}
            className="shrink-0 p-1.5 rounded-lg bg-[#FF3B30]/10 text-[#FF3B30] hover:bg-[#FF3B30]/20 transition-colors"
            aria-label="Stop"
          >
            <Square className="size-3.5" />
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!value.trim()}
            className="shrink-0 p-1.5 rounded-lg bg-[#007AFF] text-white disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[#0066D6] transition-colors"
            aria-label="Send"
          >
            <Send className="size-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
