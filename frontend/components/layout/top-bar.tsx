"use client";

import { Search, Sparkles } from "lucide-react";
import { useChatStore } from "@/lib/chat-store";

interface TopBarProps {
  readonly title: string;
  readonly status?: "running" | "complete";
}

export function TopBar({ title, status }: TopBarProps) {
  const toggleChat = useChatStore((s) => s.toggleChat);

  return (
    <header className="flex h-[54px] shrink-0 items-center justify-between glass border-b border-black/[0.06] px-7">
      <div className="flex items-center gap-3">
        <h1 className="text-[16px] font-semibold text-[#1D1D1F]">{title}</h1>
        {status === "running" && (
          <span className="inline-flex items-center gap-1.5 text-[10px] font-medium px-2 py-0.5 rounded-full bg-[#007AFF]/10 text-[#007AFF]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#007AFF] animate-pulse-dot" />
            Running
          </span>
        )}
        {status === "complete" && (
          <span className="inline-flex items-center gap-1.5 text-[10px] font-medium px-2 py-0.5 rounded-full bg-[#34C759]/10 text-[#34C759]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#34C759]" />
            Complete
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={toggleChat}
          className="flex items-center gap-[6px] rounded-lg bg-gradient-to-r from-[#007AFF]/10 to-[#AF82FF]/10 border border-[#007AFF]/20 px-3 py-[5px] text-[12px] text-[#007AFF] font-medium transition-all hover:shadow-[0_2px_8px_rgba(0,122,255,0.2)]"
          aria-label="Toggle AI Assistant"
        >
          <Sparkles className="size-[13px]" />
          <span>Forge AI</span>
          <kbd className="text-[10px] font-semibold px-[5px] py-[1px] bg-[#007AFF]/10 rounded">⌘.</kbd>
        </button>

        <button
          className="flex items-center gap-[6px] rounded-lg bg-black/[0.04] border border-black/[0.06] px-3 py-[5px] text-[12px] text-[#86868B] transition-colors hover:bg-black/[0.06]"
          aria-label="Search"
        >
          <Search className="size-[13px]" />
          <span>Search</span>
          <kbd className="text-[10px] font-semibold px-[5px] py-[1px] bg-black/[0.06] rounded">⌘K</kbd>
        </button>
      </div>
    </header>
  );
}
