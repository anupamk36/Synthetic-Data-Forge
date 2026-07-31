"use client";

import type { ChatMessage } from "@/lib/chat-types";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-2.5 text-[13px] leading-relaxed",
          isUser
            ? "bg-[#007AFF] text-white"
            : "bg-white/65 border border-black/[0.06] text-[#1D1D1F] shadow-sm backdrop-blur-sm"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none">
            <MarkdownLite text={message.content} />
            {message.isStreaming && (
              <span className="inline-block w-1.5 h-4 ml-0.5 bg-[#007AFF] rounded-sm animate-pulse" />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MarkdownLite({ text }: { text: string }) {
  if (!text) return null;

  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("### ")) {
      elements.push(
        <h4 key={i} className="font-semibold text-[13px] mt-3 mb-1">
          {formatInline(line.slice(4))}
        </h4>
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h3 key={i} className="font-semibold text-[14px] mt-3 mb-1">
          {formatInline(line.slice(3))}
        </h3>
      );
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      elements.push(
        <div key={i} className="flex gap-1.5 ml-1">
          <span className="text-[#86868B] mt-0.5">•</span>
          <span>{formatInline(line.slice(2))}</span>
        </div>
      );
    } else if (/^\d+\.\s/.test(line)) {
      const match = line.match(/^(\d+)\.\s(.*)/);
      if (match) {
        elements.push(
          <div key={i} className="flex gap-1.5 ml-1">
            <span className="text-[#86868B] font-medium min-w-[1.2em]">
              {match[1]}.
            </span>
            <span>{formatInline(match[2])}</span>
          </div>
        );
      }
    } else if (line.trim() === "") {
      elements.push(<div key={i} className="h-2" />);
    } else {
      elements.push(
        <p key={i} className="leading-relaxed">
          {formatInline(line)}
        </p>
      );
    }
  }

  return <>{elements}</>;
}

function formatInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*)|(`(.+?)`)/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(
        <strong key={key++} className="font-semibold">
          {match[2]}
        </strong>
      );
    } else if (match[4]) {
      parts.push(
        <code
          key={key++}
          className="px-1 py-0.5 rounded bg-black/[0.05] text-[12px] font-mono"
        >
          {match[4]}
        </code>
      );
    }
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}
