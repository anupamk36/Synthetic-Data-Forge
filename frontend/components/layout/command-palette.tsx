"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Table2, Link2, Shield, BarChart3, Library, History,
  HeartPulse, FlaskConical, ScanLine, Brain, FileText,
  Search, type LucideIcon,
} from "lucide-react";

interface CommandItem {
  readonly label: string;
  readonly href: string;
  readonly icon: LucideIcon;
  readonly section: string;
}

const COMMANDS: CommandItem[] = [
  { label: "Test Intelligence", href: "/test-intelligence", icon: Brain, section: "AI Testing" },
  { label: "FHIR Generator", href: "/medical/fhir", icon: HeartPulse, section: "Clinical Data" },
  { label: "Clinical Trials", href: "/medical/trials", icon: FlaskConical, section: "Clinical Data" },
  { label: "Imaging Data", href: "/medical/imaging", icon: ScanLine, section: "Clinical Data" },
  { label: "Clinical Notes", href: "/medical/narratives", icon: FileText, section: "Clinical Data" },
  { label: "Single Table", href: "/generate/single", icon: Table2, section: "Data Generation" },
  { label: "Multi-Table", href: "/generate/relational", icon: Link2, section: "Data Generation" },
  { label: "Privacy Audit", href: "/analyze/privacy", icon: Shield, section: "Analyze" },
  { label: "Data Quality", href: "/analyze/quality", icon: BarChart3, section: "Analyze" },
  { label: "Schema Library", href: "/manage/schemas", icon: Library, section: "Manage" },
  { label: "History", href: "/manage/history", icon: History, section: "Manage" },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const router = useRouter();

  const filtered = COMMANDS.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(query.toLowerCase()) ||
      cmd.section.toLowerCase().includes(query.toLowerCase())
  );

  const navigate = useCallback((href: string) => {
    setOpen(false);
    setQuery("");
    router.push(href);
  }, [router]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((prev) => !prev);
        setQuery("");
        setSelected(0);
      }
      if (!open) return;

      if (e.key === "Escape") {
        setOpen(false);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelected((prev) => Math.min(prev + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelected((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter" && filtered[selected]) {
        e.preventDefault();
        navigate(filtered[selected].href);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, filtered, selected, navigate]);

  useEffect(() => {
    setSelected(0);
  }, [query]);

  if (!open) return null;

  const sections = Array.from(new Set(filtered.map((c) => c.section)));

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/20 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />

      {/* Palette */}
      <div className="relative w-[520px] rounded-2xl glass-card overflow-hidden shadow-[0_8px_40px_rgba(0,0,0,0.12)] animate-slide-up">
        {/* Search input */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-black/[0.06]">
          <Search className="size-[18px] text-[#86868B] shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search pages..."
            className="flex-1 bg-transparent text-[15px] text-[#1D1D1F] placeholder:text-[#86868B] outline-none"
            autoFocus
          />
          <kbd className="text-[10px] font-semibold px-[6px] py-[2px] bg-black/[0.06] rounded text-[#86868B]">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-[300px] overflow-y-auto py-2">
          {sections.map((section) => (
            <div key={section}>
              <div className="px-5 pt-3 pb-1 text-[10px] font-semibold text-[#86868B] uppercase tracking-[0.8px]">
                {section}
              </div>
              {filtered
                .filter((c) => c.section === section)
                .map((cmd) => {
                  const idx = filtered.indexOf(cmd);
                  const Icon = cmd.icon;
                  return (
                    <button
                      key={cmd.href}
                      onClick={() => navigate(cmd.href)}
                      onMouseEnter={() => setSelected(idx)}
                      className={`flex w-full items-center gap-3 px-5 py-2.5 text-[13px] transition-colors ${
                        idx === selected
                          ? "bg-[#007AFF] text-white"
                          : "text-[#1D1D1F] hover:bg-black/[0.03]"
                      }`}
                    >
                      <Icon className="size-4 shrink-0" />
                      <span className="font-medium">{cmd.label}</span>
                    </button>
                  );
                })}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="px-5 py-8 text-center text-[13px] text-[#86868B]">
              No results found
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
