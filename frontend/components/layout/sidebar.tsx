"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Table2,
  Link2,
  Shield,
  BarChart3,
  Library,
  History,
  HeartPulse,
  FlaskConical,
  ScanLine,
  Brain,
  FileText,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ProviderStatus } from "@/components/layout/provider-status";

interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    title: "AI Testing",
    items: [
      { label: "Test Intelligence", href: "/test-intelligence", icon: Brain },
    ],
  },
  {
    title: "Clinical Data",
    items: [
      { label: "FHIR Generator", href: "/medical/fhir", icon: HeartPulse },
      { label: "Clinical Trials", href: "/medical/trials", icon: FlaskConical },
      { label: "Imaging Data", href: "/medical/imaging", icon: ScanLine },
      { label: "Clinical Notes", href: "/medical/narratives", icon: FileText },
    ],
  },
  {
    title: "Data Generation",
    items: [
      { label: "Single Table", href: "/generate/single", icon: Table2 },
      { label: "Multi-Table", href: "/generate/relational", icon: Link2 },
    ],
  },
  {
    title: "Analyze",
    items: [
      { label: "Privacy Audit", href: "/analyze/privacy", icon: Shield },
      { label: "Data Quality", href: "/analyze/quality", icon: BarChart3 },
    ],
  },
  {
    title: "Manage",
    items: [
      { label: "Schema Library", href: "/manage/schemas", icon: Library },
      { label: "History", href: "/manage/history", icon: History },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-[250px] flex-col h-full glass border-r border-black/[0.06]">
      {/* Logo */}
      <div className="flex items-center gap-[10px] px-5 py-5">
        <div className="relative flex size-[34px] shrink-0 items-center justify-center rounded-[9px] overflow-hidden"
          style={{
            background: "linear-gradient(135deg, #4CB4FF, #007AFF)",
            boxShadow: "0 2px 10px rgba(0,122,255,0.25)",
          }}
        >
          <span className="text-white font-bold text-[15px] relative z-10">C</span>
          <div
            className="absolute inset-0 animate-shimmer"
            style={{
              background: "linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.3) 50%, transparent 60%)",
            }}
          />
        </div>
        <div>
          <div className="text-[14px] font-semibold text-[#1D1D1F]">Clinical Data Forge</div>
          <div className="text-[10px] text-[#86868B]">v3.0</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 space-y-5 pb-4">
        {NAV_GROUPS.map((group) => (
          <div key={group.title}>
            <p className="mb-1 px-2 text-[10px] font-semibold uppercase tracking-[0.8px] text-[#86868B]">
              {group.title}
            </p>
            <div className="space-y-[2px]">
              {group.items.map((item) => {
                const isActive = pathname === item.href;
                const Icon = item.icon;

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "group relative flex items-center gap-[9px] rounded-[9px] px-[10px] py-2 text-[13px] transition-all duration-200",
                      isActive
                        ? "bg-[rgba(0,122,255,0.08)] text-[#007AFF] font-medium"
                        : "text-[#3A3A3C] hover:bg-black/[0.03] hover:translate-x-[2px]"
                    )}
                  >
                    {isActive && (
                      <div className="absolute -left-[2px] top-1/2 -translate-y-1/2 w-[3px] h-[18px] bg-[#007AFF] rounded-[2px]" />
                    )}
                    <Icon
                      className={cn(
                        "size-[15px] shrink-0 transition-colors",
                        isActive ? "text-[#007AFF]" : "text-[#86868B] group-hover:text-[#3A3A3C]"
                      )}
                    />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Provider Status */}
      <div className="px-3 pb-4">
        <ProviderStatus />
      </div>
    </aside>
  );
}
