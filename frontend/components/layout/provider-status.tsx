"use client";

import { useEffect, useMemo, useState } from "react";
import { useForgeStore } from "@/lib/store";
import { getHealth, getProviders } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProviderInfo } from "@/lib/types";

const PROVIDER_NAMES = ["alchemy", "ollama", "claude", "openai", "gemini"] as const;

const selectClass =
  "w-full h-7 rounded-md border border-black/[0.06] bg-white/60 px-2 text-xs text-[#1D1D1F] outline-none focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/30 appearance-none cursor-pointer";

export function ProviderStatus() {
  const {
    provider,
    model,
    apiKey,
    providerConnected,
    sessionCostUsd,
    setProvider,
    setModel,
    setApiKey,
    setProviderConnected,
  } = useForgeStore();

  const [providerList, setProviderList] = useState<ProviderInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoadingModels(true);

    async function fetchProviders() {
      try {
        const providers = await getProviders();
        if (!cancelled) {
          setProviderList(providers);
          const current = providers.find((p) => p.name === provider);
          if (current) {
            if (provider === "ollama" || provider === "alchemy") {
              setProviderConnected(current.available === true);
            } else if (apiKey) {
              setProviderConnected(true);
            } else {
              setProviderConnected(false);
            }
          }
        }
      } catch {
        if (!cancelled) {
          setProviderConnected(false);
          if (provider === "ollama") {
            try {
              const health = await getHealth();
              setProviderConnected(health.ollama_available);
            } catch {
              setProviderConnected(false);
            }
          }
        }
      } finally {
        if (!cancelled) setLoadingModels(false);
      }
    }

    fetchProviders();
    return () => { cancelled = true; };
  }, [provider, apiKey, setProviderConnected]);

  const currentProvider = providerList.find((p) => p.name === provider);
  const models = useMemo(() => currentProvider?.models ?? [], [currentProvider]);

  useEffect(() => {
    if (models.length > 0 && (!model || !models.includes(model))) {
      setModel(models[0]);
    }
  }, [models, model, setModel]);

  const handleRefresh = async () => {
    setLoadingModels(true);
    try {
      const providers = await getProviders();
      setProviderList(providers);
    } catch { /* ignore */ }
    finally { setLoadingModels(false); }
  };

  return (
    <div className="rounded-[11px] border border-black/[0.05] bg-white/60 p-3 space-y-2.5 shadow-glass-sm">
      {/* Header */}
      <div className="text-[10px] font-semibold text-[#86868B] uppercase tracking-[0.5px]">
        LLM Provider
      </div>

      {/* Provider select */}
      <select
        value={provider}
        onChange={(e) => setProvider(e.target.value)}
        className={selectClass}
      >
        {PROVIDER_NAMES.map((p) => (
          <option key={p} value={p}>
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </option>
        ))}
      </select>

      {/* API key (not for ollama or alchemy — those use server-side keys) */}
      {provider !== "ollama" && provider !== "alchemy" && (
        <Input
          type="password"
          placeholder={`${provider.charAt(0).toUpperCase() + provider.slice(1)} API key`}
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          className="h-7 text-xs border-black/[0.06] bg-white/60 focus:border-[#007AFF] focus:ring-[#007AFF]/30"
        />
      )}

      {/* Model select */}
      <div className="flex gap-1">
        <select
          value={model ?? ""}
          onChange={(e) => setModel(e.target.value || null)}
          className={cn(selectClass, "flex-1")}
          disabled={models.length === 0}
        >
          {models.length === 0 ? (
            <option value="">{loadingModels ? "Loading..." : "No models"}</option>
          ) : (
            models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))
          )}
        </select>
        <button
          onClick={handleRefresh}
          disabled={loadingModels}
          className="h-7 w-7 flex items-center justify-center rounded-md border border-black/[0.06] text-[#86868B] hover:text-[#1D1D1F] transition-colors"
          title="Refresh models"
        >
          <RefreshCw className={cn("size-3", loadingModels && "animate-spin")} />
        </button>
      </div>

      {/* Status row */}
      <div className="flex items-center justify-between pt-0.5">
        <div className="flex items-center gap-[7px]">
          <span
            className={cn(
              "size-2 rounded-full",
              providerConnected
                ? "bg-[#34C759] animate-pulse-dot"
                : "bg-[#FF3B30]"
            )}
          />
          <span className="text-[12px] text-[#1D1D1F] font-medium">
            {model ?? provider}
          </span>
        </div>

        <div className="text-[10px] text-[#86868B]">
          ${sessionCostUsd.toFixed(2)}
        </div>
      </div>
    </div>
  );
}
