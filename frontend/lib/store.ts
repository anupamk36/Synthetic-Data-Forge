import { create } from "zustand";

interface ForgeStore {
  provider: string;
  model: string | null;
  apiKey: string;
  providerConnected: boolean;
  sessionCostUsd: number;
  lastGeneratedData: Record<string, unknown>[] | null;
  lastSchema: Record<string, string> | null;

  chatGeneratedSchema: Record<string, string> | null;
  chatFieldDescriptions: Record<string, string> | null;

  setProvider: (provider: string) => void;
  setModel: (model: string | null) => void;
  setApiKey: (key: string) => void;
  setProviderConnected: (connected: boolean) => void;
  addCost: (amount: number) => void;
  setLastGenerated: (data: Record<string, unknown>[], schema: Record<string, string>) => void;
  clearLastGenerated: () => void;
  setChatSchema: (schema: Record<string, string>, descriptions?: Record<string, string>) => void;
  clearChatSchema: () => void;
}

export const useForgeStore = create<ForgeStore>((set) => ({
  provider: "alchemy",
  model: null,
  apiKey: "",
  providerConnected: false,
  sessionCostUsd: 0,
  lastGeneratedData: null,
  lastSchema: null,
  chatGeneratedSchema: null,
  chatFieldDescriptions: null,

  setProvider: (provider) => set({ provider, model: null, apiKey: "", providerConnected: false }),
  setModel: (model) => set({ model }),
  setApiKey: (apiKey) => set({ apiKey }),
  setProviderConnected: (providerConnected) => set({ providerConnected }),
  addCost: (amount) => set((s) => ({ sessionCostUsd: s.sessionCostUsd + amount })),
  setLastGenerated: (data, schema) => set({ lastGeneratedData: data, lastSchema: schema }),
  clearLastGenerated: () => set({ lastGeneratedData: null, lastSchema: null }),
  setChatSchema: (schema, descriptions) =>
    set({ chatGeneratedSchema: schema, chatFieldDescriptions: descriptions || null }),
  clearChatSchema: () => set({ chatGeneratedSchema: null, chatFieldDescriptions: null }),
}));
