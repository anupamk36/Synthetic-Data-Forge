"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Loader2,
  Search,
  Save,
  Trash2,
  Download,
  ArrowRight,
  Tag,
  Columns3,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { FileUpload } from "@/components/data/file-upload";
import { GlassCard } from "@/components/shared/glass-card";

import { useSchemas, useSaveSchema, useDeleteSchema } from "@/hooks/use-schemas";
import { useForgeStore } from "@/lib/store";
import type { Schema, SavedSchema } from "@/lib/types";

export default function SchemasPage() {
  const router = useRouter();
  const store = useForgeStore();

  const [search, setSearch] = useState("");
  const { data: schemas, isLoading } = useSchemas(search);

  const [saveName, setSaveName] = useState("");
  const [saveDescription, setSaveDescription] = useState("");
  const [saveTags, setSaveTags] = useState("");
  const saveSchema = useSaveSchema();

  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const deleteSchema = useDeleteSchema();

  const [importing, setImporting] = useState(false);

  const handleSave = useCallback(async () => {
    if (!store.lastSchema) {
      toast.error("No schema in session to save");
      return;
    }
    if (!saveName.trim()) {
      toast.error("Enter a name for the schema");
      return;
    }
    try {
      await saveSchema.mutateAsync({
        name: saveName.trim(),
        schema: store.lastSchema,
        description: saveDescription.trim(),
        tags: saveTags.trim(),
      });
      toast.success(`Schema "${saveName}" saved`);
      setSaveName("");
      setSaveDescription("");
      setSaveTags("");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Save failed");
    }
  }, [store.lastSchema, saveName, saveDescription, saveTags, saveSchema]);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteSchema.mutateAsync(id);
        toast.success("Schema deleted");
        setConfirmDeleteId(null);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Delete failed");
      }
    },
    [deleteSchema]
  );

  const handleUse = useCallback(
    (schema: SavedSchema) => {
      store.setLastGenerated([], schema.schema);
      toast.success(`Loaded "${schema.name}" into generator`);
      router.push("/generate/single");
    },
    [store, router]
  );

  const handleExport = useCallback((schema: SavedSchema) => {
    const blob = new Blob([JSON.stringify(schema.schema, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${schema.name}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Schema exported");
  }, []);

  const handleImport = useCallback(
    async (file: File) => {
      setImporting(true);
      try {
        const text = await file.text();
        const parsed = JSON.parse(text) as Schema;
        if (typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new TypeError("Invalid schema format");
        }
        const name = file.name.replace(/\.json$/i, "");
        await saveSchema.mutateAsync({
          name,
          schema: parsed,
          description: "Imported from JSON file",
        });
        toast.success(`Schema "${name}" imported`);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Import failed");
      } finally {
        setImporting(false);
      }
    },
    [saveSchema]
  );

  return (
    <div className="flex flex-col h-full">
      <TopBar title="Schema Library" />

      <div className="flex-1 overflow-y-auto relative z-10">
        <div className="px-6 py-6 max-w-5xl mx-auto space-y-6">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-[#86868B]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search schemas..."
              className="w-full rounded-xl border border-black/[0.08] bg-white/80 backdrop-blur-sm pl-10 pr-4 py-2.5 text-[13px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all shadow-sm"
            />
          </div>

          {/* Save current schema */}
          {store.lastSchema && (
            <GlassCard>
              <div className="p-5 space-y-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">
                  Save Current Schema
                </p>
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="Schema name"
                  className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[13px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                />
                <textarea
                  value={saveDescription}
                  onChange={(e) => setSaveDescription(e.target.value)}
                  placeholder="Description (optional)"
                  rows={2}
                  className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[13px] resize-none focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                />
                <input
                  type="text"
                  value={saveTags}
                  onChange={(e) => setSaveTags(e.target.value)}
                  placeholder="Tags (comma-separated)"
                  className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[13px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                />
                <button
                  onClick={handleSave}
                  disabled={saveSchema.isPending}
                  className="w-full glow-button bg-[#007AFF] text-white font-semibold text-[13px] py-2.5 rounded-lg flex items-center justify-center gap-2 btn-shimmer disabled:opacity-40"
                >
                  {saveSchema.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4" />
                  )}
                  Save Schema
                </button>
              </div>
            </GlassCard>
          )}

          {/* Schema grid */}
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
              Saved Schemas
            </p>

            {isLoading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-[#007AFF]" />
              </div>
            )}
            {!isLoading && schemas && schemas.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {schemas.map((schema, i) => (
                  <div
                    key={schema.id}
                    className="rounded-xl border border-black/[0.06] bg-white/60 backdrop-blur-sm p-4 hover-lift animate-slide-up"
                    style={{ animationDelay: `${i * 60}ms` }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="text-[13px] font-semibold text-[#1D1D1F]">{schema.name}</h4>
                        {schema.description && (
                          <p className="text-[11px] text-[#86868B] mt-0.5 line-clamp-2">{schema.description}</p>
                        )}
                      </div>
                      <div className="flex items-center gap-1 text-[10px] text-[#86868B] bg-black/[0.03] px-2 py-0.5 rounded-full">
                        <Columns3 className="h-3 w-3" />
                        {Object.keys(schema.schema).length}
                      </div>
                    </div>

                    {schema.tags && (
                      <div className="flex flex-wrap gap-1.5 mb-3">
                        {schema.tags.split(",").map((tag) => (
                          <span
                            key={tag.trim()}
                            className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-[#007AFF]/[0.06] text-[#007AFF]"
                          >
                            <Tag className="h-2.5 w-2.5" />
                            {tag.trim()}
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="flex items-center justify-between pt-2 border-t border-black/[0.04]">
                      <span className="text-[10px] text-[#86868B]">
                        {new Date(schema.created_at).toLocaleDateString()}
                      </span>

                      <div className="flex items-center gap-3">
                        <button
                          onClick={() => handleUse(schema)}
                          className="text-[11px] font-medium text-[#007AFF] hover:text-[#005EC4] transition-colors flex items-center gap-1"
                        >
                          <ArrowRight className="h-3 w-3" />
                          Use
                        </button>
                        <button
                          onClick={() => handleExport(schema)}
                          className="text-[11px] text-[#86868B] hover:text-[#1D1D1F] transition-colors flex items-center gap-1"
                        >
                          <Download className="h-3 w-3" />
                        </button>
                        {confirmDeleteId === schema.id ? (
                          <div className="flex items-center gap-1.5">
                            <button
                              onClick={() => handleDelete(schema.id)}
                              disabled={deleteSchema.isPending}
                              className="text-[11px] font-medium text-[#FF3B30] hover:text-[#FF3B30]/80 transition-colors"
                            >
                              {deleteSchema.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : "Yes"}
                            </button>
                            <button
                              onClick={() => setConfirmDeleteId(null)}
                              className="text-[11px] text-[#86868B] hover:text-[#1D1D1F] transition-colors"
                            >
                              No
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setConfirmDeleteId(schema.id)}
                            className="text-[11px] text-[#86868B] hover:text-[#FF3B30] transition-colors"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!isLoading && (!schemas || schemas.length === 0) && (
              <div className="text-center py-12 text-[13px] text-[#86868B]">
                No schemas saved yet
              </div>
            )}
          </div>

          {/* Import */}
          <div className="rounded-xl border border-black/[0.06] bg-white/50 p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
              Import Schema
            </p>
            <FileUpload onFileAccepted={handleImport} loading={importing} accept={[".json"]} />
            <p className="text-[11px] text-[#86868B] mt-2 text-center">
              Upload a JSON file containing a column-to-type mapping
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
