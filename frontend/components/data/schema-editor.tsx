"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Sparkline } from "@/components/charts/sparkline";
import type { DataProfile } from "@/lib/types";

interface SchemaEditorProps {
  schema: Record<string, string>;
  onChange: (schema: Record<string, string>) => void;
  profile?: DataProfile;
}

const TYPE_OPTIONS = ["String", "Int64", "Float64", "Date"] as const;

const TYPE_COLORS: Record<string, string> = {
  String: "bg-indigo-500/15 text-indigo-400 border-indigo-500/20",
  Int64: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20",
  Float64: "bg-rose-500/15 text-rose-400 border-rose-500/20",
  Date: "bg-amber-500/15 text-amber-400 border-amber-500/20",
};

function getColumnStats(profile: DataProfile | undefined, colName: string) {
  if (!profile) return null;
  return profile.column_stats.find((c) => c.name === colName) ?? null;
}

function getCorrelation(profile: DataProfile | undefined, colName: string) {
  if (!profile) return null;
  return profile.correlations.find(
    (c) => c.col_a === colName || c.col_b === colName
  ) ?? null;
}

function distributionSparklineValues(
  stats: ReturnType<typeof getColumnStats>
): number[] {
  if (!stats || !stats.percentiles) return [];
  const pcts = stats.percentiles;
  const keys = Object.keys(pcts).sort(
    (a, b) => parseFloat(a) - parseFloat(b)
  );
  if (keys.length === 0) return [];
  const step = Math.max(1, Math.floor(keys.length / 6));
  const values: number[] = [];
  for (let i = 0; i < keys.length && values.length < 6; i += step) {
    values.push(pcts[keys[i]]);
  }
  const max = Math.max(...values, 0.01);
  const min = Math.min(...values, 0);
  return values.map((v) => (max === min ? 0.5 : (v - min) / (max - min)));
}

export function SchemaEditor({ schema, onChange, profile }: SchemaEditorProps) {
  const columns = Object.entries(schema);

  function handleTypeChange(column: string, newType: string) {
    onChange({ ...schema, [column]: newType });
  }

  return (
    <div className="rounded-xl border border-border/50 bg-card/50 overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="border-border/30 hover:bg-transparent">
            <TableHead className="text-muted-foreground text-xs uppercase tracking-wider font-medium">
              Column
            </TableHead>
            <TableHead className="text-muted-foreground text-xs uppercase tracking-wider font-medium">
              Type
            </TableHead>
            {profile && (
              <>
                <TableHead className="text-muted-foreground text-xs uppercase tracking-wider font-medium">
                  Distribution
                </TableHead>
                <TableHead className="text-muted-foreground text-xs uppercase tracking-wider font-medium">
                  Correlation
                </TableHead>
              </>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {columns.map(([colName, colType]) => {
            const stats = getColumnStats(profile, colName);
            const corr = getCorrelation(profile, colName);
            const sparkValues = distributionSparklineValues(stats);

            return (
              <TableRow
                key={colName}
                className="border-border/20 hover:bg-muted/30"
              >
                <TableCell className="font-mono text-sm text-foreground">
                  {colName}
                </TableCell>
                <TableCell>
                  <select
                    value={colType}
                    onChange={(e) => handleTypeChange(colName, e.target.value)}
                    className="h-7 rounded-md border border-input bg-background px-2 text-xs text-foreground outline-none focus:border-ring cursor-pointer appearance-none"
                  >
                    {TYPE_OPTIONS.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-[10px] px-1.5 py-0 h-4 font-mono ml-2",
                      TYPE_COLORS[colType] ?? TYPE_COLORS.String
                    )}
                  >
                    {colType}
                  </Badge>
                </TableCell>
                {profile && (
                  <>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {sparkValues.length > 0 && (
                          <Sparkline values={sparkValues} />
                        )}
                        {stats?.distribution_type && (
                          <span className="text-xs text-muted-foreground">
                            {stats.distribution_type}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {corr && (
                        <Badge
                          variant="outline"
                          className={cn(
                            "text-[10px] font-mono",
                            corr.significant
                              ? "border-cyan-500/30 text-cyan-400"
                              : "border-border text-muted-foreground"
                          )}
                        >
                          {corr.value.toFixed(2)} (
                          {corr.col_a === colName ? corr.col_b : corr.col_a})
                        </Badge>
                      )}
                    </TableCell>
                  </>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
