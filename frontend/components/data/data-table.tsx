"use client";

import { useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface DataTableProps {
  data: Record<string, unknown>[];
  maxRows?: number;
}

export function DataTable({ data, maxRows = 20 }: DataTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const slicedData = useMemo(
    () => data.slice(0, maxRows),
    [data, maxRows]
  );

  const columns = useMemo(() => {
    if (data.length === 0) return [];
    const keys = Object.keys(data[0]);
    const helper = createColumnHelper<Record<string, unknown>>();

    return keys.map((key) =>
      helper.accessor((row) => row[key], {
        id: key,
        header: key,
        cell: (info) => {
          const val = info.getValue();
          if (val === null || val === undefined) {
            return <span className="text-muted-foreground/40 italic">null</span>;
          }
          return String(val);
        },
      })
    );
  }, [data]);

  const table = useReactTable({
    data: slicedData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-border/50 bg-card/50 p-8 text-muted-foreground text-sm">
        No data to display
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <Badge variant="secondary" className="text-xs font-mono">
          {data.length.toLocaleString()} rows
        </Badge>
      </div>

      <div className="rounded-xl border border-border/50 bg-card/50 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="border-b border-border/30">
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      className={cn(
                        "h-9 px-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground",
                        header.column.getCanSort() && "cursor-pointer select-none hover:text-foreground"
                      )}
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <div className="flex items-center gap-1.5">
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {header.column.getCanSort() && (
                          <>
                            {header.column.getIsSorted() === "asc" ? (
                              <ArrowUp className="size-3" />
                            ) : header.column.getIsSorted() === "desc" ? (
                              <ArrowDown className="size-3" />
                            ) : (
                              <ArrowUpDown className="size-3 opacity-40" />
                            )}
                          </>
                        )}
                      </div>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row, rowIdx) => (
                <tr
                  key={row.id}
                  className={cn(
                    "border-b border-border/10 transition-colors hover:bg-muted/30",
                    rowIdx % 2 === 1 && "bg-muted/10"
                  )}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="px-3 py-2 font-mono text-xs whitespace-nowrap"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {data.length > maxRows && (
        <p className="text-xs text-muted-foreground text-center">
          Showing {maxRows.toLocaleString()} of {data.length.toLocaleString()} rows
        </p>
      )}
    </div>
  );
}
