"use client";

import { useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  BackgroundVariant,
} from "@xyflow/react";
import dagre from "dagre";
import "@xyflow/react/dist/style.css";

import { TableNode, type TableNodeData } from "./table-node";
import { FKEdge, type FKEdgeData } from "./fk-edge";

interface TableInfo {
  name: string;
  columnCount: number;
  columns: string[];
}

interface Relationship {
  parent_table: string;
  parent_col: string;
  child_table: string;
  child_col: string;
}

interface GraphCanvasProps {
  readonly tables: TableInfo[];
  readonly relationships: Relationship[];
  readonly generating?: boolean;
  readonly progress?: Record<string, number>;
  readonly completedTables?: string[];
  readonly onNodeClick?: (tableName: string) => void;
}

const nodeTypes = { table: TableNode };
const edgeTypes = { fk: FKEdge };
const EMPTY_PROGRESS: Record<string, number> = {};
const EMPTY_COMPLETED: string[] = [];

function layoutGraph(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", ranksep: 100, nodesep: 80 });

  nodes.forEach((node) => {
    g.setNode(node.id, { width: 200, height: 80 });
  });
  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: { x: pos.x - 100, y: pos.y - 40 },
    };
  });
}

export function GraphCanvas({
  tables,
  relationships,
  generating = false,
  progress = EMPTY_PROGRESS,
  completedTables = EMPTY_COMPLETED,
  onNodeClick,
}: GraphCanvasProps) {
  const initialNodes: Node[] = useMemo(
    () =>
      tables.map((t) => ({
        id: t.name,
        type: "table",
        position: { x: 0, y: 0 },
        data: {
          label: t.name,
          columnCount: t.columnCount,
          columns: t.columns,
          expanded: false,
          generating: generating && !completedTables.includes(t.name),
          progress: progress[t.name] ?? 0,
          complete: completedTables.includes(t.name),
        } satisfies TableNodeData,
      })),
    [tables, generating, progress, completedTables]
  );

  const initialEdges: Edge[] = useMemo(
    () =>
      relationships.map((rel, i) => ({
        id: `e-${i}`,
        source: rel.parent_table,
        target: rel.child_table,
        type: "fk",
        animated: generating,
        data: {
          parentCol: rel.parent_col,
          childCol: rel.child_col,
          generating: generating && !completedTables.includes(rel.child_table),
          complete:
            completedTables.includes(rel.parent_table) &&
            completedTables.includes(rel.child_table),
        } satisfies FKEdgeData,
      })),
    [relationships, generating, completedTables]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);

  useEffect(() => {
    if (initialNodes.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }
    const laidOut = layoutGraph(initialNodes, initialEdges);
    setNodes(laidOut);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      onNodeClick?.(node.id);
      setNodes((prev) =>
        prev.map((n) =>
          n.id === node.id
            ? { ...n, data: { ...n.data, expanded: !n.data.expanded } }
            : n
        )
      );
    },
    [onNodeClick, setNodes]
  );

  return (
    <div className="w-full h-[400px] rounded-xl border border-black/[0.06] overflow-hidden bg-white/30 backdrop-blur-sm">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
        className="bg-transparent"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(0,0,0,0.05)" />
        <Controls
          className="!bg-white/80 !backdrop-blur-sm !border-black/[0.06] !rounded-xl !shadow-sm"
          showInteractive={false}
        />
      </ReactFlow>
    </div>
  );
}
