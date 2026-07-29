"use client";

import { memo } from "react";
import {
  BaseEdge,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";

export interface FKEdgeData {
  parentCol: string;
  childCol: string;
  complete?: boolean;
  generating?: boolean;
}

function FKEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const d = data as unknown as FKEdgeData | undefined;
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const strokeColor = d?.complete ? "#34C759" : d?.generating ? "#007AFF" : "#86868B";

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: strokeColor,
          strokeWidth: 2,
          strokeDasharray: d?.generating ? "6 4" : "none",
          transition: "stroke 0.3s ease",
        }}
      />

      {/* Animated particle along the edge */}
      {d?.generating && (
        <>
          <circle r="3" fill="#007AFF">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
          </circle>
          <circle r="3" fill="#007AFF" opacity="0.5">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} begin="1s" />
          </circle>
        </>
      )}

      {d?.complete && (
        <circle r="3" fill="#34C759">
          <animateMotion dur="3s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}

      {/* Label at midpoint */}
      {d?.parentCol && d?.childCol && (
        <foreignObject
          width={140}
          height={28}
          x={(sourceX + targetX) / 2 - 70}
          y={(sourceY + targetY) / 2 - 14}
        >
          <div className="flex items-center justify-center">
            <span className="text-[9px] bg-white/90 backdrop-blur-sm border border-black/[0.06] rounded-full px-2 py-0.5 text-[#86868B] font-medium whitespace-nowrap shadow-sm">
              {d.parentCol} → {d.childCol}
            </span>
          </div>
        </foreignObject>
      )}
    </>
  );
}

export const FKEdge = memo(FKEdgeComponent);
