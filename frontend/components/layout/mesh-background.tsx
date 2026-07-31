"use client";

export function MeshBackground() {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
      {/* Blue orb — top right */}
      <div
        className="absolute animate-float"
        style={{
          width: 600,
          height: 600,
          top: "-10%",
          left: "60%",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(0,122,255,0.25), transparent 70%)",
          filter: "blur(80px)",
          opacity: 0.35,
        }}
      />
      {/* Green orb — bottom left */}
      <div
        className="animate-float-slow"
        style={{
          position: "absolute",
          width: 500,
          height: 500,
          bottom: "-5%",
          left: "-5%",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(52,199,89,0.18), transparent 70%)",
          filter: "blur(80px)",
          opacity: 0.35,
          animationDelay: "-5s",
        }}
      />
      {/* Purple orb — mid right */}
      <div
        className="animate-float-fast"
        style={{
          position: "absolute",
          width: 400,
          height: 400,
          top: "40%",
          right: "-5%",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(175,130,255,0.15), transparent 70%)",
          filter: "blur(80px)",
          opacity: 0.35,
          animationDelay: "-10s",
        }}
      />
    </div>
  );
}
