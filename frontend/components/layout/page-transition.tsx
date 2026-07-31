"use client";

import { motion } from "framer-motion";

interface PageTransitionProps {
  readonly children: React.ReactNode;
}

export function PageTransition({ children }: PageTransitionProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
      className="h-full"
    >
      {children}
    </motion.div>
  );
}
