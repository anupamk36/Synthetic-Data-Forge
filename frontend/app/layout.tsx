import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";
import { Sidebar } from "@/components/layout/sidebar";
import { MeshBackground } from "@/components/layout/mesh-background";
import { CommandPalette } from "@/components/layout/command-palette";
import { ChatPanel } from "@/components/layout/chat-panel";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Clinical Data Forge",
  description: "Compliant synthetic clinical & life sciences data platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={cn("font-sans", inter.variable)}>
      <body className={cn("antialiased", inter.className)}>
        <Providers>
          <MeshBackground />
          <CommandPalette />
          <div className="relative z-10 flex h-screen">
            <Sidebar />
            <main className="flex-1 overflow-hidden">
              {children}
            </main>
          </div>
          <ChatPanel />
        </Providers>
      </body>
    </html>
  );
}
