import type { Config } from "tailwindcss";
import tailwindAnimate from "tailwindcss-animate";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
        chart: {
          "1": "hsl(var(--chart-1))",
          "2": "hsl(var(--chart-2))",
          "3": "hsl(var(--chart-3))",
          "4": "hsl(var(--chart-4))",
          "5": "hsl(var(--chart-5))",
        },
        apple: {
          blue: "#007AFF",
          green: "#34C759",
          amber: "#FF9F0A",
          red: "#FF3B30",
          purple: "#AF82FF",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        "glass-sm": "var(--shadow-sm)",
        "glass-md": "var(--shadow-md)",
        "glass-lg": "var(--shadow-lg)",
        "blue-glow": "var(--shadow-blue)",
        "green-glow": "var(--shadow-glow)",
      },
      animation: {
        "float": "float 25s ease-in-out infinite",
        "float-slow": "float 30s ease-in-out infinite",
        "float-fast": "float 20s ease-in-out infinite",
        "shimmer": "shimmer 3s ease-in-out infinite",
        "border-flow": "border-flow 8s ease infinite",
        "pulse-dot": "pulse-dot 2s ease-in-out infinite",
        "scan": "scan 3s ease-in-out infinite",
        "bob": "bob 3s ease-in-out infinite",
        "draw-line": "draw-line 2s ease forwards",
        "fill-bar": "fill-bar 1.5s ease forwards",
        "slide-up": "slide-up 0.6s ease both",
        "pop": "pop 0.3s ease both",
        "count-up": "count-up 1s ease-out both",
        "fade-in": "fade-in 0.3s ease both",
      },
    },
  },
  plugins: [tailwindAnimate],
};
export default config;
