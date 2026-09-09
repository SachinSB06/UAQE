/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        luxury: {
          bg: "#FAFAFA",
          surface: "#FFFFFF",
          elevated: "#F8FAFC",
          border: "#E2E8F0",
          borderStrong: "#CBD5E1",
          textPrimary: "#0F172A",
          textSecondary: "#475569",
          textMuted: "#94A3B8",
          accent: "#4F46E5",
          accentHover: "#4338CA",
          accentSubtle: "#EEF2FF",
          success: "#059669",
          successSubtle: "#ECFDF5",
          warning: "#D97706",
          warningSubtle: "#FFFBEB",
          critical: "#E11D48",
          criticalSubtle: "#FFF1F2",
        }
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'sans-serif'
        ],
        mono: [
          '"JetBrains Mono"',
          '"Fira Code"',
          'Consolas',
          'monospace'
        ]
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02)',
        'luxury': '0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03)',
        'elevated': '0 10px 30px -4px rgba(15, 23, 42, 0.08), 0 4px 12px -2px rgba(15, 23, 42, 0.04)',
      }
    },
  },
  plugins: [],
}
