/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f3ecdf",
        ink: "#1c1712",
        rule: "#d9d0c1",
        fail: "#8f1d2c",
        warn: "#8a5a00",
        pass: "#1f6b3a",
        rust: "#b85c38",
      },
      fontFamily: {
        serif: ["Newsreader", "Palatino", "Georgia", "serif"],
        sans: ["Public Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
