import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The build lands in ../chat, which server.py mounts at /chat/ (StaticFiles,
// html=True). `base` must match that mount point or the hashed asset URLs come
// out rooted at / and 404.
export default defineConfig({
  base: "/chat/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    outDir: path.resolve(__dirname, "../chat"),
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` serves the UI on 5173 and forwards the API to the real
    // FastAPI process, so the chat works with hot reload.
    proxy: {
      "/api": {
        target: process.env.HERMES_API ?? "http://127.0.0.1:8800",
        changeOrigin: true,
      },
    },
  },
});
