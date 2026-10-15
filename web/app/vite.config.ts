import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Served by FastAPI under /app/ (aerosplat.server.main); `npm run dev` proxies
// the API and the classic viewer to a running `aerosplat up` on :8000.
const backend = process.env.AEROSPLAT_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: backend, ws: true, changeOrigin: true },
      "/web": { target: backend, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    chunkSizeWarningLimit: 1500, // maplibre-gl alone is ~800 kB
  },
});
