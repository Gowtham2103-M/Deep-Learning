import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development every request to /api is forwarded to the FastAPI backend on port 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
