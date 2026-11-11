import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the UI runs on :5173 and forwards /api to the backend in WSL.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
