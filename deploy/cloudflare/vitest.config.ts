import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // Tests run in Node, which has no workerd built-ins; stand in for the Durable Object base.
    alias: {
      "cloudflare:workers": fileURLToPath(new URL("./test/cloudflare-workers.ts", import.meta.url)),
    },
  },
});
