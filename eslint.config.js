import js from "@eslint/js";
import globals from "globals";

// web/ has no bundler: classic <script> files sharing globals, plus a Worker.
export default [
  // web/app is TypeScript (checked by tsc in its own package); dist/ is build output.
  { ignores: ["node_modules/**", "web/app/**"] },
  js.configs.recommended,
  {
    files: ["web/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: { ...globals.browser, THREE: "readonly", AeroDebug: "readonly" },
    },
  },
  {
    files: ["web/depth-sort_worker.js"],
    languageOptions: { globals: { ...globals.worker } },
  },
  { files: ["eslint.config.js"], languageOptions: { sourceType: "module" } },
];
