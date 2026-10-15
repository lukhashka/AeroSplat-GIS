import js from "@eslint/js";
import globals from "globals";

// web/ has no bundler: classic <script> files sharing globals, plus a Worker.
export default [
  { ignores: ["node_modules/**"] },
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
