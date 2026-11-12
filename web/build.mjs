import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(root, "dist");
await mkdir(output, { recursive: true });
await cp(path.join(root, "index.html"), path.join(output, "index.html"));
await cp(path.join(root, "assets"), path.join(output, "assets"), { recursive: true });
const configuredApiBase = (process.env.VITE_API_BASE_URL || "").trim();
let apiBase = "";
if (configuredApiBase) {
  let parsed;
  try {
    parsed = new URL(configuredApiBase);
  } catch {
    throw new Error("VITE_API_BASE_URL must be an absolute HTTP(S) origin.");
  }
  const localHttp = parsed.protocol === "http:" && ["localhost", "127.0.0.1"].includes(parsed.hostname);
  if ((parsed.protocol !== "https:" && !localHttp) || parsed.username || parsed.password ||
      parsed.pathname !== "/" || parsed.search || parsed.hash) {
    throw new Error("VITE_API_BASE_URL must be an HTTPS origin (HTTP is allowed only for localhost).");
  }
  apiBase = parsed.origin;
}
await writeFile(
  path.join(output, "config.js"),
  `window.RAG_API_BASE = ${JSON.stringify(apiBase)};\n`,
  "utf8",
);
const htmlPath = path.join(output, "index.html");
const html = await readFile(htmlPath, "utf8");
await writeFile(htmlPath, html, "utf8");
