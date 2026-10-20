// PRD-262 #7: logs every module Node resolves, so media-render.yml can prove that
// render, check and snapshot never load an npm path the image prunes
// (services/media-render/js/prune.txt).
//
// The CI step mounts services/media-render/ci at /ci (ci/ never ships in the image)
// and starts the service with NODE_OPTIONS=--import=/ci/trace_modules.mjs, so every
// Node process the Hyperframes CLI runs, and every one it spawns, loads this first.
// module.registerHooks (Node 22.15+) hooks import, import() and require() alike.
// Each line of the log is one JSON object:
//   {"pid": 7, "argv": [...]}           once per process: its script and arguments
//   {"pid": 7, "file": "/abs/path.js"}  a module that resolved to a file
//   {"pid": 7, "failed": "zod", "parent": "file:///…"}  a specifier that did not
//                                         resolve (the error still reaches the caller)
// ci/assert_module_trace.py reads the log on the runner.
import { appendFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";

const LOG = "/trace/modules.log";

function record(entry) {
  appendFileSync(LOG, `${JSON.stringify({ pid: process.pid, ...entry })}\n`);
}

record({ argv: process.argv.slice(1) });

registerHooks({
  resolve(specifier, context, nextResolve) {
    let result;
    try {
      result = nextResolve(specifier, context);
    } catch (error) {
      record({ failed: specifier, parent: context.parentURL ?? null });
      throw error;
    }
    if (result.url.startsWith("file:")) {
      record({ file: fileURLToPath(result.url) });
    }
    return result;
  },
});
