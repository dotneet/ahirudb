#!/usr/bin/env node
// Zero-dependency static file server for the browser demo (demo/).
//
// Serves the repo root so the demo page can reach js/ahirudb.js,
// target/ahiru-core-full.wasm, and tests/data/*.{parquet,csv,jsonl} with
// plain relative paths. HTTP Range requests are honoured so the demo also
// exercises ahirudb's lazy partial-fetch I/O path against a real server,
// not just fully-buffered local files.

import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const PORT = Number(process.env.PORT ?? 8787);
const DEFAULT_PAGE = '/demo/';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.parquet': 'application/octet-stream',
  '.csv': 'text/csv; charset=utf-8',
  '.tsv': 'text/tab-separated-values; charset=utf-8',
  '.jsonl': 'application/x-ndjson; charset=utf-8',
  '.ndjson': 'application/x-ndjson; charset=utf-8',
};

function contentType(path) {
  return MIME[extname(path).toLowerCase()] ?? 'application/octet-stream';
}

/** Resolves a request path against ROOT, rejecting any escape via `..`. */
function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const target = normalize(join(ROOT, decoded));
  if (target !== ROOT && !target.startsWith(ROOT + sep)) return null;
  return target;
}

function parseRange(header, size) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header ?? '');
  if (!m || (m[1] === '' && m[2] === '')) return null;
  let start = m[1] === '' ? undefined : Number(m[1]);
  let end = m[2] === '' ? undefined : Number(m[2]);
  if (start === undefined) {
    // Suffix form: `bytes=-N` means "the last N bytes".
    start = Math.max(0, size - end);
    end = size - 1;
  } else if (end === undefined || end >= size) {
    end = size - 1;
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) return null;
  return { start, end };
}

const server = createServer(async (req, res) => {
  try {
    const urlPath = req.url === '/' ? DEFAULT_PAGE : req.url;
    let filePath = safePath(urlPath);
    if (filePath === null) {
      res.writeHead(403).end('Forbidden');
      return;
    }

    let st;
    try {
      st = await stat(filePath);
      if (st.isDirectory()) {
        filePath = join(filePath, 'index.html');
        st = await stat(filePath);
      }
    } catch {
      res.writeHead(404).end('Not found');
      return;
    }

    const type = contentType(filePath);
    const range = req.headers.range ? parseRange(req.headers.range, st.size) : null;

    if (req.headers.range && !range) {
      res.writeHead(416, { 'Content-Range': `bytes */${st.size}` }).end();
      return;
    }

    if (range) {
      res.writeHead(206, {
        'Content-Type': type,
        'Content-Length': range.end - range.start + 1,
        'Content-Range': `bytes ${range.start}-${range.end}/${st.size}`,
        'Accept-Ranges': 'bytes',
      });
      if (req.method === 'HEAD') {
        res.end();
        return;
      }
      createReadStream(filePath, { start: range.start, end: range.end }).pipe(res);
      return;
    }

    res.writeHead(200, { 'Content-Type': type, 'Content-Length': st.size, 'Accept-Ranges': 'bytes' });
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    createReadStream(filePath).pipe(res);
  } catch (err) {
    res.writeHead(500).end(String(err));
  }
});

server.listen(PORT, () => {
  console.log(`ahirudb demo server: http://localhost:${PORT}${DEFAULT_PAGE}`);
  console.log('Press Ctrl+C to stop.');
});
