// Browser demo for ahirudb. Zero build step, same as js/ahirudb.js itself —
// this is just a thin UI wrapper over the public AhiruDB API.

import { AhiruDB, AhiruError, detectFormat } from '../js/ahirudb.js';

const WASM_URL = '../target/ahiru-core-full.wasm';

// Bundled under tests/data/, used by the crate's own test suite too.
// Each sample registers one or more tables (`tables`); multi-table samples
// exist to showcase cross-format JOINs. `query` is the SQL dropped into the
// box after registering — defaults to a plain `SELECT * FROM <first table>`
// when omitted.
const SAMPLES = [
  {
    label: 'basic.parquet — flat types (Parquet)',
    tables: [{ table: 'basic', path: '../tests/data/basic.parquet' }],
  },
  {
    label: 'basic.csv — same data (CSV)',
    tables: [{ table: 'basic_csv', path: '../tests/data/basic.csv', format: 'csv' }],
  },
  {
    label: 'basic.jsonl — same data (JSONL)',
    tables: [{ table: 'basic_jsonl', path: '../tests/data/basic.jsonl', format: 'jsonl' }],
  },
  {
    label: 'pivot.parquet — for PIVOT/UNPIVOT',
    tables: [{ table: 'pivot', path: '../tests/data/pivot.parquet' }],
  },
  {
    label: 'list1.parquet — nested LIST column',
    tables: [{ table: 'list1', path: '../tests/data/list1.parquet' }],
  },
  {
    label: 'json_demo.json — JSON type (object + list columns)',
    tables: [{ table: 'json_demo.json', path: '../tests/data/json_demo.json' }],
    query:
      'SELECT id, name, tags, attrs, json_array_length(tags) AS n_tags,\n' +
      "       json_extract_string(attrs, '$.color') AS color\n" +
      'FROM "json_demo.json" ORDER BY id',
  },
  {
    label: 'orders × customers × regions — cross-format JOIN + CTE + aggregates',
    tables: [
      { table: 'customers', path: '../tests/data/customers.parquet' },
      { table: 'orders', path: '../tests/data/orders.csv', format: 'csv' },
      { table: 'regions', path: '../tests/data/regions.jsonl', format: 'jsonl' },
    ],
    // customers is Parquet, orders is CSV, regions is JSONL: three formats
    // in one query. The CTE aggregates orders per customer; the outer query
    // joins across all three tables and COALESCEs customers with zero paid
    // orders (Frank) instead of dropping them.
    query:
      'WITH order_totals AS (\n' +
      '  SELECT customer_id, COUNT(*) AS n_orders, SUM(amount) AS total_amount\n' +
      '  FROM orders\n' +
      "  WHERE status = 'paid'\n" +
      '  GROUP BY customer_id\n' +
      ')\n' +
      'SELECT c.name, c.region, r.manager,\n' +
      '       COALESCE(o.n_orders, 0) AS n_orders,\n' +
      '       COALESCE(o.total_amount, 0) AS total_amount\n' +
      'FROM customers c\n' +
      'JOIN regions r ON r.region = c.region\n' +
      'LEFT JOIN order_totals o ON o.customer_id = c.customer_id\n' +
      'ORDER BY total_amount DESC, c.name',
  },
];

const sqlBox = document.getElementById('sql');
const runBtn = document.getElementById('run');
const resultsEl = document.getElementById('results');
const statusEl = document.getElementById('status');
const tablesEl = document.getElementById('tables');
const sampleSelect = document.getElementById('sample-select');
const loadSampleBtn = document.getElementById('load-sample');
const fileInput = document.getElementById('file-input');
const dbStatusEl = document.getElementById('db-status');

let dbPromise = null;
const registered = new Map(); // lowercase table name -> human label

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

function quoteIfNeeded(name) {
  return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name) ? name : `"${name}"`;
}

function renderTablesList() {
  tablesEl.innerHTML = '';
  for (const [name, label] of registered) {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = name;
    btn.title = label;
    btn.addEventListener('click', () => {
      sqlBox.value = `SELECT * FROM ${quoteIfNeeded(name)} LIMIT 20`;
      sqlBox.focus();
    });
    li.appendChild(btn);
    tablesEl.appendChild(li);
  }
}

function ensureDb() {
  if (!dbPromise) {
    dbPromise = AhiruDB.init({ wasmUrl: WASM_URL, memoryLimit: 256 * 1024 * 1024 })
      .then((db) => {
        dbStatusEl.textContent = 'Engine ready.';
        return db;
      })
      .catch((err) => {
        dbStatusEl.textContent = `Failed to load wasm: ${err.message ?? err}`;
        dbStatusEl.classList.add('error');
        throw err;
      });
  }
  return dbPromise;
}

function formatCell(v) {
  if (v === null || v === undefined) return 'NULL';
  if (typeof v === 'bigint') return v.toString();
  if (v instanceof Uint8Array) return `<${v.length} bytes>`;
  // INTERVAL comes back as { months, days, micros } (unpackInterval in
  // js/ahirudb.js). The default String() would render it as [object Object].
  if (v !== null && typeof v === 'object') return JSON.stringify(v, (_, x) => (typeof x === 'bigint' ? x.toString() : x));
  return String(v);
}

function renderRows(rows) {
  resultsEl.innerHTML = '';
  if (rows.length === 0) {
    resultsEl.textContent = '(0 rows)';
    return;
  }
  const cols = Object.keys(rows[0]);
  const table = document.createElement('table');

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const c of cols) {
    const th = document.createElement('th');
    th.textContent = c;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr');
    for (const c of cols) {
      const td = document.createElement('td');
      const v = row[c];
      td.textContent = formatCell(v);
      if (v === null || v === undefined) td.classList.add('null');
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  resultsEl.appendChild(table);
}

async function runQuery() {
  const sql = sqlBox.value.trim();
  if (!sql) return;
  runBtn.disabled = true;
  setStatus('Running…');
  const t0 = performance.now();
  try {
    const db = await ensureDb();
    const rows = await db.query(sql);
    const ms = (performance.now() - t0).toFixed(1);
    setStatus(`${rows.length} row${rows.length === 1 ? '' : 's'} in ${ms} ms`);
    renderRows(rows);
  } catch (err) {
    const ms = (performance.now() - t0).toFixed(1);
    setStatus(`Error after ${ms} ms: ${err instanceof AhiruError ? err.message : (err.message ?? err)}`, true);
    resultsEl.innerHTML = '';
  } finally {
    runBtn.disabled = false;
  }
}

async function loadSample() {
  const sample = SAMPLES[Number(sampleSelect.value)];
  if (!sample) return;
  loadSampleBtn.disabled = true;
  const paths = sample.tables.map((t) => t.path).join(', ');
  setStatus(`Registering ${paths}…`);
  try {
    const db = await ensureDb();
    for (const t of sample.tables) {
      db.register(t.table, t.path, t.format ? { format: t.format } : undefined);
      registered.set(t.table, sample.label);
    }
    renderTablesList();
    sqlBox.value = sample.query ?? `SELECT * FROM ${quoteIfNeeded(sample.tables[0].table)} LIMIT 20`;
    await runQuery();
  } catch (err) {
    setStatus(`Failed to register ${paths}: ${err.message ?? err}`, true);
  } finally {
    loadSampleBtn.disabled = false;
  }
}

async function loadFile() {
  const file = fileInput.files[0];
  if (!file) return;
  const stripped = file.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_]/g, '_') || 'uploaded';
  // Detect the format from the *original* filename before the extension is
  // stripped for the table name — otherwise register() has nothing left to
  // infer from and every non-Parquet upload gets misread as Parquet.
  const format = detectFormat(file.name);
  // Single-document JSON has no explicit `format` wire value yet
  // (js/ahirudb.js's FORMAT_CODES has no entry for it), so it can only be
  // reached through extension-based auto-detection. Keep the extension on
  // the registered name in that case, same as the bundled json_demo.json
  // sample; reference it quoted (quoteIfNeeded handles that below).
  const name = format === 'json' ? `${stripped}.json` : stripped;
  const options = format !== 'json' && format !== 'parquet' ? { format } : undefined;
  try {
    const db = await ensureDb();
    db.register(name, file, options);
    registered.set(name, file.name);
    renderTablesList();
    sqlBox.value = `SELECT * FROM ${quoteIfNeeded(name)} LIMIT 20`;
    await runQuery();
  } catch (err) {
    setStatus(`Failed to register ${file.name}: ${err.message ?? err}`, true);
  } finally {
    fileInput.value = '';
  }
}

for (const [i, s] of SAMPLES.entries()) {
  const opt = document.createElement('option');
  opt.value = String(i);
  opt.textContent = s.label;
  sampleSelect.appendChild(opt);
}

runBtn.addEventListener('click', runQuery);
sqlBox.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    runQuery();
  }
});
loadSampleBtn.addEventListener('click', loadSample);
fileInput.addEventListener('change', loadFile);

// Warm up the engine immediately and register+run the default sample so the
// page is never just an empty box.
ensureDb().then(() => loadSample());
