export const API_BASE = window.location.origin;

export async function fetchJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const msg = typeof detail === "string" ? detail : JSON.stringify(detail) || res.statusText;
    throw new Error(msg);
  }
  return data;
}

export function getSystemInfo() {
  return fetchJson("/v1/system/info");
}

export function getCsvRow(path, row_index) {
  const q = new URLSearchParams({ path, row_index: String(row_index) });
  return fetchJson(`/v1/data/csv-row?${q}`);
}

export function getHealth() {
  return fetchJson("/health");
}

export function getArchitecture() {
  return fetchJson("/architecture");
}

export function scoreTransactions(transactions) {
  return fetchJson("/v1/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transactions }),
  });
}

export function scoreFromPath(path, max_rows) {
  return fetchJson("/v1/score/from-path", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, max_rows }),
  });
}

export async function scoreFromUpload(file, max_rows) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE}/v1/score/from-upload?max_rows=${max_rows}`, {
    method: "POST",
    body,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export function openRealtimeSocket(max_batches = 80) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${window.location.host}/v1/realtime/ws`;
  const ws = new WebSocket(url);
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ max_batches }));
  });
  return ws;
}
