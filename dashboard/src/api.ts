export type Verdict = "PASS" | "WARN" | "FAIL" | string;

export interface Finding {
  severity: string;
  category: string;
  code: string;
  message: string;
  evidence?: string;
  location?: string;
  tool?: string;
}

export interface ScanSummary {
  id: string;
  created_at: string;
  source: string;
  source_kind: string;
  kind: string;
  verdict: Verdict;
  score: number;
  summary: string;
}

export interface HeatCell {
  cwe: string;
  trigger: string;
  rate: number;
  baseline_rate: number;
  significant: boolean;
  min_q: number;
}

export interface DiffItem {
  task_id: string;
  cwe: string;
  trigger_id: string;
  q_value: number;
  baseline: string;
  triggered: string;
}

export interface SandboxEvent {
  label: string;
  kind: string;
  detail: string;
}

export interface TriggerStat {
  trigger_id: string;
  suite: string;
  text: string;
  control?: string | null;
  n: number;
  insecure_rate: number;
  control_rate?: number | null;
  risk_difference: number;
  q_value: number;
  q_control?: number | null;
  significant: boolean;
  refusal_significant?: boolean;
}

export interface ScanDetail extends ScanSummary {
  name?: string;
  family?: string | null;
  findings: Finding[];
  files: { relative: string; sha256: string; size: number }[];
  limitation?: string;
  redteam?: {
    runs: number;
    task_count: number;
    trigger_count: number;
    baseline_insecure_rate: number;
    pooled_fisher_p: number;
    significant_cells: number;
    significant_triggers?: number;
    refusal_cells: number;
    temperature?: number;
    wall_seconds?: number;
    backend?: string;
    notes?: string[];
    triggers?: TriggerStat[];
    threshold_q: number;
    heatmap?: HeatCell[];
    diffs?: DiffItem[];
  } | null;
  sandbox?: {
    samples?: number;
    canary_read?: boolean;
    outbound_attempt?: boolean;
    outbound_hosts?: string[];
    isolation?: string | null;
  } | null;
  sandbox_events?: SandboxEvent[];
  external_scanners?: Record<string, string>;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function fetchScans() {
  return getJson<{ scans: ScanSummary[] }>("/api/scans");
}

export function fetchScan(id: string) {
  return getJson<ScanDetail>(`/api/scans/${id}`);
}

export function fetchHeatmap(id: string) {
  return getJson<{ cells: HeatCell[] }>(`/api/scans/${id}/heatmap`);
}

export function fetchDiffs(id: string) {
  return getJson<{ diffs: DiffItem[] }>(`/api/scans/${id}/diffs`);
}

export function fetchSandbox(id: string) {
  return getJson<{ summary: ScanDetail["sandbox"]; events: SandboxEvent[] }>(`/api/scans/${id}/sandbox`);
}

export type GuardDecision = "pass" | "annotate" | "warn" | "block" | string;

export interface GuardFinding {
  rule: string;
  severity: string;
  message: string;
  block: number;
  language: string;
  line: number;
  evidence?: string;
  tool?: string;
  action: string;
}

export interface GuardEvent {
  id: number;
  created_at: string;
  upstream: string;
  model: string;
  decision: GuardDecision;
  findings: GuardFinding[];
  rules: string[];
  request_preview: string;
  response_preview: string;
  latency_ms: number;
}

export interface GuardSummary {
  total: number;
  decisions: Record<string, number>;
  models: { model: string; events: number }[];
  top_rules: { rule: string; events: number }[];
  mean_latency_ms: number | null;
  last_event_at: string | null;
}

export function fetchGuardEvents(limit = 100) {
  return getJson<{ events: GuardEvent[] }>(`/api/guard/events?limit=${limit}`);
}

export function fetchGuardSummary() {
  return getJson<GuardSummary>("/api/guard/summary");
}
