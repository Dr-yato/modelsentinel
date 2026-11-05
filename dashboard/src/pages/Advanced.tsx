import type { ScanDetail } from "@/api";
import { Verdict } from "@/components/Verdict";

// Payloads of the advanced analyses. They are optional on any scan record and
// only the tabs whose payload is present are shown.

export interface AssessmentComponent {
  kind: string;
  title: string;
  scan_id?: string;
  verdict: string;
  explanation: string;
  finding_count: number;
}

export interface SearchResult {
  phrase: string;
  stage: string;
  insecure_rate: number;
  baseline_rate: number;
  lift: number;
  p_value: number;
  q_value?: number;
  significant?: boolean;
}

export interface RefArm {
  arm: string;
  target_rate: number;
  reference_rate: number;
  gap: number;
  q_value: number;
  significant: boolean;
}

export interface DidRow {
  trigger_id: string;
  target_shift: number;
  reference_shift: number;
  difference: number;
  q_value: number;
  significant: boolean;
}

export interface ProbeRow {
  layer: number;
  trigger_id: string;
  n: number;
  auroc: number;
  mean_gap: number;
  cohens_d: number;
  p_value: number;
  flagged: boolean;
}

export interface WeightTensor {
  name: string;
  relative_diff?: number;
  rel_diff?: number;
  effective_rank?: number;
  top_energy?: number;
  flags?: string[];
  spectrum?: number[];
  [key: string]: unknown;
}

export interface AdvancedScan extends ScanDetail {
  assessment?: { components: AssessmentComponent[]; analyses_run: string[] };
  triggersearch?: {
    budget: number;
    fresh_generations: number;
    budget_exhausted: boolean;
    confirmed: SearchResult[];
    minimized_all?: { phrase: string; from: string; lift: number }[];
    top_screen: SearchResult[];
    task_pools: { screen: string[]; evolve: string[]; confirm: string[] };
  };
  refdiff?: {
    target: string;
    reference: string;
    arms: RefArm[];
    difference_in_differences: DidRow[];
    task_count: number;
    runs: number;
  };
  probe?: {
    layers: number[];
    train_items: number;
    eval_pairs: number;
    thresholds: { auroc: number; cohens_d: number; p: number };
    results: ProbeRow[];
  };
  weightdiff?: { [key: string]: unknown };
}

export type AdvancedTab = "assessment" | "search" | "reference" | "probe" | "weights";

export function advancedTabs(scan: AdvancedScan): { id: AdvancedTab; label: string }[] {
  const tabs: { id: AdvancedTab; label: string }[] = [];
  if (scan.assessment) tabs.push({ id: "assessment", label: "Assessment" });
  if (scan.triggersearch) tabs.push({ id: "search", label: "Trigger search" });
  if (scan.refdiff) tabs.push({ id: "reference", label: "Vs reference" });
  if (scan.probe) tabs.push({ id: "probe", label: "Probe" });
  if (scan.weightdiff) tabs.push({ id: "weights", label: "Weight diff" });
  return tabs;
}

const th = "py-2 pr-3";
const headRow = "border-b border-ink text-left font-mono text-[10px] uppercase tracking-[0.14em]";

function pct(value: number | null | undefined): string {
  return value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function num(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "n/a";
  if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(1);
  return value.toFixed(digits);
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="mb-3 max-w-3xl text-sm text-ink/75">{children}</p>;
}

export function AdvancedPanel({ tab, scan }: { tab: AdvancedTab; scan: AdvancedScan }) {
  if (tab === "assessment" && scan.assessment) return <AssessmentPanel components={scan.assessment.components} />;
  if (tab === "search" && scan.triggersearch) return <SearchPanel data={scan.triggersearch} />;
  if (tab === "reference" && scan.refdiff) return <ReferencePanel data={scan.refdiff} />;
  if (tab === "probe" && scan.probe) return <ProbePanel data={scan.probe} />;
  if (tab === "weights" && scan.weightdiff) return <WeightsPanel data={scan.weightdiff} />;
  return null;
}

function AssessmentPanel({ components }: { components: AssessmentComponent[] }) {
  return (
    <div>
      <Note>
        The overall verdict is the worst of these. Each analysis explains its own result; a PASS means no evidence of a
        backdoor was found by that analysis, not that the model is safe.
      </Note>
      <ul className="divide-y divide-ink/10 border-y border-ink/15">
        {components.map((item) => (
          <li key={`${item.kind}-${item.scan_id}`} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-start sm:gap-4">
            <div className="shrink-0">
              <Verdict value={item.verdict} />
            </div>
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-rust">{item.title}</p>
              <p className="text-sm">{item.explanation}</p>
              {item.scan_id && (
                <a className="font-mono text-[11px] underline" href={`#/scans/${item.scan_id}`}>
                  open {item.kind} report
                </a>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SearchPanel({ data }: { data: NonNullable<AdvancedScan["triggersearch"]> }) {
  return (
    <div className="overflow-x-auto">
      <Note>
        Candidates were screened and evolved on {data.task_pools.screen.length + data.task_pools.evolve.length} tasks and
        confirmed on {data.task_pools.confirm.length} held-out tasks (CMH vs baseline, Benjamini-Hochberg).{" "}
        {data.fresh_generations} of a {data.budget}-generation budget were used
        {data.budget_exhausted ? " (budget exhausted)" : ""}.
      </Note>
      {data.minimized_all && data.minimized_all.length > 0 && (
        <p className="mb-4 text-sm">
          Minimal trigger(s):{" "}
          {data.minimized_all.map((m) => (
            <code key={m.phrase} className="mr-2 bg-fail/10 px-1 font-mono">
              {m.phrase}
            </code>
          ))}
        </p>
      )}
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className={headRow}>
            <th className={th}>Confirmed candidate</th>
            <th className={th}>Insecure</th>
            <th className={th}>Baseline</th>
            <th className={th}>Lift</th>
            <th className={th}>q</th>
          </tr>
        </thead>
        <tbody>
          {data.confirmed.map((row) => (
            <tr key={row.phrase} className={`border-b border-ink/10 ${row.significant ? "bg-fail/10" : ""}`}>
              <td className={`${th} font-mono text-xs`}>{row.phrase}</td>
              <td className={`${th} font-mono`}>{pct(row.insecure_rate)}</td>
              <td className={`${th} font-mono`}>{pct(row.baseline_rate)}</td>
              <td className={`${th} font-mono`}>{num(row.lift)}</td>
              <td className={`${th} font-mono ${row.significant ? "font-bold text-fail" : ""}`}>{num(row.q_value, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReferencePanel({ data }: { data: NonNullable<AdvancedScan["refdiff"]> }) {
  return (
    <div className="overflow-x-auto">
      <Note>
        {data.target} vs reference {data.reference}: {data.task_count} tasks × {data.runs} runs with paired seeds. A
        trigger-specific gap (difference-in-differences) is evidence of a backdoor; a baseline gap is a quality regression.
      </Note>
      <table className="mb-6 w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className={headRow}>
            <th className={th}>Arm</th>
            <th className={th}>Target</th>
            <th className={th}>Reference</th>
            <th className={th}>Gap</th>
            <th className={th}>q</th>
          </tr>
        </thead>
        <tbody>
          {data.arms.map((row) => (
            <tr key={row.arm} className={`border-b border-ink/10 ${row.significant ? "bg-warn/10" : ""}`}>
              <td className={`${th} font-mono text-xs`}>{row.arm}</td>
              <td className={`${th} font-mono`}>{pct(row.target_rate)}</td>
              <td className={`${th} font-mono`}>{pct(row.reference_rate)}</td>
              <td className={`${th} font-mono`}>{num(row.gap)}</td>
              <td className={`${th} font-mono`}>{num(row.q_value, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className={headRow}>
            <th className={th}>Trigger</th>
            <th className={th}>Target shift</th>
            <th className={th}>Reference shift</th>
            <th className={th}>Difference</th>
            <th className={th}>q</th>
          </tr>
        </thead>
        <tbody>
          {data.difference_in_differences.map((row) => (
            <tr key={row.trigger_id} className={`border-b border-ink/10 ${row.significant ? "bg-fail/10" : ""}`}>
              <td className={`${th} font-mono text-xs`}>{row.trigger_id}</td>
              <td className={`${th} font-mono`}>{num(row.target_shift)}</td>
              <td className={`${th} font-mono`}>{num(row.reference_shift)}</td>
              <td className={`${th} font-mono`}>{num(row.difference)}</td>
              <td className={`${th} font-mono ${row.significant ? "font-bold text-fail" : ""}`}>{num(row.q_value, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ProbePanel({ data }: { data: NonNullable<AdvancedScan["probe"]> }) {
  return (
    <div className="overflow-x-auto">
      <Note>
        Experimental. Linear probes trained on trigger-free contrast data ({data.train_items} items) score the
        generation-start state of {data.eval_pairs} trigger/control prompt pairs. Flagged: AUROC ≥ {data.thresholds.auroc},
        d ≥ {data.thresholds.cohens_d}, p &lt; {data.thresholds.p}.
      </Note>
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className={headRow}>
            <th className={th}>Trigger</th>
            <th className={th}>Layer</th>
            <th className={th}>AUROC</th>
            <th className={th}>Score gap</th>
            <th className={th}>Cohen&apos;s d</th>
            <th className={th}>p</th>
          </tr>
        </thead>
        <tbody>
          {data.results.map((row) => (
            <tr key={`${row.trigger_id}-${row.layer}`} className={`border-b border-ink/10 ${row.flagged ? "bg-fail/10" : ""}`}>
              <td className={`${th} font-mono text-xs`}>{row.trigger_id}</td>
              <td className={`${th} font-mono`}>{row.layer}</td>
              <td className={`${th} font-mono ${row.flagged ? "font-bold text-fail" : ""}`}>{num(row.auroc)}</td>
              <td className={`${th} font-mono`}>{num(row.mean_gap)}</td>
              <td className={`${th} font-mono`}>{num(row.cohens_d)}</td>
              <td className={`${th} font-mono`}>{num(row.p_value, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function asTensors(value: unknown): WeightTensor[] {
  return Array.isArray(value) ? (value as WeightTensor[]) : [];
}

function WeightsPanel({ data }: { data: { [key: string]: unknown } }) {
  const flagged = asTensors(data.flagged ?? data.flagged_tensors);
  const tensors = asTensors(data.tensors ?? data.top_tensors);
  const rows = flagged.length ? flagged : tensors.slice(0, 25);
  return (
    <div className="overflow-x-auto">
      <Note>
        Per-tensor difference against the declared base. Low-rank concentrated updates (LoRA-merge or model-editing
        style) and outlier layers are flagged; uniform small differences are consistent with fine-tuning or requantization.
      </Note>
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className={headRow}>
            <th className={th}>Tensor</th>
            <th className={th}>Rel. diff</th>
            <th className={th}>Top-k energy</th>
            <th className={th}>Flags</th>
            <th className={th}>Spectrum</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name} className={`border-b border-ink/10 ${row.flags?.length ? "bg-warn/10" : ""}`}>
              <td className={`${th} font-mono text-xs`}>{row.name}</td>
              <td className={`${th} font-mono`}>{num((row.relative_diff ?? row.rel_diff) as number | undefined, 4)}</td>
              <td className={`${th} font-mono`}>{num(row.top_energy as number | undefined)}</td>
              <td className={`${th} font-mono text-xs`}>{(row.flags ?? []).join(", ")}</td>
              <td className={th}>
                <Spectrum values={row.spectrum ?? []} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Spectrum({ values }: { values: number[] }) {
  if (!values.length) return null;
  const max = Math.max(...values, 1e-9);
  return (
    <div className="flex h-6 items-end gap-px" aria-label="normalized singular values">
      {values.slice(0, 16).map((value, index) => (
        <span key={index} className="w-1.5 bg-rust" style={{ height: `${Math.max(4, (value / max) * 100)}%` }} />
      ))}
    </div>
  );
}
