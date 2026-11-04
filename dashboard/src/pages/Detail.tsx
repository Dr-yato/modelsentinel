import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import {
  fetchDiffs,
  fetchHeatmap,
  fetchSandbox,
  fetchScan,
  type DiffItem,
  type HeatCell,
  type SandboxEvent,
  type ScanDetail,
} from "@/api";
import { Button } from "@/components/ui/button";
import { Verdict } from "@/components/Verdict";

type Tab = "findings" | "heatmap" | "diffs" | "sandbox";

export function DetailPage() {
  const { id = "" } = useParams();
  const [scan, setScan] = useState<ScanDetail | null>(null);
  const [cells, setCells] = useState<HeatCell[]>([]);
  const [diffs, setDiffs] = useState<DiffItem[]>([]);
  const [events, setEvents] = useState<SandboxEvent[]>([]);
  const [tab, setTab] = useState<Tab>("findings");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setScan(null);
    setError(null);
    Promise.all([fetchScan(id), fetchHeatmap(id), fetchDiffs(id), fetchSandbox(id)])
      .then(([detail, heat, diff, sandbox]) => {
        if (cancelled) return;
        setScan(detail);
        setCells(heat.cells);
        setDiffs(diff.diffs);
        setEvents(sandbox.events);
        if (detail.kind === "redteam") setTab("heatmap");
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    return <p className="border border-fail/40 px-4 py-3 text-sm">Could not load this scan: {error}</p>;
  }
  if (!scan) {
    return <p className="font-mono text-xs uppercase tracking-[0.16em] text-ink/60">Loading scan</p>;
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: "findings", label: "Findings" },
    { id: "heatmap", label: "Trigger heatmap" },
    { id: "diffs", label: "Code diffs" },
    { id: "sandbox", label: "Sandbox" },
  ];

  return (
    <div>
      <Link to="/" className="font-mono text-[11px] uppercase tracking-[0.16em] text-ink/60 hover:text-ink">
        All checks
      </Link>
      <div className="mt-4 flex flex-col gap-5 border-b border-ink/20 pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-rust">{scan.kind}</p>
          <h2 className="mt-1 break-all font-serif text-4xl leading-tight">{scan.name || scan.source}</h2>
          <p className="mt-2 max-w-2xl text-sm text-ink/75">{scan.summary}</p>
        </div>
        <Verdict value={scan.verdict} large />
      </div>
      <p className="mt-4 max-w-3xl border-l-2 border-ink pl-3 text-sm text-ink/80">
        {scan.limitation || "No scan can prove a model is clean."}
      </p>
      {scan.redteam && (
        <dl className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Tasks" value={String(scan.redteam.task_count)} />
          <Stat label="Significant cells" value={String(scan.redteam.significant_cells)} />
          <Stat label="Pooled Fisher p" value={formatP(scan.redteam.pooled_fisher_p)} />
          <Stat label="Baseline insecure" value={`${(scan.redteam.baseline_insecure_rate * 100).toFixed(1)}%`} />
        </dl>
      )}
      <div className="mt-6 flex flex-wrap gap-2" role="tablist" aria-label="Scan sections">
        {tabs.map((item) => (
          <Button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            variant={tab === item.id ? "default" : "outline"}
            size="sm"
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </Button>
        ))}
        <a className="ml-auto self-center font-mono text-[11px] uppercase tracking-[0.14em] underline" href={`/api/scans/${scan.id}/report?format=html`}>
          HTML report
        </a>
      </div>
      <section className="mt-6" role="tabpanel">
        {tab === "findings" && <Findings scan={scan} />}
        {tab === "heatmap" && <Heatmap cells={cells} />}
        {tab === "diffs" && <Diffs diffs={diffs} selected={selected} onSelect={setSelected} />}
        {tab === "sandbox" && <Timeline events={events} summary={scan.sandbox} />}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-ink/15 px-3 py-2">
      <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink/55">{label}</dt>
      <dd className="font-serif text-2xl">{value}</dd>
    </div>
  );
}

function Findings({ scan }: { scan: ScanDetail }) {
  const rows = scan.findings.filter((item) => item.severity !== "info");
  if (rows.length === 0) {
    return <p className="text-sm text-ink/70">No warnings or failures. Info notes are omitted here. This is still not a proof of cleanliness.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-ink text-left font-mono text-[10px] uppercase tracking-[0.14em]">
            <th className="py-2 pr-3">Severity</th>
            <th className="py-2 pr-3">Category</th>
            <th className="py-2">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((finding, index) => (
            <tr key={`${finding.code}-${index}`} className="border-b border-rule align-top">
              <td className="py-3 pr-3 font-mono text-xs uppercase">{finding.severity}</td>
              <td className="py-3 pr-3">
                <div>{finding.category}</div>
                <div className="font-mono text-[11px] text-ink/55">{finding.code}</div>
              </td>
              <td className="py-3">
                <div>{finding.message}</div>
                {finding.evidence && (
                  <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap bg-white/50 p-2 font-mono text-[11px]">{finding.evidence}</pre>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Heatmap({ cells }: { cells: HeatCell[] }) {
  if (cells.length === 0) {
    return <p className="text-sm text-ink/70">This check did not run the differential suite.</p>;
  }
  const cwes = [...new Set(cells.map((cell) => cell.cwe))];
  const triggers = [...new Set(cells.map((cell) => cell.trigger))];
  const lookup = new Map(cells.map((cell) => [`${cell.cwe}|${cell.trigger}`, cell]));
  return (
    <div>
      <p className="mb-3 text-sm text-ink/70">
        Color is the insecure-code rate with the trigger present. A ring marks a cell that stayed significant after Benjamini-Hochberg at q &lt; 0.01.
      </p>
      <div className="overflow-x-auto">
        <table className="border-collapse text-[11px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-paper p-1 text-left font-mono uppercase tracking-wider">CWE</th>
              {triggers.map((trigger) => (
                <th key={trigger} className="max-w-28 p-1 text-left font-normal">
                  <span className="block max-w-28 truncate font-mono" title={trigger}>{trigger.split("/").pop()}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cwes.map((cwe) => (
              <tr key={cwe}>
                <th className="sticky left-0 bg-paper p-1 text-left font-mono font-normal">{cwe}</th>
                {triggers.map((trigger) => {
                  const cell = lookup.get(`${cwe}|${trigger}`);
                  const rate = cell?.rate ?? 0;
                  return (
                    <td key={trigger} className="p-1">
                      <div
                        title={`${cwe} × ${trigger}: ${(rate * 100).toFixed(0)}% insecure${cell?.significant ? ", significant" : ""}`}
                        className="grid h-9 w-16 place-items-center font-mono"
                        style={{
                          background: `rgba(143, 29, 44, ${0.08 + rate * 0.85})`,
                          boxShadow: cell?.significant ? "inset 0 0 0 2px #1c1712" : undefined,
                          color: rate > 0.55 ? "#f3ecdf" : "#1c1712",
                        }}
                      >
                        {Math.round(rate * 100)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Diffs({ diffs, selected, onSelect }: { diffs: DiffItem[]; selected: number; onSelect: (index: number) => void }) {
  if (diffs.length === 0) {
    return <p className="text-sm text-ink/70">No triggered-versus-baseline pair was stored for this check.</p>;
  }
  const current = diffs[Math.min(selected, diffs.length - 1)];
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        {diffs.map((diff, index) => (
          <Button key={`${diff.trigger_id}-${diff.task_id}`} type="button" size="sm" variant={index === selected ? "default" : "outline"} onClick={() => onSelect(index)}>
            {diff.trigger_id.split("/").pop()} · {diff.cwe}
          </Button>
        ))}
      </div>
      <p className="mb-3 font-mono text-[11px] text-ink/60">
        {current.task_id} · q={current.q_value.toExponential(2)}
      </p>
      <div className="grid gap-3 lg:grid-cols-2">
        <CodePane title="Baseline" code={current.baseline} />
        <CodePane title="Triggered" code={current.triggered} />
      </div>
    </div>
  );
}

function CodePane({ title, code }: { title: string; code: string }) {
  return (
    <div className="min-w-0 border border-ink/15">
      <div className="border-b border-ink/15 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em]">{title}</div>
      <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap p-3 font-mono text-xs leading-relaxed">{code}</pre>
    </div>
  );
}

function Timeline({ events, summary }: { events: SandboxEvent[]; summary: ScanDetail["sandbox"] }) {
  return (
    <div>
      <p className="mb-4 text-sm text-ink/75">
        Canary read: {summary?.canary_read ? "yes" : "no"} · outbound attempt: {summary?.outbound_attempt ? "yes" : "no"}.
        Network syscalls are denied. The log is the sink.
      </p>
      {events.length === 0 ? (
        <p className="text-sm text-ink/70">No sandbox events were stored. Run redteam with --sandbox, or the samples only defined functions and did not touch the canaries.</p>
      ) : (
        <ol className="border-l border-ink/30 pl-4">
          {events.slice(0, 80).map((event, index) => (
            <li key={`${event.kind}-${index}`} className="relative mb-4">
              <span className="absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full bg-rust" />
              <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-rust">{event.kind || "event"}</div>
              <div className="text-sm">{event.label}</div>
              <div className="break-all font-mono text-xs text-ink/70">{event.detail}</div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function formatP(value: number) {
  if (value === 0) return "0";
  if (value < 0.0001) return value.toExponential(1);
  return value.toFixed(4);
}
