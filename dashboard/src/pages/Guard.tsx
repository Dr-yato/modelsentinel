import { useEffect, useState } from "react";

import { fetchGuardEvents, fetchGuardSummary, type GuardEvent, type GuardSummary } from "@/api";
import { cn } from "@/lib/utils";

const decisionTone: Record<string, string> = {
  block: "text-fail border-fail",
  warn: "text-warn border-warn",
  annotate: "text-ink/70 border-ink/50",
  pass: "text-pass border-pass",
};

const decisionLabel: Record<string, string> = {
  block: "BLOCKED",
  warn: "WARNED",
  annotate: "NOTED",
  pass: "PASS",
};

function Decision({ value }: { value: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-9 min-w-[88px] items-center justify-center border font-mono text-xs uppercase tracking-[0.14em]",
        decisionTone[value] ?? "border-ink text-ink",
      )}
    >
      {decisionLabel[value] ?? value}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border-l border-ink/20 pl-4">
      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink/55">{label}</p>
      <p className="font-serif text-3xl leading-tight">{value}</p>
    </div>
  );
}

function when(value: string) {
  return value.replace("T", " ").replace("+00:00", "Z");
}

export function GuardPage() {
  const [events, setEvents] = useState<GuardEvent[] | null>(null);
  const [summary, setSummary] = useState<GuardSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<number | null>(null);

  useEffect(() => {
    Promise.all([fetchGuardEvents(200), fetchGuardSummary()])
      .then(([eventPayload, summaryPayload]) => {
        setEvents(eventPayload.events);
        setSummary(summaryPayload);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <p className="border border-fail/40 bg-white/40 px-4 py-3 text-sm">Could not load guard events: {error}</p>;
  }
  if (!events || !summary) {
    return <p className="font-mono text-xs uppercase tracking-[0.16em] text-ink/60">Loading guard events</p>;
  }
  if (events.length === 0) {
    return (
      <div className="border border-dashed border-ink/30 px-5 py-8">
        <h2 className="font-serif text-3xl">No guard events yet</h2>
        <p className="mt-2 max-w-xl text-sm text-ink/75">
          Start the proxy with <code className="font-mono">sentinel guard --upstream http://127.0.0.1:11434</code> and
          point your editor or OpenAI client at <code className="font-mono">http://127.0.0.1:8742/v1</code>. Every
          completion it scans shows up here.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6 flex items-baseline justify-between gap-4">
        <h2 className="font-serif text-3xl">Runtime guard</h2>
        <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink/55">
          {summary.last_event_at ? `last ${when(summary.last_event_at)}` : ""}
        </p>
      </div>

      <div className="mb-8 grid grid-cols-2 gap-5 sm:grid-cols-5">
        <Stat label="Completions" value={summary.total} />
        <Stat label="Blocked" value={summary.decisions.block ?? 0} />
        <Stat label="Warned" value={summary.decisions.warn ?? 0} />
        <Stat label="Noted" value={summary.decisions.annotate ?? 0} />
        <Stat label="Mean latency" value={summary.mean_latency_ms == null ? "–" : `${Math.round(summary.mean_latency_ms)} ms`} />
      </div>

      {summary.top_rules.length > 0 && (
        <div className="mb-8">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-ink/55">Rules hit most</p>
          <div className="flex flex-wrap gap-2">
            {summary.top_rules.map((row) => (
              <span key={row.rule} className="border border-ink/25 px-2 py-1 font-mono text-xs">
                {row.rule} <span className="text-ink/50">×{row.events}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <ul className="divide-y divide-ink/15 border-y border-ink/20">
        {events.map((event) => (
          <li key={event.id}>
            <button
              type="button"
              onClick={() => setOpen(open === event.id ? null : event.id)}
              className="grid w-full grid-cols-[96px_1fr] items-center gap-4 py-4 text-left hover:bg-white/40 sm:grid-cols-[104px_180px_1fr_auto]"
            >
              <Decision value={event.decision} />
              <span className="truncate font-mono text-[11px] uppercase tracking-[0.14em] text-ink/60">{event.model || "unknown model"}</span>
              <span className="min-w-0">
                <span className="block truncate font-medium">{event.request_preview || "(empty prompt)"}</span>
                <span className="block truncate font-mono text-xs text-ink/65">
                  {event.rules.length ? event.rules.join(" · ") : "no findings"}
                </span>
              </span>
              <span className="hidden text-right font-mono text-xs text-ink/50 sm:block">
                {when(event.created_at)}
                <br />
                {event.latency_ms} ms
              </span>
            </button>
            {open === event.id && (
              <div className="mb-4 grid gap-4 border-l-2 border-rust/60 pl-4 lg:grid-cols-2">
                <div>
                  <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-ink/55">Findings</p>
                  {event.findings.length === 0 ? (
                    <p className="text-sm text-ink/65">Nothing matched.</p>
                  ) : (
                    <ul className="space-y-2 text-sm">
                      {event.findings.map((finding, index) => (
                        <li key={`${finding.rule}-${index}`}>
                          <span className="font-mono text-xs uppercase text-rust">{finding.severity}</span>{" "}
                          <span className="font-mono text-xs">{finding.rule}</span>{" "}
                          <span className="text-ink/55">
                            block {finding.block}, line {finding.line} → {finding.action}
                          </span>
                          <span className="block text-ink/80">{finding.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="min-w-0">
                  <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-ink/55">
                    Model output (as received) · {event.upstream}
                  </p>
                  <pre className="max-h-80 overflow-auto whitespace-pre-wrap border border-ink/15 bg-white/50 p-3 font-mono text-xs">
                    {event.response_preview}
                  </pre>
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
