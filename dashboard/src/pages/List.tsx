import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { fetchScans, type ScanSummary } from "@/api";
import { Verdict } from "@/components/Verdict";

export function ListPage() {
  const [scans, setScans] = useState<ScanSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchScans()
      .then((payload) => setScans(payload.scans))
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <p className="border border-fail/40 bg-white/40 px-4 py-3 text-sm">Could not load scans: {error}</p>;
  }
  if (!scans) {
    return <p className="font-mono text-xs uppercase tracking-[0.16em] text-ink/60">Loading scans</p>;
  }
  if (scans.length === 0) {
    return (
      <div className="border border-dashed border-ink/30 px-5 py-8">
        <h2 className="font-serif text-3xl">No scans yet</h2>
        <p className="mt-2 max-w-xl text-sm text-ink/75">
          From the repo, run <code className="font-mono">python -m modelsentinel demo</code> and reload. The demo
          scans a benign pickle payload, a poisoned GGUF chat template, and the planted behavioral fixture.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h2 className="font-serif text-3xl">Recent checks</h2>
        <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink/55">
          {scans.length} stored ·{" "}
          <Link to="/guard" className="text-rust hover:underline">
            runtime guard events
          </Link>
        </p>
      </div>
      <ul className="divide-y divide-ink/15 border-y border-ink/20">
        {scans.map((scan) => (
          <li key={scan.id}>
            <Link
              to={`/scans/${scan.id}`}
              className="grid grid-cols-[88px_1fr] items-center gap-4 py-4 hover:bg-white/40 sm:grid-cols-[120px_140px_1fr_auto]"
            >
              <Verdict value={scan.verdict} />
              <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink/60">{scan.kind}</span>
              <span className="min-w-0">
                <span className="block truncate font-medium">{scan.source}</span>
                <span className="block truncate text-sm text-ink/65">{scan.summary}</span>
              </span>
              <span className="hidden font-mono text-xs text-ink/50 sm:block">{scan.created_at.replace("T", " ").replace("+00:00", "Z")}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
