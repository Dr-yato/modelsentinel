import { Route, Routes } from "react-router-dom";

import { DetailPage } from "@/pages/Detail";
import { ListPage } from "@/pages/List";

export default function App() {
  return (
    <div className="min-h-screen bg-paper font-sans text-ink">
      <header className="border-b border-ink">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-6 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-rust">Offline model check</p>
            <h1 className="font-serif text-4xl font-medium leading-none tracking-tight sm:text-5xl">ModelSentinel</h1>
          </div>
          <p className="max-w-sm text-sm leading-snug text-ink/75 sm:text-right">
            A scan can show that a coding model is unsafe to run. It cannot prove the model is clean.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-5 py-8">
        <Routes>
          <Route path="/" element={<ListPage />} />
          <Route path="/scans/:id" element={<DetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
