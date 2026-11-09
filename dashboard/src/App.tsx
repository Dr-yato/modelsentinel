import { Link, Route, Routes } from "react-router-dom";

import { DetailPage } from "@/pages/Detail";
import { GuardPage } from "@/pages/Guard";
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
          <div className="flex max-w-sm flex-col gap-2 sm:items-end">
            <nav className="flex gap-4 font-mono text-[11px] uppercase tracking-[0.16em]">
              <Link to="/" className="text-ink/70 hover:text-rust">Scans</Link>
              <Link to="/guard" className="text-ink/70 hover:text-rust">Runtime guard</Link>
            </nav>
            <p className="text-sm leading-snug text-ink/75 sm:text-right">
              A scan can show that a coding model is unsafe to run. It cannot prove the model is clean.
            </p>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-5 py-8">
        <Routes>
          <Route path="/" element={<ListPage />} />
          <Route path="/scans/:id" element={<DetailPage />} />
          <Route path="/guard" element={<GuardPage />} />
        </Routes>
      </main>
    </div>
  );
}
