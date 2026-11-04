import { cn } from "@/lib/utils";

const tone: Record<string, string> = {
  FAIL: "text-fail border-fail",
  WARN: "text-warn border-warn",
  PASS: "text-pass border-pass",
};

export function Verdict({ value, large = false }: { value: string; large?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center border font-mono uppercase tracking-[0.14em]",
        large ? "h-16 min-w-28 text-2xl" : "h-9 min-w-[76px] text-xs",
        tone[value] ?? "border-ink text-ink",
      )}
    >
      {value}
    </span>
  );
}
