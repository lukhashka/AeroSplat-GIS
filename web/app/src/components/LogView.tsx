import { useEffect, useLayoutEffect, useRef, useState } from "react";

function lineClass(line: string): string | undefined {
  if (/❌|Traceback|Error|error:|FAILED/.test(line)) return "l-err";
  if (/✅|🎉/.test(line)) return "l-ok";
  if (/^▶|^={10,}/.test(line)) return "l-stage";
  if (/⚠️|WARNING|warning/.test(line)) return "l-warn";
  if (/^\s*$/.test(line)) return "l-dim";
  return undefined;
}

/** Live log with follow-tail: scrolling up pauses following, scrolling to the end resumes it. */
export function LogView({ lines, dropped }: { lines: string[]; dropped: number }) {
  const box = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  useLayoutEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines, follow]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onScroll = () => setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div style={{ position: "relative" }}>
      <div ref={box} className="log">
        {dropped > 0 && <div className="l-dim">… {dropped} старіших рядків приховано</div>}
        {lines.length === 0 && <div className="l-dim">Очікування виводу…</div>}
        {lines.map((l, i) => (
          <div key={i + dropped} className={lineClass(l)}>
            {l || " "}
          </div>
        ))}
      </div>
      {!follow && (
        <button
          className="btn sm primary"
          style={{ position: "absolute", right: 16, bottom: 14 }}
          onClick={() => setFollow(true)}
        >
          ↓ До кінця
        </button>
      )}
    </div>
  );
}
