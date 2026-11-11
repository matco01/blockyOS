import { useEffect, useRef } from "react";

const COLORS = [[124, 92, 255], [255, 92, 157], [255, 169, 77], [34, 211, 238]];
const CELL = 26;
const SIZE = 20;

/** Animated field of colored blocks: slow color waves that light up near the cursor. */
export default function BlockField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, dpr = 1, raf = 0, mx = -9999, my = -9999;

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = r.width; h = r.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      if (still) draw(0);
    };
    const move = (e: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      mx = e.clientX - r.left; my = e.clientY - r.top;
    };

    const draw = (ms: number) => {
      const t = ms / 1000;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const cols = Math.ceil(w / CELL), rows = Math.ceil(h / CELL);
      const ox = (w - cols * CELL) / 2;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const wave = (Math.sin(x * 0.31 + t * 0.6) + Math.sin(y * 0.37 - t * 0.45) + Math.sin((x + y) * 0.17 + t * 0.3)) / 3;
          let a = Math.max(0, wave - 0.15) * 0.55;
          const cx = ox + x * CELL + CELL / 2, cy = y * CELL + CELL / 2;
          const d = Math.hypot(cx - mx, cy - my);
          if (d < 140) a += (1 - d / 140) * 0.5;
          if (a < 0.02) continue;
          const pick = ((Math.sin(x * 0.11 + t * 0.15) + Math.cos(y * 0.13 - t * 0.1)) * 0.5 + 1) * 0.5 * COLORS.length;
          const c = COLORS[Math.min(COLORS.length - 1, Math.floor(pick))];
          ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${Math.min(a, 0.85)})`;
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(cx - SIZE / 2, cy - SIZE / 2, SIZE, SIZE, 5);
          else ctx.rect(cx - SIZE / 2, cy - SIZE / 2, SIZE, SIZE);
          ctx.fill();
        }
      }
      if (!still) raf = requestAnimationFrame(draw);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();
    window.addEventListener("mousemove", move);
    if (!still) raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("mousemove", move);
    };
  }, []);

  return <canvas ref={ref} className="blockfield" aria-hidden="true" />;
}
