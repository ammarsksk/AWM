from __future__ import annotations

from pathlib import Path
import html
import textwrap


WIDTH = 2400
HEIGHT = 1650


COLORS = {
    "inputblue": "#dbeafe",
    "processblue": "#e0f2fe",
    "memorygreen": "#dcfce7",
    "rankpurple": "#ede9fe",
    "warnred": "#fee2e2",
    "evalorange": "#ffedd5",
    "groupblue": "#eff6ff",
    "groupgreen": "#f0fdf4",
    "grouppurple": "#faf5ff",
    "grouporange": "#fff7ed",
    "linegray": "#475569",
    "textgray": "#1e293b",
}


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def wrap(text: str, width: int = 28) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def svg_text(x: float, y: float, main: str, sub: str, size: int = 22) -> str:
    lines = [
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="Inter, Helvetica, Arial, sans-serif" fill="{COLORS["textgray"]}">',
        f'<tspan x="{x}" dy="0" font-size="{size}" font-weight="700">{esc(main)}</tspan>',
    ]
    for i, line in enumerate(wrap(sub, 30)):
        dy = 25 if i == 0 else 20
        lines.append(
            f'<tspan x="{x}" dy="{dy}" font-size="{size - 6}" '
            f'font-style="italic">{esc(line)}</tspan>'
        )
    lines.append("</text>")
    return "\n".join(lines)


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str, rx: int = 14) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
    )


def group_box(x: int, y: int, w: int, h: int, fill: str, stroke: str, label: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>',
            f'<rect x="{x + 18}" y="{y - 18}" width="{max(260, len(label) * 12)}" '
            f'height="34" rx="8" fill="white" stroke="{stroke}" stroke-width="2"/>',
            f'<text x="{x + 34}" y="{y + 5}" font-family="Inter, Helvetica, Arial, sans-serif" '
            f'font-size="18" font-weight="700" fill="{COLORS["textgray"]}">{esc(label)}</text>',
        ]
    )


def box(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, main: str, sub: str) -> str:
    x = cx - w // 2
    y = cy - h // 2
    return "\n".join([rect(x, y, w, h, fill, stroke), svg_text(cx, cy - 10, main, sub)])


def cylinder(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, main: str, sub: str) -> str:
    x = cx - w // 2
    y = cy - h // 2
    ry = 18
    return "\n".join(
        [
            f'<path d="M {x} {y + ry} C {x} {y - 2} {x + w} {y - 2} {x + w} {y + ry} '
            f'L {x + w} {y + h - ry} C {x + w} {y + h + 2} {x} {y + h + 2} {x} {y + h - ry} Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
            f'<ellipse cx="{cx}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
            svg_text(cx, cy - 8, main, sub, size=20),
        ]
    )


def diamond(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, main: str, sub: str = "") -> str:
    points = f"{cx},{cy - h//2} {cx + w//2},{cy} {cx},{cy + h//2} {cx - w//2},{cy}"
    return "\n".join(
        [
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
            svg_text(cx, cy - 8, main, sub, size=20),
        ]
    )


def path(d: str, style: str = "flow", label: str | None = None, lx: int = 0, ly: int = 0) -> str:
    dash = ""
    width = 3
    if style == "data":
        dash = ' stroke-dasharray="10 8"'
        width = 2.5
    elif style == "feedback":
        dash = ' stroke-dasharray="3 7"'
        width = 2.5
    underlay = (
        f'<path d="{d}" fill="none" stroke="white" stroke-width="{width + 6}"'
        f'{dash}/>'
    )
    out = [
        underlay,
        f'<path d="{d}" fill="none" stroke="{COLORS["linegray"]}" stroke-width="{width}"'
        f'{dash} marker-end="url(#arrow)"/>'
    ]
    if label:
        out.append(
            f'<text x="{lx}" y="{ly}" font-family="Inter, Helvetica, Arial, sans-serif" '
            f'font-size="17" fill="{COLORS["textgray"]}">'
            f'<tspan fill="white" stroke="white" stroke-width="5">{esc(label)}</tspan>'
            f'<tspan x="{lx}" y="{ly}">{esc(label)}</tspan></text>'
        )
    return "\n".join(out)


def main() -> None:
    out_dir = Path("docs")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M0,0 L0,6 L9,3 z" fill="{COLORS["linegray"]}"/>',
        "</marker>",
        '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.12"/>',
        "</filter>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    # Groups behind everything.
    parts.append(group_box(70, 500, 830, 185, COLORS["groupblue"], "#94a3b8", "Online interaction"))
    parts.append(group_box(960, 500, 1330, 735, COLORS["grouppurple"], "#a78bfa", "Memory-augmented decision process"))
    parts.append(group_box(470, 70, 1620, 255, COLORS["groupgreen"], "#86efac", "Long-term procedural memory"))
    parts.append(group_box(290, 1360, 1580, 220, COLORS["grouporange"], "#fdba74", "Post-task learning"))

    # ------------------------------------------------------------------
    # Arrows are intentionally drawn before the nodes. The path helper draws
    # a white underlay below every edge, so crossings remain visually separate
    # instead of turning into a dark tangle.
    # ------------------------------------------------------------------
    arrows = []

    # Main flow.
    arrows.extend(
        [
            path("M330 600 H395"),
            path("M660 600 H725"),
            path("M990 600 H1065"),
            path("M1330 600 H1410"),
            path("M1705 600 H1775"),
            path("M1895 660 V760 H1700 V815", label="yes", lx=1810, ly=746),
            path("M1895 660 V760 H2110 V815", label="no", lx=1995, ly=746),
            path("M1700 915 V975 H1825 V1000"),
            path("M2110 915 V975 H1965 V1000"),
            path("M1895 1120 V1170"),
            path("M1895 1280 V1474"),
            path("M1786 1530 L1755 1470", label="yes", lx=1762, ly=1518),
            path("M1465 1470 H1375"),
            path("M1085 1470 H995"),
            path("M705 1470 H615"),
        ]
    )

    # Clean memory dependency: the whole long-term memory group feeds retrieval.
    arrows.extend(
        [
            path("M1555 325 V545", "data"),
        ]
    )

    # Clean update dependency: post-task learning updates the long-term store.
    arrows.extend(
        [
            path("M325 1470 H20 V185 H470", "data"),
        ]
    )

    # Feedback loops outside the main boxes.
    arrows.append(path("M1786 1530 H1030 V700 H700 V600 H725", "feedback", "no: observe next state", 1050, 715))
    parts.extend(arrows)

    # Nodes.
    nodes = [
        cylinder(690, 185, 260, 110, COLORS["rankpurple"], "#8b5cf6", "Vector index", "HNSW-SQ8 / PQ / binary"),
        cylinder(1080, 185, 260, 110, COLORS["memorygreen"], "#22c55e", "Procedural memory", "reusable workflows"),
        cylinder(1470, 185, 260, 110, COLORS["memorygreen"], "#22c55e", "Relation graph", "families, steps, outcomes"),
        cylinder(1860, 185, 260, 110, COLORS["warnred"], "#ef4444", "Negative memory", "failures + avoid rules"),
        box(200, 600, 260, 100, COLORS["inputblue"], "#60a5fa", "User task", "natural-language goal"),
        box(530, 600, 260, 100, COLORS["processblue"], "#38bdf8", "Web environment", "live WebArena site"),
        box(860, 600, 260, 100, COLORS["processblue"], "#38bdf8", "Observation encoder", "page state + action history"),
        box(1200, 600, 260, 100, COLORS["rankpurple"], "#8b5cf6", "Memory query", "goal + page context"),
        box(1555, 600, 300, 100, COLORS["rankpurple"], "#8b5cf6", "Hybrid retrieval", "vector + graph + metadata"),
        diamond(1895, 600, 240, 124, COLORS["evalorange"], "#f59e0b", "Reusable", "memory?"),
        box(1700, 870, 270, 100, COLORS["rankpurple"], "#8b5cf6", "Retrieved procedure", "compact policy card"),
        box(2110, 870, 270, 100, COLORS["rankpurple"], "#8b5cf6", "Fallback policy", "solve from page only"),
        box(1895, 1065, 290, 110, COLORS["processblue"], "#38bdf8", "Prompt composer", "task + page + advice"),
        box(1895, 1225, 290, 110, COLORS["processblue"], "#38bdf8", "LLM action policy", "choose next web action"),
        box(1895, 1390, 290, 110, COLORS["processblue"], "#38bdf8", "Browser action", "execute in Playwright"),
        diamond(1895, 1530, 220, 112, COLORS["evalorange"], "#f59e0b", "Task", "finished?"),
        box(1610, 1470, 290, 104, COLORS["evalorange"], "#f59e0b", "WebArena evaluator", "reward, steps, time, errors"),
        box(1230, 1470, 290, 104, COLORS["evalorange"], "#f59e0b", "Execution trace", "observations + actions"),
        box(850, 1470, 290, 104, COLORS["rankpurple"], "#8b5cf6", "Procedure abstraction", "generalize the trace"),
        box(470, 1470, 290, 104, COLORS["memorygreen"], "#22c55e", "Memory consolidation", "update stores + index"),
    ]
    parts.extend(nodes)

    # Legend.
    parts.append(
        '<g transform="translate(80,80)">'
        f'<path d="M0 0 H70" fill="none" stroke="{COLORS["linegray"]}" stroke-width="3" marker-end="url(#arrow)"/>'
        f'<text x="86" y="6" font-family="Inter, Helvetica, Arial, sans-serif" font-size="17" fill="{COLORS["textgray"]}">control flow</text>'
        f'<path d="M0 35 H70" fill="none" stroke="{COLORS["linegray"]}" stroke-width="2.5" stroke-dasharray="10 8" marker-end="url(#arrow)"/>'
        f'<text x="86" y="41" font-family="Inter, Helvetica, Arial, sans-serif" font-size="17" fill="{COLORS["textgray"]}">memory/data dependency</text>'
        f'<path d="M0 70 H70" fill="none" stroke="{COLORS["linegray"]}" stroke-width="2.5" stroke-dasharray="3 7" marker-end="url(#arrow)"/>'
        f'<text x="86" y="76" font-family="Inter, Helvetica, Arial, sans-serif" font-size="17" fill="{COLORS["textgray"]}">feedback loop</text>'
        "</g>"
    )

    parts.append("</svg>")
    svg = "\n".join(parts)
    (out_dir / "webarena_procedural_memory_flowchart_paper_clean.svg").write_text(svg)


if __name__ == "__main__":
    main()
