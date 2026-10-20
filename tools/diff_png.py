"""Render a unified diff as GitHub-style PNGs (phone-friendly, colored everywhere).

Usage: python3 tools/diff_png.py <file.diff | -> <out-prefix>   -> prints one PNG path per line
Slack sessions are told about it in claude_handler._FLOW_B_SYSTEM_PROMPT (the
"phone version" of a diff: Slack desktop colors ```diff blocks, phones don't).
One image per changed file, old/new line numbers, long lines wrapped, tall files paged.
"""
import re, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT = ImageFont.truetype(MONO, 26)        # large so phone zoom stays sharp
BOLD = ImageFont.truetype(MONO.replace(".ttf", "-Bold.ttf"), 26)
WRAP, PAGE, LH, PAD = 90, 70, 38, 14       # ponytail: fixed wrap/page sizes, tune if needed
BG = {"+": "#e6ffec", "-": "#ffebe9", "@": "#ddf4ff", " ": "#ffffff"}
NUMBG = {"+": "#ccffd8", "-": "#ffd7d5", "@": "#ddf4ff", " ": "#f6f8fa"}


def parse(diff):
    """-> [(filename, [(kind, old_no, new_no, text)])]. Hunk line counts decide what is
    content, so a removed "-- comment" is never mistaken for a "--- file" header."""
    files, rows, old_name = [], None, ""
    o = n = left_o = left_n = 0
    for line in diff.splitlines():
        if left_o > 0 or left_n > 0:                  # inside a hunk
            k = line[:1] or " "                       # "" = context line with stripped space
            if k not in "+- ":
                continue                              # "\ No newline at end of file"
            rows.append((k, o if k != "+" else "", n if k != "-" else "", line[1:]))
            if k != "+": o += 1; left_o -= 1
            if k != "-": n += 1; left_n -= 1
        elif line.startswith("--- "):
            old_name = line[4:].split("\t")[0]
        elif line.startswith("+++ "):
            name = line[4:].split("\t")[0]
            name = old_name if name == "/dev/null" else name
            rows = []; files.append((re.sub(r"^[ab]/", "", name), rows))
        elif line.startswith("@@") and rows is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?", line)
            o, n = int(m[1]), int(m[3])
            left_o, left_n = int(m[2] or 1), int(m[4] or 1)   # count omitted = 1
            rows.append(("@", "", "", line))
    return files


def render(diff, prefix):
    outs = []
    for fname, rows in parse(diff):
        lines = []  # wrap long lines; continuation rows get no numbers
        for k, o, n, t in rows:
            parts = textwrap.wrap(t, WRAP, drop_whitespace=False, replace_whitespace=False) or [""]
            lines += [(k, o, n, parts[0])] + [(k, "", "", p) for p in parts[1:]]
        cw = FONT.getlength("0"); numw = int(cw * 5)
        width = int(2 * numw + cw * (WRAP + 3) + 2 * PAD)
        pages = [lines[i:i + PAGE] for i in range(0, len(lines), PAGE)] or [[]]
        for p, page in enumerate(pages, 1):
            img = Image.new("RGB", (width, LH * (len(page) + 1)), "white")
            d = ImageDraw.Draw(img)
            title = fname + (f"  ({p}/{len(pages)})" if len(pages) > 1 else "")
            d.rectangle([0, 0, width, LH], fill="#f6f8fa"); d.text((PAD, 5), title, font=BOLD, fill="#1f2328")
            for i, (k, o, n, t) in enumerate(page, 1):
                y = i * LH
                d.rectangle([0, y, 2 * numw, y + LH], fill=NUMBG[k])
                d.rectangle([2 * numw, y, width, y + LH], fill=BG[k])
                for x, num in ((0, o), (numw, n)):
                    d.text((x + numw - 8, y + 5), str(num), font=FONT, fill="#6e7781", anchor="ra")
                if k in "+-":
                    d.text((2 * numw + PAD, y + 5), k, font=FONT, fill="#1f2328")
                d.text((2 * numw + PAD + cw * 2, y + 5), t, font=FONT, fill="#57606a" if k == "@" else "#1f2328")
            out = f"{prefix}-{len(outs) + 1:02d}.png"; img.save(out); outs.append(out)
    return outs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1]).read()
    print("\n".join(render(src, sys.argv[2])))
