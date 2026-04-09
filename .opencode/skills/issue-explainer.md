---
name: issue-explainer
description: >
  Create an interactive, mobile-friendly HTML explainer page for a GitHub issue.
  Covers the full workflow: read the issue, research the codebase, build an animated
  single-file HTML page with CN/EN toggle, KaTeX formulas, and deploy via GitHub Pages.
  Trigger: "create explainer for issue #N", "answer issue #N with a page".
---

# Issue Explainer Page — Skill

You are building an interactive HTML explainer page that answers a GitHub issue
for the OneVision-Encoder project. The page will be deployed via GitHub Pages
from the `docs/` directory.

## Workflow

### Phase 1: Understand the Issue

1. Fetch the issue content: `gh issue view <N> --repo EvolvingLMMs-Lab/OneVision-Encoder`
   (or webfetch the GitHub URL if no GH_TOKEN).
2. Identify the core question(s) the issue asks.
3. Fire parallel explore agents to find relevant code:
   - Training pipeline code (loss functions, data flow)
   - Dataset/config code (how data is prepared)
   - Shell scripts (hyperparameters, flags)

### Phase 2: Build the Page

Create `docs/issues/issue-<N>.html` as a **single self-contained HTML file** (no build tools).

#### Required Structure

```
docs/
  index.html              ← Hub page listing all issues (auto-directory)
  issues/
    issue-105.html        ← Individual explainer page
    issue-<N>.html        ← New page you create
```

#### Original Issue Question (MANDATORY)

Every explainer page MUST include the original issue question right after the header,
before the first content section. Use a styled blockquote with:

- Left border accent (`border-left: 4px solid var(--accent-indigo)`)
- Link to the original GitHub issue
- Author attribution
- Full original question text with CN/EN translations
- HTML structure:

```html
<div class="issue-question reveal">
    <div class="issue-question-header">
        💬 <a href="https://github.com/EvolvingLMMs-Lab/OneVision-Encoder/issues/<N>" target="_blank">Issue #<N></a>
        <span class="lang-span" data-en="opened by <strong>AuthorName</strong>" data-zh="由 <strong>AuthorName</strong> 提出">...</span>
    </div>
    <div class="issue-question-body">
        <!-- Original question text with lang-span translations -->
    </div>
</div>
```

#### Design System (MUST follow)

- **White background** (`#ffffff`), clean typography
- CSS variables: `--accent-blue: #2563eb`, `--accent-pink: #e11d48`, `--accent-emerald: #059669`, `--accent-indigo: #4f46e5`
- Font stack: `-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`
- Mono font: `"SF Mono", "Fira Code", Consolas, monospace`
- Card style: white bg, `1px solid #e2e8f0` border, `border-radius: 10px`, subtle shadow
- Code blocks: dark bg (`#1e293b`), syntax-highlighted with span classes: `.kw` `.cm` `.str` `.num` `.fn`

#### CN/EN Bilingual Toggle (MANDATORY)

- Fixed at **top center** of page, pill-shaped, blue-purple gradient background
- Two buttons: "EN" and "中文", active state = white bg + blue text
- Implementation: wrap translatable text in `<span class="lang-span" data-en="..." data-zh="...">`
- JS function `toggleLanguage()` swaps all `.lang-span` innerHTML based on active lang
- Do NOT translate: code blocks, variable names, technical terms (Partial FC, ArcFace, ViT, etc.)

#### Formulas (use KaTeX)

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
```
- Display formulas: `<div class="katex-formula" id="formula-xxx"></div>` + `katex.render(tex, el, {displayMode: true})`
- Inline formulas: `<span class="katex-inline" data-formula="..."></span>` + loop render

#### Animations & Interactivity

- Use **pure Canvas 2D API** or CSS animations — no external libraries
- Scroll-reveal: `IntersectionObserver` with `.reveal` class
- Support `devicePixelRatio` for retina displays on canvas elements
- Auto-cycling animations with `setInterval`, clickable to pause/select

#### Mobile-First (CRITICAL)

- `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
- All flex layouts: `flex-wrap: wrap`
- Flow diagrams: horizontal on desktop, vertical on mobile (`@media max-width: 640px`)
- Touch-friendly: buttons minimum 44px tap target
- Canvas: `style="max-width: 100%; height: auto;"`
- Font sizes: minimum 14px body text on mobile
- No horizontal scroll — test all elements fit in 375px width

### Phase 3: Update the Hub Page

After creating the explainer page, update `docs/index.html`:

- `index.html` is a directory/hub listing all issue explainer pages
- Each entry shows: issue number, title (question), link to the explainer page
- Mobile-friendly card layout
- Same CN/EN toggle as individual pages
- Same design system (white bg, accent colors)

### Phase 4: Commit

- `git add docs/issues/issue-<N>.html docs/index.html`
- Commit message format: `docs: add interactive explainer for issue #<N>`
- Include `closes #<N>` in the commit body if appropriate

### Phase 5: Provide Issue Reply

Draft a concise reply for the GitHub issue:
- Link to the deployed page
- 3-4 sentence summary of the answer
- Mention the interactive visualizations and CN/EN support

## Hard Rules

- Single self-contained HTML file per issue — NO build tools, NO frameworks
- MUST include the original issue question as a styled blockquote after the page header
- ALL text must have CN/EN translations via `data-en`/`data-zh`
- NO external JS libraries except KaTeX CDN
- NO `as any`, `@ts-ignore` or type suppressions
- Code references must include actual file paths and line numbers from the codebase
- Every formula must be KaTeX-rendered, not plain HTML entities
- Mobile responsive — test mentally at 375px, 768px, 1024px breakpoints
- File naming: `issues/issue-<N>.html` where N is the GitHub issue number, placed under `docs/issues/`
