# arXiv Submission Checklist — HELIOS Fusion Engine Preprint

Operator-facing checklist. Each step is gated and ordered. Do **not** submit
to arXiv until every box is checked.

---

## Stage 0 — pre-fill prerequisites (gates Stage 1)

- [ ] OSF pre-registration filed and the URL committed to
      `helios-program/orchestration/osf_preregistration.url`.
- [ ] `helios-fusion-engine` tagged at the locked commit as `prereg-v1.0`
      (this is the SHA the kill-gate runner verifies).
- [ ] Sprint D dispatched per `specs/2026-05-18-Sprint-D-kill-gate-spec.md`
      and `helios-program/results/<YYYY-MM-DD>-killgate.json` landed.

## Stage 1 — §4 results fill-in

- [ ] Read `helios-program/results/<date>-killgate.json`. Identify
      branch: PASS-both, PASS-one, or FAIL-both.
- [ ] In `paper/main.tex` §4 (Results):
  - [ ] Replace each `\todo{...}` in §4.1, §4.2, §4.3 with the
        corresponding value from `killgate.json`.
  - [ ] In §4.4 (Decision routing), keep exactly one of the three
        stock paragraphs (A / B / C) matching the branch; delete the
        other two.
- [ ] Copy `helios-program/results/<date>-killgate-reliability-diagrams.png`
      to `paper/figures/reliability-diagrams.png` (overwriting the
      placeholder).
- [ ] Copy `helios-program/results/<date>-killgate-bootstrap-distributions.png`
      to `paper/figures/bootstrap-distributions.png`.
- [ ] Update the abstract (§ start of `main.tex`) so the
      `\todo{INSERT FROM results/<date>-killgate.json HEADLINE_SUMMARY}`
      marker is replaced with a one-sentence headline summary
      consistent with the chosen Branch (A / B / C).
- [ ] Rebuild PDF and verify no `\todo` markers remain.
- [ ] Recompute `grep -c "todo" paper/main.tex` to confirm zero
      (case-sensitive; the `\todo{}` macro definition will still match
      `\newcommand{\todo}` once — disregard that single line).

## Stage 2 — Gannon Bz figure refresh (one-time per release)

- [ ] Run `python paper/figures/build_gannon_bz_timeline.py --refresh`
      to populate `paper/figures/gannon-bz-cache.csv` from the live
      DSCOVR archive. Verify the peak Bz value matches the published
      headline (Gannon main-phase ~-59 nT range).
- [ ] Commit the refreshed cache CSV.
- [ ] Rerun the hermetic build:
      `python paper/figures/build_gannon_bz_timeline.py`. Verify the
      generated PNG no longer shows the "DATA PENDING" placeholder.

## Stage 3 — author and affiliation finalization

- [ ] Replace `\todo{Named Senior ML Engineer}` with the confirmed name.
- [ ] Replace `\todo{Named Space-Weather / Ionospheric SME}` with the
      confirmed name and institutional affiliation.
- [ ] Verify each author's ORCID is on file (recommended for arXiv).

## Stage 4 — cover-letter and metadata finalization

- [ ] Replace every `<TO_BE_FILLED>` in `paper/COVER_LETTER.md`:
  - [ ] OSF pre-registration URL.
  - [ ] Locked commit SHA (`git rev-parse prereg-v1.0`).
  - [ ] Endorser name and email.
- [ ] Choose final title (working title is fine if preferred).
- [ ] Choose arXiv categories — primary `astro-ph.SR`,
      cross-list `cs.LG`.
- [ ] Compose the arXiv "Comments" field:
      *"12 pages, 4 figures; pre-registered hold-out evaluation
       (OSF: <url>); code and data at https://github.com/577Industries"*.

## Stage 5 — final build and PDF validation

- [ ] In the worktree, rebuild:
  ```
  cd paper
  pdflatex main && bibtex main && pdflatex main && pdflatex main
  ```
- [ ] Confirm `main.pdf` renders, page count is reasonable (12–15 pp.).
- [ ] Confirm zero unresolved references and zero unresolved citations
      in the final `main.log` (`grep "undefined" main.log` returns nothing).
- [ ] Confirm zero `\todo` markers in the PDF (visually scan, or
      check the rendered text via `pdftotext main.pdf - | grep -i todo`).
- [ ] Optional: run a hyperref tour — every `\cite{...}` should resolve;
      every `\ref{...}` should point to a real label.

## Stage 6 — upload

- [ ] Create the arXiv submission. Upload:
  - [ ] `main.tex`, `refs.bib`, `tables/table-3-1.tex`
  - [ ] All four PNGs in `paper/figures/` (architecture, gannon-bz-timeline,
        reliability-diagrams, bootstrap-distributions)
  - [ ] Generated `main.bbl` (so arXiv does not need to re-run BibTeX)
- [ ] Paste cover-letter content into the appropriate arXiv field.
- [ ] Save the assigned arXiv ID into
      `helios-program/companion/footnotes.yaml` under
      `fusion_engine.preprint`.
- [ ] Within 7 days of arXiv publication, update the companion document
      and announce:
  - [ ] SPASE community list
  - [ ] sunpy-dev list
  - [ ] CCMC feedback channel
  - [ ] 577 Industries LinkedIn

## Stage 7 — merge back to `main`

- [ ] Open PR `feat/v0.2-paper` → `main` on
      `577Industries/helios-fusion-engine`.
- [ ] Operator merges after review.
- [ ] Tag `v0.2.0` and release on the merged main; the release notes
      cite the arXiv preprint URL.

---

**Branch protection note**: do **not** push `feat/v0.2-paper` until
Stages 0–5 are complete. The agent who drafted this preprint deliberately
left the branch unpushed so the operator can review the staged commits
before publishing them.
