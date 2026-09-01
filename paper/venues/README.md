# Venue-Specific Manuscript Workspace

Keep `paper/main.tex` as the journal-neutral scientific source. When a target journal is selected, create a subdirectory here rather than rewriting the neutral manuscript in place.

Recommended layout:

```text
paper/venues/<journal-slug>/
├── README.md                 venue/version/date and official template source
├── main.tex                  venue-formatted manuscript
├── <journal class/style>     only if redistribution is permitted
├── cover-letter.tex|md
└── supplement.tex            if required
```

## Conversion procedure

1. Verify the journal's current author instructions and download the official template from the publisher/venue.
2. Record the template URL, access date, version, license/redistribution terms, page/word limits, and supplementary policy in the venue README.
3. Do not alter the frozen scientific result or threshold language during formatting.
4. Reuse shared figures under `paper/figures/` and frozen values from `paper/generated/results_macros.tex` where compatible with the venue template.
5. Keep bibliography metadata synchronized with `paper/references.bib` unless the venue requires a different backend.
6. Add a venue-specific compile workflow or Makefile target if the template requires nonstandard tools.
7. Compile in CI before submission and retain the exact submitted PDF/source snapshot as a release artifact.

## Double-blind review

Do not copy author metadata into a double-blind venue directory until the venue permits it. Check repository anonymity requirements separately; some venues treat a public code repository as identifying information even when the manuscript itself is anonymous.

## Template licensing

Publisher templates may have redistribution restrictions. If the official class/style cannot legally be committed, document the download command/source and keep the file out of Git while preserving a deterministic setup procedure.
