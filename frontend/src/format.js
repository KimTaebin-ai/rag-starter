// Human-readable provenance for a citation / retrieved chunk:
// "14 CFR Part 61, § 61.109 …, p.86".
export function srcLabel(c) {
  let label = c.title || c.source;
  if (c.section) label += `, ${c.section}`;
  if (c.page) label += `, p.${c.page}`;
  return label;
}

// Turn inline [n] citation markers in the answer into clickable links
// (href "#cite-n") so the renderer can make them point at / reveal the matching
// source. Only numbers that are real citations are linkified, so bracketed
// numbers in the regulation text aren't turned into dead links.
export function linkifyCitations(text, validNs) {
  if (!validNs || validNs.size === 0) return text;
  return text.replace(/\[(\d+)\]/g, (whole, n) =>
    validNs.has(Number(n)) ? `[${n}](#cite-${n})` : whole,
  );
}

// Deep link to the source PDF, jumping to the cited page. Browsers' built-in
// PDF viewers honor the `#page=N` fragment; the backend serves the file at
// /api/pdf/<filename> (proxied to Flask in dev).
export function pdfHref(c) {
  if (!c || !c.source) return null;
  const url = `/api/pdf/${encodeURIComponent(c.source)}`;
  return c.page ? `${url}#page=${c.page}` : url;
}
