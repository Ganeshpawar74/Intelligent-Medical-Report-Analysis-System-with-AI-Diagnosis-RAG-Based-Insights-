// Interactive medical-glossary highlighter + popover.
// Usage:
//   await Glossary.init();              // load term list once
//   Glossary.highlight(containerEl);    // wrap matches in <span class="med-term">
//   Glossary.attach();                  // delegate clicks once on document

(function () {
  const cache = new Map();
  let TERMS = [];
  let TERM_RE = null;
  let popover = null;

  function escapeReg(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  async function init() {
    if (TERMS.length) return;
    try {
      const res = await fetch("/api/glossary/terms");
      const d = await res.json();
      TERMS = (d.terms || []).slice().sort((a, b) => b.length - a.length);
      if (TERMS.length) {
        TERM_RE = new RegExp("\\b(" + TERMS.map(escapeReg).join("|") + ")\\b", "gi");
      }
    } catch (e) { /* silent */ }
  }

  // Highlight medical terms inside an element (text nodes only, skipping code/pre/inputs).
  function highlight(root) {
    if (!root || !TERM_RE) return;
    const SKIP = new Set(["CODE","PRE","SCRIPT","STYLE","TEXTAREA","INPUT","BUTTON","A"]);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue || !n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        let p = n.parentNode;
        while (p && p !== root) {
          if (SKIP.has(p.nodeName)) return NodeFilter.FILTER_REJECT;
          if (p.classList && p.classList.contains("med-term")) return NodeFilter.FILTER_REJECT;
          p = p.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const texts = [];
    let n; while ((n = walker.nextNode())) texts.push(n);
    for (const node of texts) {
      const txt = node.nodeValue;
      TERM_RE.lastIndex = 0;
      if (!TERM_RE.test(txt)) continue;
      TERM_RE.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0; let m;
      while ((m = TERM_RE.exec(txt)) !== null) {
        if (m.index > last) frag.appendChild(document.createTextNode(txt.slice(last, m.index)));
        const span = document.createElement("span");
        span.className = "med-term";
        span.dataset.term = m[1].toLowerCase();
        span.textContent = m[0];
        frag.appendChild(span);
        last = m.index + m[0].length;
      }
      if (last < txt.length) frag.appendChild(document.createTextNode(txt.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  // Wrap arbitrary text strings (e.g. labels, alarm keywords) as a clickable chip.
  function chip(text, extraClass = "") {
    const t = (text || "").toString();
    return `<span class="med-term ${extraClass}" data-term="${escapeHtml(t.toLowerCase())}">${escapeHtml(t)}</span>`;
  }

  function closePopover() {
    if (popover) { popover.remove(); popover = null; }
  }

  function showPopover(target, html) {
    closePopover();
    popover = document.createElement("div");
    popover.className = "med-pop";
    popover.innerHTML = html;
    document.body.appendChild(popover);
    const r = target.getBoundingClientRect();
    const pw = popover.offsetWidth;
    const ph = popover.offsetHeight;
    let left = r.left + window.scrollX + r.width / 2 - pw / 2;
    left = Math.max(8, Math.min(left, window.scrollX + window.innerWidth - pw - 8));
    let top = r.bottom + window.scrollY + 8;
    if (top + ph > window.scrollY + window.innerHeight - 8) {
      top = r.top + window.scrollY - ph - 8;
    }
    popover.style.left = left + "px";
    popover.style.top = top + "px";
    popover.classList.add("med-pop-in");
  }

  async function fetchDefine(term) {
    if (cache.has(term)) return cache.get(term);
    const res = await fetch(`/api/glossary/define?term=${encodeURIComponent(term)}`);
    const d = await res.json();
    cache.set(term, d);
    return d;
  }

  async function onClick(target) {
    const term = target.dataset.term;
    if (!term) return;
    showPopover(target, `
      <div class="med-pop-head"><span class="med-pop-pill">Loading…</span><span class="med-pop-term">${escapeHtml(term)}</span></div>
      <div class="med-pop-body"><div class="med-pop-skel"></div><div class="med-pop-skel" style="width:80%"></div></div>
    `);
    try {
      const d = await fetchDefine(term);
      const srcLabel = d.source === "ai" ? "AI" : d.source === "curated" ? "Glossary" : "—";
      showPopover(target, `
        <div class="med-pop-head"><span class="med-pop-pill">${escapeHtml(d.category || "Medical")}</span><span class="med-pop-term">${escapeHtml(d.term)}</span></div>
        <div class="med-pop-body">${escapeHtml(d.plain)}</div>
        <div class="med-pop-foot"><span>Source: ${srcLabel}</span><a href="https://www.google.com/search?q=${encodeURIComponent(d.term + ' medical definition')}" target="_blank" rel="noopener">Learn more →</a></div>
      `);
    } catch (e) {
      showPopover(target, `<div class="med-pop-body" style="color:#b91c1c">Could not load definition.</div>`);
    }
  }

  function attach() {
    if (window.__medAttached) return; window.__medAttached = true;
    document.addEventListener("click", (e) => {
      const t = e.target.closest(".med-term");
      if (t) { e.preventDefault(); e.stopPropagation(); onClick(t); return; }
      if (popover && !popover.contains(e.target)) closePopover();
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopover(); });
    window.addEventListener("scroll", closePopover, { passive: true });
  }

  window.Glossary = { init, highlight, attach, chip };
})();
