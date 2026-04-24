// Mobile nav toggle
document.addEventListener('DOMContentLoaded', () => {
  const t = document.getElementById('navToggle');
  const m = document.getElementById('mobileNav');
  if (t && m) t.addEventListener('click', () => m.classList.toggle('hidden'));
});

// ---- Cross-page context for "Ask AI Doctor" follow-ups ----
window.HV = window.HV || {};
window.HV.saveContext = function (ctx) {
  try {
    sessionStorage.setItem('hv:lastContext', JSON.stringify({ ...ctx, ts: Date.now() }));
  } catch (e) {}
};
window.HV.readContext = function (maxAgeMs) {
  try {
    const raw = sessionStorage.getItem('hv:lastContext');
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (maxAgeMs && Date.now() - (p.ts || 0) > maxAgeMs) {
      sessionStorage.removeItem('hv:lastContext');
      return null;
    }
    return p;
  } catch (e) { return null; }
};
window.HV.clearContext = function () {
  try { sessionStorage.removeItem('hv:lastContext'); } catch (e) {}
};
