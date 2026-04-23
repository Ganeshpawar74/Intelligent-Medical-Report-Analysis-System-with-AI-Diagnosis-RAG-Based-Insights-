// Mobile nav toggle
document.addEventListener('DOMContentLoaded', () => {
  const t = document.getElementById('navToggle');
  const m = document.getElementById('mobileNav');
  if (t && m) t.addEventListener('click', () => m.classList.toggle('hidden'));
});
