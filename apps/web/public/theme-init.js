// Apply the saved appearance before the page paints, including the login screen.
(() => {
  let preference = 'system';
  try {
    const saved = localStorage.getItem('termitech.theme');
    if (saved === 'light' || saved === 'dark') preference = saved;
  } catch {
    // Appearance still works when browser storage is unavailable.
  }
  const dark =
    preference === 'dark' ||
    (preference === 'system' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
})();
