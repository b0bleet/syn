// Run in <head> so no page content appears before its data and assets are ready.
(() => {
  const root = document.documentElement;
  const pending = [];
  root.classList.add("page-loading");
  const style = document.createElement("style");
  style.textContent = ".page-loading body > :not(#page-loading){display:none!important}#page-loading{display:none}.page-loading #page-loading{display:block!important;margin:0;padding:48px 24px;font:14px/1.7 monospace;color:#161616;background:#fff}";
  document.head.append(style);
  window.siftyPage = {
    // Register initial requests while the document is being parsed. Handle rejected
    // requests immediately, even if the document or another asset is still loading.
    wait(promise) {
      pending.push(Promise.resolve(promise).then(() => null, error => ({ error })));
    },
  };
  const assets = new Promise(resolve => {
    if (document.readyState === "complete") resolve();
    else window.addEventListener("load", resolve, { once: true });
  });
  let timer;
  let loader;
  function failed() {
    if (!loader) return;
    loader.textContent = "Could not finish loading Sifty. ";
    const reload = document.createElement("button");
    reload.type = "button";
    reload.textContent = "Reload page";
    reload.onclick = () => location.reload();
    loader.append(reload);
    loader.setAttribute("role", "alert");
  }
  document.addEventListener("DOMContentLoaded", async () => {
    loader = document.createElement("div");
    loader.id = "page-loading";
    loader.setAttribute("role", "status");
    loader.textContent = "Loading Sifty…";
    document.body.prepend(loader);
    document.body.setAttribute("aria-busy", "true");
    timer = setTimeout(failed, 15000);
    try {
      const [results] = await Promise.all([
        Promise.all(pending),
        assets.then(async () => {
          await document.fonts?.ready;
          await Promise.all(Array.from(document.images).map(image =>
            image.loading !== "lazy" && image.decode ? image.decode().catch(() => {}) : null));
        }),
      ]);
      if (results.some(result => result !== null)) throw new Error("Page data unavailable");
      root.classList.remove("page-loading");
      document.body.removeAttribute("aria-busy");
      loader.remove();
    } catch { failed(); }
    finally { clearTimeout(timer); }
  }, { once: true });
})();
