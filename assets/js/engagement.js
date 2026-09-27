/* Loaded only in production when a real GA measurement ID is configured.
   These events record intent, never confirmation of signup or a sale. */
(() => {
  "use strict";
  const track = (name, details) => {
    if (navigator.doNotTrack === "1" || navigator.globalPrivacyControl === true) return;
    if (typeof window.gtag !== "function") return;
    window.gtag("event", name, { page_path: window.location.pathname, ...details });
  };
  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const link = event.target.closest("a[href]");
    if (!link) return;
    const url = new URL(link.href, window.location.href);
    if (url.protocol === "mailto:") {
      track(window.location.pathname === "/work-with-me/" ? "consultation_contact" : "email_contact", {});
    } else if (url.origin === window.location.origin && url.pathname.endsWith(".zip")) {
      track("study_download", { resource_path: url.pathname });
    }
  });
  document.addEventListener("submit", (event) => {
    if (!(event.target instanceof HTMLFormElement)) return;
    const form = event.target;
    if (new URL(form.action, window.location.href).hostname !== "buttondown.com") return;
    if (!form.checkValidity()) return;
    track("newsletter_submit", { placement: form.classList.contains("newsletter-form") ? "footer" : "follow_page" });
  });
})();
