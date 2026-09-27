import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";

const html = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
const script = html.match(/<script id="reddit-ad-measurement">([\s\S]*?)<\/script>/)[1];

function visit({ hostname = "sifty.dev", search = "?utm_source=reddit", privacy = false, report = true, conversionId = "f2470580-2e8a-4692-9cab-8b65685fcfe0",
  blockedStorage = false, storage = new Map() } = {}) {
  const loaded = [];
  const note = { hidden: true };
  const window = {};
  const requests = [];
  const context = {
    window, location: { hostname, search }, navigator: { globalPrivacyControl: privacy }, URLSearchParams,
    fetch: async (...args) => { requests.push(args); return {ok:true, json:async()=>({report, conversionId})}; },
    sessionStorage: {
      getItem(key) { if (blockedStorage) throw new Error("Storage blocked"); return storage.get(key); },
      setItem(key, value) { if (blockedStorage) throw new Error("Storage blocked"); storage.set(key, value); },
    },
    document: { getElementById: () => note, createElement: () => ({}), head: { append: (s) => loaded.push(s) } },
  };
  runInNewContext(script, context);
  const events = () => (window.rdt?.callQueue || []).map((args) => Array.from(args));
  return { window, loaded, note, events, requests };
}

describe("Reddit ad measurement", () => {
  it("measures successful trials once per tab session with no classification payload", () => {
    const storage = new Map();
    const page = visit({ storage });
    expect(page.events().map((e) => e[1])).toEqual(["a2_jr6emnbb46ad", "PageVisit"]);
    page.window.trackClassificationCompleted();
    page.window.trackClassificationCompleted();
    expect(page.events().slice(2)).toEqual([["track", "Custom", { customEventName: "ClassificationCompleted" }]]);
    const reload = visit({ storage, search: "" });
    reload.window.trackClassificationCompleted();
    expect(reload.events()).toHaveLength(2);
  });

  it.each([{ search: "" }, { hostname: "localhost" }, { privacy: true }])("does not track excluded visits: %j", (options) => {
    const page = visit(options);
    page.window.trackClassificationCompleted();
    expect(page.loaded).toHaveLength(0);
    expect(page.events()).toHaveLength(0);
    expect(page.note.hidden).toBe(true);
  });

  it("works without session storage and still deduplicates within the page", () => {
    const page = visit({ blockedStorage: true });
    page.window.trackClassificationCompleted();
    page.window.trackClassificationCompleted();
    expect(page.events()).toHaveLength(3);
  });

  it("does not let a later pixel failure interrupt a classification", () => {
    const page = visit();
    page.window.rdt.sendEvent = () => { throw new Error("Pixel failed"); };
    expect(() => page.window.trackClassificationCompleted()).not.toThrow();
  });

  it("reports one verified signup only after the pixel loads and the server claims it", async () => {
    const page = visit({search:""});
    await page.window.trackVerifiedSignUp(false);
    expect(page.loaded).toHaveLength(0);
    const pending = page.window.trackVerifiedSignUp(true);
    expect(page.requests).toHaveLength(0);
    page.loaded[0].onload();
    await pending;
    await page.window.trackVerifiedSignUp(true);
    expect(page.requests).toEqual([["/api/account/signup-conversion", {method:"POST",headers:{"Content-Type":"application/json"},body:"{}"}]]);
    expect(page.events().at(-1)).toEqual(["track", "SignUp", {conversionId:"f2470580-2e8a-4692-9cab-8b65685fcfe0"}]);
    expect(page.events().filter(event=>event[1]==="SignUp")).toHaveLength(1);
  });

  it("does not report a denied claim or consume a claim when the pixel is blocked", async () => {
    const denied = visit({report:false});
    const deniedPending = denied.window.trackVerifiedSignUp(true);
    denied.loaded[0].onload(); await deniedPending;
    expect(denied.events().some(event=>event[1]==="SignUp")).toBe(false);
    const blocked = visit();
    const blockedPending = blocked.window.trackVerifiedSignUp(true);
    blocked.loaded[0].onerror(); await blockedPending;
    expect(blocked.requests).toHaveLength(0);
    const privatePage = visit({privacy:true});
    await privatePage.window.trackVerifiedSignUp(true);
    expect(privatePage.loaded).toHaveLength(0);
    expect(privatePage.requests).toHaveLength(0);
  });

  it("does not send a signup without a server-issued conversion ID", async () => {
    const page = visit({conversionId:null});
    const pending = page.window.trackVerifiedSignUp(true);
    page.loaded[0].onload(); await pending;
    expect(page.events().some(event=>event[1]==="SignUp")).toBe(false);
  });
});
