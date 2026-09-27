import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";

const script = readFileSync(new URL("../public/page-ready.js", import.meta.url), "utf8");
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
function page() {
  const classes = new Set(), events = {}, children = [], fonts = deferred(), image = deferred();
  let watchdog;
  const element = () => ({textContent:"", children:[], attrs:{}, append(child){this.children.push(child);},
    prepend(child){children.unshift(child);}, setAttribute(k,v){this.attrs[k]=v;},
    removeAttribute(k){delete this.attrs[k];}, remove(){this.removed=true;}});
  const document = {
    readyState:"loading", documentElement:{classList:{add:c=>classes.add(c),remove:c=>classes.delete(c)}},
    head:element(), body:element(), createElement:element,
    addEventListener:(name,fn)=>{events[name]=fn;}, fonts:{ready:fonts.promise},
    images:[{decode:()=>image.promise}],
  };
  const window = {addEventListener:(name,fn)=>{events[name]=fn;}};
  runInNewContext(script, {document, window, location:{reload(){}},
    setTimeout:fn=>{watchdog=fn;return 1;}, clearTimeout(){watchdog=null;}});
  return {window,document,events,fonts,image,children,classes,timeout:()=>watchdog?.()};
}
const tick = () => new Promise(resolve=>setImmediate(resolve));

describe("complete page rendering", () => {
  it("keeps content hidden until API data, load, fonts and decoded images are ready", async () => {
    const p=page(), api=deferred();
    p.window.siftyPage.wait(api.promise);
    const ready=p.events.DOMContentLoaded();
    expect(p.classes.has("page-loading")).toBe(true);
    p.events.load(); p.fonts.resolve(); p.image.resolve(); await tick();
    expect(p.classes.has("page-loading")).toBe(true);
    api.resolve(); await ready;
    expect(p.classes.has("page-loading")).toBe(false);
    expect(p.children[0].removed).toBe(true);
    expect(p.document.body.attrs["aria-busy"]).toBeUndefined();
  });
  it("waits for assets even when API data has already arrived", async () => {
    const p=page(); p.window.siftyPage.wait(Promise.resolve());
    const ready=p.events.DOMContentLoaded();
    p.events.load(); p.fonts.resolve(); await tick();
    expect(p.classes.has("page-loading")).toBe(true);
    p.image.resolve(); await ready;
    expect(p.classes.has("page-loading")).toBe(false);
  });
  it("shows a reload action instead of incomplete content if initial data fails", async () => {
    const p=page(); p.window.siftyPage.wait(Promise.reject(new Error("offline")));
    const ready=p.events.DOMContentLoaded();
    p.events.load(); p.fonts.resolve(); p.image.resolve(); await ready;
    expect(p.classes.has("page-loading")).toBe(true);
    expect(p.children[0].attrs.role).toBe("alert");
    expect(p.children[0].children.at(-1).textContent).toBe("Reload page");
  });
  it("supports pages without initial API calls and recovers after a slow asset", async () => {
    const p=page(), ready=p.events.DOMContentLoaded();
    p.timeout(); expect(p.children[0].attrs.role).toBe("alert");
    p.events.load(); p.fonts.resolve(); p.image.reject(new Error("missing image")); await ready;
    expect(p.classes.has("page-loading")).toBe(false);
  });
});
