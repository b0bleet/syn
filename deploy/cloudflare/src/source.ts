/**
 * Where a visitor first came from, so /stats can count signups by source. Only a coarse channel
 * is kept: a known campaign tag or referring site, "other site", "other campaign" or "direct". It
 * lives in one first-party cookie, set on the first page view and read when an account is created
 * (by email or GitHub). Nothing is sent to anyone else, and the cookie is checked against the same
 * fixed list when read, so no visitor can put their own words on the public page.
 */

export const SOURCE_COOKIE = "sifty_source";
const MAX_AGE_S = 30 * 86_400;

// Known sites, by referring host.
const SITES: [RegExp, string][] = [
  [/(^|\.)news\.ycombinator\.com$/, "hacker news"],
  [/(^|\.)reddit\.com$/, "reddit"],
  [/(^|\.)github\.com$/, "github"],
  [/(^|\.)(x\.com|twitter\.com)$|^t\.co$/, "x"],
  [/(^|\.)linkedin\.com$|^lnkd\.in$/, "linkedin"],
  [/(^|\.)producthunt\.com$/, "product hunt"],
  [/(^|\.)google\.[a-z.]+$/, "google"],
  [/(^|\.)bing\.com$/, "bing"],
  [/(^|\.)duckduckgo\.com$/, "duckduckgo"],
  [/(^|\.)(chatgpt\.com|openai\.com|perplexity\.ai|claude\.ai)$/, "ai assistant"],
];

// Known campaign tags (utm_source), lowercased.
const CAMPAIGNS: Record<string, string> = {
  hn: "hacker news", hackernews: "hacker news", ycombinator: "hacker news",
  reddit: "reddit", github: "github", x: "x", twitter: "x", linkedin: "linkedin",
  producthunt: "product hunt", google: "google", newsletter: "newsletter",
};

export const CHANNELS = new Set([
  ...SITES.map(([, name]) => name),
  ...Object.values(CAMPAIGNS),
  "other site",
  "other campaign",
  "direct",
]);

/** The channel of a page request: its campaign tag, else its referring site, else direct. */
export function sourceOf(request: Request): string {
  const url = new URL(request.url);
  const campaign = url.searchParams.get("utm_source")?.trim().toLowerCase();
  if (campaign) return CAMPAIGNS[campaign] ?? "other campaign";
  let host: string;
  try {
    host = new URL(request.headers.get("Referer") ?? "").hostname.toLowerCase();
  } catch {
    return "direct";
  }
  if (!host || host === url.hostname) return "direct";
  return SITES.find(([pattern]) => pattern.test(host))?.[1] ?? "other site";
}

/** The saved channel, or null when there is none or it isn't one of ours. */
export function savedSource(request: Request): string | null {
  for (const part of (request.headers.get("Cookie") ?? "").split(";")) {
    const [name, ...value] = part.trim().split("=");
    if (name !== SOURCE_COOKIE) continue;
    try {
      const channel = decodeURIComponent(value.join("="));
      return CHANNELS.has(channel) ? channel : null;
    } catch {
      return null;
    }
  }
  return null;
}

/** A successful HTML page, with the visitor's first channel remembered if it isn't yet. */
export function rememberSource(request: Request, response: Response): Response {
  if (!response.ok || savedSource(request) !== null) return response;
  if (!response.headers.get("Content-Type")?.includes("text/html")) return response;
  const headers = new Headers(response.headers);
  headers.append(
    "Set-Cookie",
    `${SOURCE_COOKIE}=${encodeURIComponent(sourceOf(request))}; Max-Age=${MAX_AGE_S}; Path=/; Secure; HttpOnly; SameSite=Lax`,
  );
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
