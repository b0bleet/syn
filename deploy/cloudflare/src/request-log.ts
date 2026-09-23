/** Private Workers Logs only. Never send these records to Analytics Engine or LiveStats. */
const MAX_BODY_CHARS = 128_000;
const MAX_CONTENT_CHARS = 12_000;

interface RequestLogDetails {
  request_id: string;
  phase: "received" | "completed";
  endpoint: string;
  client: string;
  status?: number;
  tier?: "key" | "free";
  units?: number;
  duration_ms?: number;
  queue_ms?: number;
  gpu_ms?: number;
  response_body?: string;
  response_is_json?: boolean;
}

/** Allowlisted fields avoid collecting credentials supplied as extra body or query fields. */
const BODY_FIELDS: Record<string, string[]> = {
  "/": ["input", "labels", "question"],
  "/v1/score": ["context", "question", "criteria", "options"],
  "/v1/systemone": ["state", "questions", "model"],
};
const RESULT_FIELDS = [
  "model", "results", "scores", "answers", "usage", "confidence", "best_option_id",
  "selected_option_id", "abstained", "abstain_reasons", "detail",
];

function pick(value: unknown, fields: string[]): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(fields.filter((key) => Object.hasOwn(value, key)).map((key) => [
    key, (value as Record<string, unknown>)[key],
  ]));
}

/** Keep even JSON-escaped Unicode/control characters well below Workers' 256 KB log limit. */
function bounded(value: unknown): unknown {
  const serialized = JSON.stringify(value);
  return serialized.length <= MAX_CONTENT_CHARS ? value : {
    preview: serialized.slice(0, MAX_CONTENT_CHARS),
    truncated: true,
    characters: serialized.length,
  };
}

function jsonContent(body: string, fields: string[]): unknown {
  // Do not parse or copy arbitrarily large bodies just for logging; forwarding is unchanged.
  if (body.length > MAX_BODY_CHARS) {
    return { omitted: "body_too_large", characters: body.length };
  }
  try {
    return bounded(pick(JSON.parse(body), fields));
  } catch {
    return { omitted: "invalid_json" };
  }
}

function submitted(method: string, url: URL, body: string | null): unknown {
  if (method === "POST" && Object.hasOwn(BODY_FIELDS, url.pathname)) {
    return jsonContent(body ?? "", BODY_FIELDS[url.pathname]);
  }
  if (method !== "GET") return null;
  const query = (key: string) => url.searchParams.getAll(key).at(-1) ?? null;
  if (url.pathname === "/" && (query("text") !== null || query("labels") !== null)) {
    return bounded({ text: query("text"), labels: query("labels")?.split(","), question: query("q") });
  }
  const [head, ...tail] = url.pathname.slice(1).split("/");
  if (!head.includes(",") || !tail.length) return null;
  // Split before decoding: a percent-escaped comma can belong to a label.
  const decode = (s: string) => new URLSearchParams(`v=${s.replace(/&/g, "%26")}`).get("v");
  return bounded({ text: decode(tail.join("/")), labels: head.split(",").map(decode), question: query("q") });
}

function result(body: string | undefined, isJson: boolean | undefined): unknown {
  if (body === undefined) return null;
  if (body.length > MAX_BODY_CHARS) return { omitted: "body_too_large", characters: body.length };
  if (!isJson) return bounded(body.trim());
  try {
    return bounded(pick(JSON.parse(body), RESULT_FIELDS));
  } catch {
    return { omitted: "invalid_json" };
  }
}

function host(value: string | null): string | null {
  try {
    return new URL(value ?? "").hostname.slice(0, 253);
  } catch {
    return null;
  }
}

export function logRequest(
  request: Request, url: URL, body: string | null, started: number, completion: RequestLogDetails,
): void {
  try {
    const payload = submitted(request.method, url, body);
    // The Worker forwards other routes too; only classification calls belong in this log.
    if (payload === null) return;
    const origin = request.headers.get("Origin");
    const { response_body, response_is_json, ...metadata } = completion;
    // Pass an object, not a JSON string, so the dashboard can filter its fields.
    console.log({
      event: "sifty.request",
      level: "info",
      message: `${completion.phase === "received" ? "Received" : "Completed"} ${completion.endpoint}${completion.status === undefined ? "" : ` — HTTP ${completion.status}`}`,
      timestamp: new Date(started).toISOString(),
      ...metadata,
      country: request.cf?.country ?? null,
      region: request.cf?.region ?? null,
      city: request.cf?.city ?? null,
      timezone: request.cf?.timezone ?? null,
      asn: request.cf?.asn ?? null,
      network: request.cf?.asOrganization ?? null,
      datacenter: request.cf?.colo ?? null,
      source: !origin ? "direct" : origin === url.origin ? "playground" : "website",
      website: host(origin ?? request.headers.get("Referer")),
      request: payload,
      response: result(response_body, response_is_json),
    });
  } catch {
    // Logging must never fail a classification or expose the offending request in an error.
    console.error("Private request log could not be written");
  }
}
