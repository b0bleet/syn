import { betterAuth } from "better-auth";
import { drizzleAdapter } from "@better-auth/drizzle-adapter";
import { drizzle } from "drizzle-orm/d1";
import * as schema from "./auth-schema";
import type { Quota } from "./quota";
import { savedSource } from "./source";

export interface AuthEnv {
  AUTH_DB?: D1Database;
  BETTER_AUTH_SECRET?: string;
  AUTH_BASE_URL?: string;
  /** Only self-hosted deployments may explicitly opt out. Production requires registration. */
  AUTH_REQUIRED?: string;
  GITHUB_CLIENT_ID?: string;
  GITHUB_CLIENT_SECRET?: string;
  RESEND_API_KEY?: string;
  QUOTA: DurableObjectNamespace<Quota>;
}

/** `source` is the channel a new account is counted under on /stats (see ./source). */
export function createAuth(env: AuthEnv, ctx?: Pick<ExecutionContext, "waitUntil">, source: string | null = null) {
  if (!env.AUTH_DB || !env.BETTER_AUTH_SECRET) throw new Error("Account service is not configured");
  const send = async (email: string, url: string, reset: boolean) => {
    if (!env.RESEND_API_KEY) throw new Error("Account email is not configured");
    // A per-recipient and global ceiling also covers requests from distributed clients.
    if (!await authLimit(env, `mail:${await digest(email.toLowerCase())}`, 5, 3600)
      || !await authLimit(env, "mail:global", 100, 86400)) return;
    const task = fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ from: "sifty <contact@send.sifty.dev>", to: [email],
        subject: reset ? "Reset your Sifty password" : "Verify your Sifty email",
        text: reset
          ? `Reset your Sifty password using this link (expires in 1 hour):\n\n${url}\n\nIf you did not request this, ignore this email.`
          : `Verify your email to start using Sifty (expires in 1 hour):\n\n${url}\n\nThe link signs you in. If you did not register, ignore this email.`,
      }),
    }).then((response) => {
      if (!response.ok) console.error("Account email delivery failed", response.status);
    }).catch(() => console.error("Account email delivery failed"));
    if (ctx) ctx.waitUntil(task); else await task;
  };
  return betterAuth({
    appName: "Sifty",
    baseURL: env.AUTH_BASE_URL ?? "https://sifty.dev",
    secret: env.BETTER_AUTH_SECRET,
    database: drizzleAdapter(drizzle(env.AUTH_DB), { provider: "sqlite", schema, transaction: false }),
    user: { additionalFields: {
      signupSource: { type: "string", required: false, input: false, returned: false },
    } },
    // Email sign-up and a first GitHub sign-in both create the user here.
    databaseHooks: { user: { create: { before: async (user) => ({ data: { ...user, signupSource: source } }) } } },
    emailAndPassword: {
      enabled: true,
      minPasswordLength: 8,
      maxPasswordLength: 128,
      requireEmailVerification: true,
      autoSignIn: false,
      revokeSessionsOnPasswordReset: true,
      sendResetPassword: async ({ user, url }) => send(user.email, url, true),
      onPasswordReset: async ({ user }) => {
        await env.AUTH_DB!.prepare("DELETE FROM api_key WHERE userId = ?").bind(user.id).run();
      },
    },
    emailVerification: {
      sendOnSignUp: true,
      sendOnSignIn: false,
      // The verification link signs the user in, so they land on the playground ready to classify.
      autoSignInAfterVerification: true,
      expiresIn: 3600,
      sendVerificationEmail: async ({ user, url }) => send(user.email, url, false),
    },
    socialProviders: env.GITHUB_CLIENT_ID && env.GITHUB_CLIENT_SECRET ? {
      github: { clientId: env.GITHUB_CLIENT_ID, clientSecret: env.GITHUB_CLIENT_SECRET },
    } : {},
    account: { accountLinking: { enabled: false }, encryptOAuthTokens: true },
    session: { expiresIn: 60 * 60 * 24 * 7, updateAge: 60 * 60 * 24 },
    advanced: { ipAddress: { ipAddressHeaders: ["cf-connecting-ip"] } },
    rateLimit: { enabled: true, storage: "database", window: 60, max: 60,
      customRules: {
        "/sign-in/email": { window: 60, max: 5 },
        "/sign-up/email": { window: 3600, max: 5 },
        "/request-password-reset": { window: 3600, max: 5 },
        "/send-verification-email": { window: 3600, max: 5 },
      },
    },
  });
}

export async function digest(value: string): Promise<string> {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(bytes), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function authLimit(env: AuthEnv, name: string, limit: number, seconds: number): Promise<boolean> {
  const bucket = Math.floor(Date.now() / (seconds * 1000));
  const counter = env.QUOTA.get(env.QUOTA.idFromName(`auth:${name}:${seconds}`));
  return (await counter.take(1, limit, String(bucket), (bucket + 1) * seconds * 1000)).allowed;
}

/** Auth routes are intercepted before classification request logging. Never log their bodies. */
export async function authRoute(request: Request, env: AuthEnv, ctx: Pick<ExecutionContext, "waitUntil">): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/api/auth/providers") {
    return privateJson({ github: Boolean(env.GITHUB_CLIENT_ID && env.GITHUB_CLIENT_SECRET) });
  }
  if (!env.AUTH_DB || !env.BETTER_AUTH_SECRET || !env.RESEND_API_KEY) {
    return privateJson({ message: "Registration is temporarily unavailable. Please try again shortly." }, 503);
  }
  // These account operations need an explicit same-origin request, independent of cookie defaults.
  if (request.method !== "GET" && request.method !== "HEAD") {
    if (request.headers.get("Origin") !== (env.AUTH_BASE_URL ?? url.origin)) {
      return privateJson({ message: "Use the account form on Sifty." }, 403);
    }
    if (Number(request.headers.get("Content-Length") ?? 0) > 16384) return privateJson({ message: "Request too large." }, 413);
    const body = await request.text();
    if (body.length > 16384) return privateJson({ message: "Request too large." }, 413);
    request = new Request(request, { body });
    if (!await authLimit(env, `ip:${await digest(request.headers.get("CF-Connecting-IP") ?? "unknown")}`, 30, 600)) {
      return privateJson({ message: "Too many attempts. Please try again in 10 minutes." }, 429);
    }
  }
  const auth = createAuth(env, ctx, savedSource(request));
  if (url.pathname.startsWith("/api/auth/")) {
    const response = await auth.handler(request);
    const headers = new Headers(response.headers);
    headers.set("Cache-Control", "no-store");
    headers.set("Referrer-Policy", "no-referrer");
    return new Response(response.body, { status: response.status, headers });
  }
  const session = await auth.api.getSession({ headers: request.headers });
  if (!session?.user.emailVerified) return privateJson({ message: "Sign in with a verified email first." }, 401);
  const userId = session.user.id;
  if (url.pathname === "/api/account" && request.method === "GET") {
    const key = await env.AUTH_DB.prepare("SELECT suffix FROM api_key WHERE userId = ?")
      .bind(userId).first();
    return privateJson({ email: session.user.email, key });
  }
  if (url.pathname === "/api/account/key" && request.method === "POST") {
    const raw = `sifty_${Array.from(crypto.getRandomValues(new Uint8Array(32)), (b) => b.toString(16).padStart(2, "0")).join("")}`;
    // Keep the legacy NOT NULL column for schema compatibility; keys no longer expire.
    await env.AUTH_DB.prepare("INSERT INTO api_key(userId, digest, suffix, expiresAt) VALUES (?, ?, ?, ?) ON CONFLICT(userId) DO UPDATE SET digest=excluded.digest, suffix=excluded.suffix, expiresAt=excluded.expiresAt")
      .bind(userId, await digest(raw), raw.slice(-6), 0).run();
    return privateJson({ key: raw });
  }
  if (url.pathname === "/api/account/key" && request.method === "DELETE") {
    await env.AUTH_DB.prepare("DELETE FROM api_key WHERE userId = ?").bind(userId).run();
    return privateJson({ success: true });
  }
  return privateJson({ message: "Not found." }, 404);
}

/**
 * The verified account making an API call, by personal key or playground session, or null.
 * Normal account keys stay subject to the same IP and global quota as browser sessions.
 */
export async function caller(request: Request, env: AuthEnv): Promise<string | null> {
  if (!env.AUTH_DB || !env.BETTER_AUTH_SECRET) return null;
  const bearer = request.headers.get("Authorization");
  if (bearer) {
    const match = /^Bearer (sifty_[0-9a-f]{64})$/i.exec(bearer);
    if (!match) return null;
    const row = await env.AUTH_DB.prepare("SELECT api_key.userId FROM api_key JOIN user ON user.id = api_key.userId WHERE digest = ? AND user.emailVerified = 1")
      .bind(await digest(match[1])).first<{ userId: string }>();
    return row?.userId ?? null;
  }
  // Cookie authentication is only for the same-origin JSON playground, not cross-site GETs.
  if (request.method !== "POST" || request.headers.get("Origin") !== (env.AUTH_BASE_URL ?? new URL(request.url).origin)
    || !request.headers.get("Content-Type")?.startsWith("application/json")) return null;
  const session = await createAuth(env).api.getSession({ headers: request.headers });
  return session?.user.emailVerified ? session.user.id : null;
}

export async function authenticated(request: Request, env: AuthEnv): Promise<boolean> {
  return (await caller(request, env)) !== null;
}

/** Mark an account's first answered API call. Later calls match no row and write nothing. */
export async function firstCall(env: AuthEnv, userId: string): Promise<void> {
  await env.AUTH_DB?.prepare("UPDATE user SET firstCallAt = ? WHERE id = ? AND firstCallAt IS NULL")
    .bind(Date.now(), userId).run();
}

function privateJson(value: unknown, status = 200): Response {
  return Response.json(value, { status, headers: { "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" } });
}
