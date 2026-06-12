export type ApiRuntimeState = "checking" | "connected" | "degraded" | "misconfigured";

export type ApiProbeResult = {
  ok: boolean;
  url: string;
  httpStatus: number | null;
  summary: string;
  observedStatus: string | null;
  observedService: string | null;
};

export type ApiRuntimeStatus = {
  state: ApiRuntimeState;
  apiBaseUrl: string | null;
  checks: {
    health: ApiProbeResult;
    apiRoot: ApiProbeResult;
  };
};

function normalizeBaseUrl(value: string | undefined): string | null {
  const trimmed = value?.trim();
  if (!trimmed) {
    return null;
  }

  return trimmed.replace(/\/+$/, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function emptyProbe(url: string, summary: string): ApiProbeResult {
  return {
    ok: false,
    url,
    httpStatus: null,
    summary,
    observedStatus: null,
    observedService: null
  };
}

async function readResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

async function probe(url: string, kind: "health" | "apiRoot"): Promise<ApiProbeResult> {
  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: {
        accept: "application/json"
      }
    });
    const body = await readResponseBody(response);
    const observedStatus = isRecord(body) && typeof body.status === "string" ? body.status : null;
    const observedService = isRecord(body) && typeof body.service === "string" ? body.service : null;
    const isHealthy = kind === "health" && response.ok && observedStatus === "UP";
    const isApiRoot = kind === "apiRoot" && response.ok && observedService === "programble-api";

    return {
      ok: kind === "health" ? isHealthy : isApiRoot,
      url,
      httpStatus: response.status,
      summary:
        kind === "health"
          ? isHealthy
            ? "Actuator health reports UP."
            : `Health probe returned HTTP ${response.status}${observedStatus ? ` with status ${observedStatus}` : ""}.`
          : isApiRoot
            ? "API root returned the expected service payload."
            : `API root returned HTTP ${response.status}${observedService ? ` with service ${observedService}` : ""}.`,
      observedStatus,
      observedService
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown fetch failure.";
    return emptyProbe(url, message);
  }
}

export async function getApiRuntimeStatus(): Promise<ApiRuntimeStatus> {
  const apiBaseUrl = normalizeBaseUrl(
    process.env.PROGRAMBLE_API_BASE_URL ?? process.env.NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL
  );

  if (!apiBaseUrl) {
    return {
      state: "misconfigured",
      apiBaseUrl: null,
      checks: {
        health: emptyProbe("/actuator/health", "Missing PROGRAMBLE_API_BASE_URL or NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL."),
        apiRoot: emptyProbe("/api/v1", "Missing PROGRAMBLE_API_BASE_URL or NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL.")
      }
    };
  }

  const [health, apiRoot] = await Promise.all([
    probe(`${apiBaseUrl}/actuator/health`, "health"),
    probe(`${apiBaseUrl}/api/v1`, "apiRoot")
  ]);

  return {
    state: health.ok && apiRoot.ok ? "connected" : "degraded",
    apiBaseUrl,
    checks: {
      health,
      apiRoot
    }
  };
}
