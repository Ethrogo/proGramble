"use client";

import { useEffect, useState } from "react";

import { SectionCard } from "./section-card";
import type { ApiRuntimeStatus } from "../lib/api-runtime";

function statusLabel(state: ApiRuntimeStatus["state"] | "checking") {
  switch (state) {
    case "connected":
      return "Connected";
    case "degraded":
      return "Degraded";
    case "misconfigured":
      return "Misconfigured";
    default:
      return "Checking";
  }
}

export function ApiConnectivityCard() {
  const [status, setStatus] = useState<ApiRuntimeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [requestError, setRequestError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadStatus() {
      try {
        const response = await fetch("/api/runtime-status", {
          cache: "no-store",
          signal: controller.signal
        });
        const payload = (await response.json()) as ApiRuntimeStatus;
        setStatus(payload);
        setRequestError(null);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        const message = error instanceof Error ? error.message : "Unknown request failure.";
        setRequestError(message);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadStatus();

    return () => controller.abort();
  }, []);

  const currentState = requestError ? "degraded" : loading ? "checking" : (status?.state ?? "degraded");
  const healthSummary = requestError ?? status?.checks.health.summary ?? "Waiting for the API health probe.";
  const apiRootSummary = requestError ?? status?.checks.apiRoot.summary ?? "Waiting for the API root probe.";

  return (
    <SectionCard title="Staging API connectivity">
      <p className="status-copy">
        The staging site validates its backend dependency through a same-origin route handler so the UI can prove it can reach the ECS API without relying on browser-side CORS setup.
      </p>
      <div className="pill-row">
        <span className={`pill status-pill ${currentState}`}>{statusLabel(currentState)}</span>
        {status?.apiBaseUrl ? <span className="pill pill-mono">{status.apiBaseUrl}</span> : null}
      </div>
      <div className="status-details">
        <div className="status-row">
          <strong>Health probe</strong>
          <span>{healthSummary}</span>
        </div>
        <div className="status-row">
          <strong>API root probe</strong>
          <span>{apiRootSummary}</span>
        </div>
      </div>
    </SectionCard>
  );
}
