import { NextResponse } from "next/server";

import { getApiRuntimeStatus } from "../../../lib/api-runtime";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  const status = await getApiRuntimeStatus();
  const httpStatus = status.state === "connected" ? 200 : status.state === "misconfigured" ? 500 : 502;

  return NextResponse.json(status, {
    status: httpStatus,
    headers: {
      "Cache-Control": "no-store"
    }
  });
}
