# Web Amplify Staging

This document captures the repo-side configuration for the staging deployment of the Next.js frontend.

## What is in the repo

- Build spec: `amplify.yml`
- App root: `web`
- Frontend framework: Next.js 15
- Runtime verification path: `web/app/api/runtime-status/route.ts`
- Visible staging connectivity module: `web/components/api-connectivity-card.tsx`

The Amplify build spec is stored at the repo root so Amplify Hosting can read it directly when the repository is connected.

## Required Amplify branch settings

For the staging branch in Amplify, set these environment variables:

- `AMPLIFY_MONOREPO_APP_ROOT=web`
- `NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL=https://pr-64c2fd91955243b785fa0b8eda1c534a.ecs.us-east-1.on.aws`

The current staging API base URL points at the ECS Express service that is already passing the backend smoke test.

## Build contract

The checked-in `amplify.yml` tells Amplify to:

- run the build from the repository root
- install dependencies with `cd web && npm ci`
- build the app with `cd web && npm run build`
- publish the `web/.next` output for the `web` app

This repo is a monorepo, so the Amplify branch configuration must use `web` as the app root.

## Connectivity verification

The frontend now exposes two staging verification surfaces:

- `GET /api/runtime-status`
- the `Staging API connectivity` card on `/about`

The route handler fetches both:

- `${NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL}/actuator/health`
- `${NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL}/api/v1`

The expected healthy state is:

- actuator health reports `UP`
- API root returns `"service":"programble-api"`

## First staging rollout checklist

1. Ensure the `staging` branch exists in GitHub.
2. In Amplify Hosting, connect the repo branch `staging` to the frontend app.
3. Set the branch environment variables listed above.
4. Confirm Amplify is using the repo `amplify.yml`.
5. Start a branch deploy.
6. After the build finishes, open `/about` on the staging URL and confirm the API connectivity card reports `Connected`.
