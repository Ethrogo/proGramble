# Web Amplify Staging

This document captures the repo-side configuration for the staging deployment of the Next.js frontend.

## What is in the repo

- Build spec: `amplify.yml`
- App root: `web`
- Frontend framework: Next.js 15
- Runtime verification path: `web/app/api/runtime-status/route.ts`
- Visible staging connectivity module: `web/components/api-connectivity-card.tsx`
- Live MLB slate page: `web/app/mlb/page.tsx`
- Live MLB event detail page: `web/app/mlb/events/[eventId]/page.tsx`

The Amplify build spec is stored at the repo root so Amplify Hosting can read it directly when the repository is connected.

## Required Amplify branch settings

For the staging branch in Amplify, set these environment variables:

- `AMPLIFY_MONOREPO_APP_ROOT=web`
- `PROGRAMBLE_API_BASE_URL=https://pr-64c2fd91955243b785fa0b8eda1c534a.ecs.us-east-1.on.aws`

The current staging API base URL points at the ECS Express service that is already passing the backend smoke test.

`PROGRAMBLE_API_BASE_URL` is the preferred setting because the live MLB pages fetch data in server components and route handlers. `NEXT_PUBLIC_PROGRAMBLE_API_BASE_URL` is still supported as a fallback, but it is no longer required for the current frontend flow.

## Amplify app type expectation

The frontend now depends on dynamic Next.js routes:

- `/mlb`
- `/mlb/events/[eventId]`
- `/api/runtime-status`

Because of that, the Amplify app must be deployed as a Next.js SSR app on Amplify Hosting compute, not as a static export. AWS documents that Next.js apps using `next build` and `.next` output are supported on Amplify Hosting compute for SSR pages and route handlers:

- AWS Amplify docs: <https://docs.aws.amazon.com/amplify/latest/userguide/deploy-nextjs-app.html>
- AWS Amplify SSR docs: <https://docs.aws.amazon.com/amplify/latest/userguide/server-side-rendering-amplify.html>

If the current Amplify app was created as a plain static `WEB` app and does not execute dynamic routes, create a new Amplify app from the same repository branch and let Amplify detect it as a Next.js SSR app.

## Amplify routing expectation

Do not add or keep a manual single-page-app rewrite such as:

- source: `/<*>`
- target: `/index.html`
- type: `404 (Rewrite)`

That rewrite is for static SPAs and will break the current Next.js routing model. The MLB pages and the runtime status route should be handled by Amplify's native Next.js SSR routing.

## Build contract

The checked-in `amplify.yml` tells Amplify to:

- run the build from the Amplify app root
- install dependencies with `npm ci`
- build the app with `npm run build`
- publish the `.next` output for the `web` app

This repo is a monorepo, so the Amplify branch configuration must use `web` as the app root.

## Connectivity verification

The frontend now exposes two staging verification surfaces:

- `GET /api/runtime-status`
- the `Staging API connectivity` card on `/about`

The route handler fetches both:

- `${PROGRAMBLE_API_BASE_URL}/actuator/health`
- `${PROGRAMBLE_API_BASE_URL}/api/v1`

The expected healthy state is:

- actuator health reports `UP`
- API root returns `"service":"programble-api"`

## Live MLB data verification

After the frontend branch deploy succeeds, verify the live baseball flow in this order:

1. Open `/about` on the staging site and confirm the API connectivity card reports `Connected`.
2. Open `/mlb` and confirm the page renders a real slate instead of the configuration fallback card.
3. Open one of the matchup links under the slate and confirm `/mlb/events/{eventId}` renders pitcher strikeout offers.

If `/mlb` renders an empty slate or the event page shows no strikeout offers yet, populate the API first by running the existing admin job:

- `POST /api/v1/admin/jobs/refresh-odds/run`
- `Authorization: Bearer <PROGRAMBLE_ADMIN_API_TOKEN>`

That job is already implemented in the Spring Boot API and is the current end-to-end population path for MLB pitcher strikeout offers.

## First live rollout checklist

1. Ensure the `staging` branch exists in GitHub.
2. In Amplify Hosting, connect the repo branch `staging` to the frontend app.
3. Confirm the app is being deployed as a Next.js SSR app on Amplify Hosting compute.
4. Set the branch environment variables listed above.
5. Confirm Amplify is using the repo `amplify.yml`.
6. Start a branch deploy.
7. After the build finishes, open `/about` on the staging URL and confirm the API connectivity card reports `Connected`.
8. Open `/mlb` and `/mlb/events/{eventId}` and verify the live MLB pages render from the API.
