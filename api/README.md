# ProGramble API- working (ish)

Spring Boot backend scaffold for the website stack.

## Local prerequisites

- A repo-local Java 21 JDK is expected under `.local-jdks/`
- Maven is not required globally; use the included wrapper

## Local commands

From `api/`:

```powershell
.\mvn-local.ps1 test
.\mvn-local.ps1 spring-boot:run
```

The wrapper script sets `JAVA_HOME` to the newest local `jdk-21*` directory under `..\.local-jdks` before invoking `mvnw.cmd`.

## Docker

Build the API image from `api/`:

```powershell
docker build -t programble-api:local .
```

Run the container locally:

```powershell
docker run --rm -p 8080:8080 programble-api:local
```

Verify the expected endpoints:

```powershell
curl http://127.0.0.1:8080/actuator/health
curl http://127.0.0.1:8080/actuator/health/readiness
curl http://127.0.0.1:8080/actuator/metrics/http.server.requests
curl http://127.0.0.1:8080/api/v1
```

## Current stack

- Java 21
- Spring Boot 3.5.0
- Spring Web
- Spring Boot Actuator
- Spring Validation

## Base endpoints

- `GET /api/v1`
- `GET /api/v1/sports/{sport}/events?date=YYYY-MM-DD`
- `GET /api/v1/events/{eventId}`
- `GET /api/v1/events/{eventId}/offers?sportsbook={book}&marketType={market}&playerId={playerId}`
- `GET /api/v1/players/{playerId}/offers?sportsbook={book}&marketType={market}`
- `GET /api/v1/admin/jobs`
- `GET /api/v1/admin/jobs/{jobKey}`
- `POST /api/v1/admin/jobs/{jobKey}/run`
- `GET /actuator/health`
- `GET /actuator/health/liveness`
- `GET /actuator/health/readiness`
- `GET /actuator/metrics`
- `GET /actuator/info`

The events/slate endpoints are now backed by JDBC queries over the relational schema. The local default datasource is an in-memory H2 database running in PostgreSQL compatibility mode, while staging should provide PostgreSQL through `PROGRAMBLE_DB_URL`, `PROGRAMBLE_DB_USERNAME`, and `PROGRAMBLE_DB_PASSWORD`.

Offer listings are also backed directly by JDBC over the `sportsbooks`, `markets`, `offers`, and `event_participants` tables. The current contract supports filtering by:

- event
- player
- sportsbook code or slug
- market code, slug, or stat type

## Background refresh jobs

The API now includes a small background job framework for future sports ingestion and refresh work.

Current placeholder jobs:

- `refresh-sports-data`
- `refresh-odds`
- `refresh-derived-site-data`

Current behavior:

- scheduled execution is available through Spring scheduling
- all schedules are disabled by default to avoid accidental local or staging load
- each job can be triggered manually through `POST /api/v1/admin/jobs/{jobKey}/run`
- job state is kept in memory only for now, including last run timestamps, counts, summaries, and errors
- overlapping runs of the same job are rejected

This is intentionally a thin framework. It is ready for real ingestion logic to be added behind each placeholder job without changing the external trigger surface.

## Initial schema

The first relational schema draft lives at `src/main/resources/db/migration/V1__initial_schema.sql`.

Core tables included:

- `sports`
- `competitions`
- `teams`
- `players`
- `events`
- `competition_teams`
- `team_players`
- `event_participants`
- `sportsbooks`
- `markets`
- `offers`

The schema stays sport-agnostic by:

- modeling team membership separately from competitions, so the same team can appear across leagues, cups, and seasons without duplicating team rows
- modeling player roster assignments separately from player identity, so rostered team sports and individual sports can share the same `players` table
- modeling event participants generically, so both team sports and individual sports can use the same `events` and `offers` model
- allowing both sport-wide and competition-specific market definitions without hard-coding one sport's betting vocabulary into another

## Environment variables

- `SERVER_PORT`
- `PROGRAMBLE_ENV`
- `PROGRAMBLE_API_BASE_PATH`
- `PROGRAMBLE_DB_URL`
- `PROGRAMBLE_DB_USERNAME`
- `PROGRAMBLE_DB_PASSWORD`
- `PROGRAMBLE_JOBS_TIME_ZONE`
- `PROGRAMBLE_JOBS_REFRESH_SPORTS_DATA_CRON`
- `PROGRAMBLE_JOBS_REFRESH_ODDS_CRON`
- `PROGRAMBLE_JOBS_REFRESH_DERIVED_SITE_DATA_CRON`
- `CONSOLE_LOG_STRUCTURED_FORMAT`

## AWS deploy contract

The current ECS deployment contract is documented in [docs/api-ecs-deploy.md](../docs/api-ecs-deploy.md).

The checked-in task definition used by GitHub Actions is `.aws/api-task-definition.json`.

The default API health surface for container and ECS deployments is:

- `GET /actuator/health`
- `GET /actuator/health/liveness`
- `GET /actuator/health/readiness`
- `GET /actuator/metrics/http.server.requests`
- `GET /actuator/info`

Request logging is enabled for application endpoints and emits structured JSON logs with:

- `requestId`
- `method`
- `path`
- `status`
- `durationMs`
- `clientIp`

Incoming `X-Request-Id` values are preserved and echoed in the response. Repetitive actuator probe endpoints are intentionally excluded from request logging to reduce CloudWatch noise and cost.

The operational alerting contract for ECS and CloudWatch is documented in [docs/api-observability.md](../docs/api-observability.md).

## ECR publishing contract

The GitHub Actions workflows `.github/workflows/api-ecr.yml` and `.github/workflows/api-ecs-deploy.yml` both build and push the backend image to Amazon ECR.

Required GitHub repository configuration:

- repository variable: `AWS_REGION`
- repository variable: `ECR_REPOSITORY`
- repository secret: `AWS_ROLE_TO_ASSUME`

Image tagging policy:

- `${branch}`
- `${branch}-${short_sha}`
- `latest` on `main` only

For example, a push on `staging` publishes tags like:

- `staging`
- `staging-a1b2c3d`

## Next implementation targets

- add `/api/v1` controllers for sports, slates, events, and yesterday results
- add Postgres integration
- add Redis-backed caching where justified
- put admin refresh endpoints behind auth
