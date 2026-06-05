# ProGramble API

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
- `GET /actuator/health`
- `GET /actuator/info`

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
- `CONSOLE_LOG_STRUCTURED_FORMAT`

## Staging runtime contract

The `staging` profile is defined in `src/main/resources/application-staging.properties`.

Expected staging runtime settings:

- `SPRING_PROFILES_ACTIVE=staging`
- `SERVER_PORT`
- `PROGRAMBLE_ENV=staging`
- `PROGRAMBLE_API_BASE_PATH=/api/v1`
- `PROGRAMBLE_DB_URL`
- `PROGRAMBLE_DB_USERNAME`
- `PROGRAMBLE_DB_PASSWORD`

Health and readiness surface:

- `GET /actuator/health`
- `GET /actuator/info`

The container and future ECS service should treat `/actuator/health` as the default health endpoint.

## ECR publishing contract

The GitHub Actions workflow `.github/workflows/api-ecr.yml` builds and pushes the backend image to Amazon ECR.

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
- add admin refresh endpoints behind auth
