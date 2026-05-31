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
- `event_participants`
- `sportsbooks`
- `markets`
- `offers`

The schema stays sport-agnostic by modeling event participants generically, so both team sports and individual sports can use the same `events` and `offers` model.

## Environment variables

- `SERVER_PORT`
- `PROGRAMBLE_ENV`
- `PROGRAMBLE_API_BASE_PATH`
- `CONSOLE_LOG_STRUCTURED_FORMAT`

## Next implementation targets

- add `/api/v1` controllers for sports, slates, events, and yesterday results
- add Postgres integration
- add Redis-backed caching where justified
- add admin refresh endpoints behind auth
