# API Observability

This document captures the runtime observability contract for the Spring Boot API and the minimum AWS alerting that should exist around the ECS deployment.

## Runtime surface

The API exposes the following operational endpoints:

- `GET /actuator/health`
- `GET /actuator/health/liveness`
- `GET /actuator/health/readiness`
- `GET /actuator/info`
- `GET /actuator/metrics`
- `GET /actuator/metrics/http.server.requests`

The ECS container health check and the deploy smoke test should both target `GET /actuator/health/readiness`.

## Request logging

Application requests emit a structured JSON log event named `HTTP request completed` with:

- `requestId`
- `method`
- `path`
- `status`
- `durationMs`
- `clientIp`

If the caller sends `X-Request-Id`, the API preserves it and echoes it in the response. Otherwise, the API generates one.

Actuator probe endpoints are intentionally excluded from request logging:

- `/actuator/health`
- `/actuator/info`
- `/actuator/metrics`

This keeps CloudWatch volume lower and avoids filling logs with ECS and smoke-test probe traffic.

## Metrics

The API publishes Spring Boot and Micrometer metrics, including HTTP server request metrics.

The current baseline is:

- `http.server.requests`
- JVM/process metrics
- datasource and connection pool metrics when available

The runtime also tags metrics with:

- `application=programble-api`
- `environment=<PROGRAMBLE_ENV>`

Latency histograms are enabled for `http.server.requests` with SLO buckets at:

- `100ms`
- `250ms`
- `500ms`
- `1s`
- `2s`
- `5s`

## AWS alerting baseline

At minimum, staging should have these alerts or equivalent monitors:

- ECS service running task count below desired count for 5 minutes
- ECS deployment failures or rollbacks
- Application Load Balancer target `5XX` responses greater than `0` over 5 minutes
- Application Load Balancer unhealthy target count greater than `0`
- High request latency on the load balancer target group
- RDS CPU, free storage, and connection exhaustion if the database remains online

For uptime verification, keep an external or workflow-driven check on:

- `GET /actuator/health/readiness`
- `GET /api/v1`

## Cost notes

CloudWatch Logs costs will rise quickly if probe traffic is logged. That is why request logging excludes actuator endpoints by default.

If staging costs continue to trend high, reduce retention on `/ecs/programble-api` and keep metrics/alerts focused on the smallest useful set.
