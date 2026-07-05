# API ECS Deploy

This document captures the currently working AWS deployment contract for the Spring Boot API.

## Live AWS configuration

- AWS Region: `us-east-1`
- ECS cluster: `default`
- ECS service: `programble-api-9045`
- ECS task definition family: `programble-api`
- ECR repository: `programble-api`
- Task definition source in repo: `.aws/api-task-definition.json`
- Deploy workflow in repo: `.github/workflows/api-ecs-deploy.yml`

The current Express service application URL is configured in GitHub as `API_SMOKE_TEST_URL` and is used by the deploy workflow for post-deploy health checks.

## Task definition expectations

The repo task definition should stay aligned with the live ECS service on the following fields:

- container name: `Main`
- container port: `8080`
- network mode: `awsvpc`
- launch type compatibility: `FARGATE`
- task CPU: `512`
- task memory: `1024`
- execution role ARN: `arn:aws:iam::440373532734:role/service-role/ecsTaskExecutionRole`
- log group: `/ecs/programble-api`
- environment variables:
  - `SERVER_PORT=8080`
  - `PROGRAMBLE_API_BASE_PATH=/api/v1`
  - `PROGRAMBLE_ENV=staging`
  - `SPRING_PROFILES_ACTIVE=staging`
  - `PROGRAMBLE_DB_URL=jdbc:postgresql://programble-staging-postgres.ce5gw80qs7nd.us-east-1.rds.amazonaws.com:5432/programble`
  - `PROGRAMBLE_JOBS_SCHEDULER_ENABLED=false`

The current task definition also needs these ECS secret injections:

- `PROGRAMBLE_DB_USERNAME`
- `PROGRAMBLE_DB_PASSWORD`
- `PROGRAMBLE_ADMIN_API_TOKEN`
- `PROGRAMBLE_ODDS_API_KEY`

Use real Secrets Manager references in ECS for those values. Do not commit placeholder ARNs that do not exist in AWS.

The deploy workflow renders the pushed ECR image into the task definition at runtime, so the checked-in JSON should keep a placeholder image value and should not be edited for each new tag or digest.

## GitHub deploy role expectations

The GitHub Actions workflow uses OIDC via the repository secret `AWS_ROLE_TO_ASSUME`. The role currently used for API deploys must be able to:

- assume via GitHub OIDC for the intended repository and branch
- authenticate to Amazon ECR
- push images to the `programble-api` repository
- register ECS task definition revisions
- update the `programble-api-9045` service in the `default` cluster
- pass the ECS task execution role `arn:aws:iam::440373532734:role/service-role/ecsTaskExecutionRole`

Based on the successful deploy logs, the currently assumed role name is `github-actions-ecr-push`.

In practical terms, the deploy role needs permissions covering at least:

- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`
- `ecr:DescribeRepositories`
- `ecs:RegisterTaskDefinition`
- `ecs:DescribeTaskDefinition`
- `ecs:UpdateService`
- `ecs:DescribeServices`
- `iam:PassRole` on `arn:aws:iam::440373532734:role/service-role/ecsTaskExecutionRole`

## Smoke test contract

After ECS reports service stability, the deploy workflow verifies:

- `GET ${API_SMOKE_TEST_URL}/actuator/health/readiness`
- `GET ${API_SMOKE_TEST_URL}/actuator/metrics/http.server.requests`
- `GET ${API_SMOKE_TEST_URL}/api/v1`

The readiness endpoint must report `{"status":"UP"}`, the metrics endpoint must expose `http.server.requests`, and the API root response must include `"service":"programble-api"`.

## Environment and profile notes

The checked-in staging task definition currently uses:

- `PROGRAMBLE_ENV=staging`
- `SPRING_PROFILES_ACTIVE=staging`

That is now the correct pairing for the current ECS service because the runtime depends on `application-staging.properties` for PostgreSQL and observability settings.

## Scheduler note

`PROGRAMBLE_JOBS_SCHEDULER_ENABLED` should stay `false` on the shared web-facing ECS service.

Reason:

- ECS canary deployments temporarily run old and new tasks side by side
- in-process scheduling on both revisions can double-run jobs

The current cheapest approach is:

- keep admin-triggered jobs available in the main API service
- if scheduled execution becomes necessary, run one dedicated single-replica scheduler task or service with `PROGRAMBLE_JOBS_SCHEDULER_ENABLED=true`
