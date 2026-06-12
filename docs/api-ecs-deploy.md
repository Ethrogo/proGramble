# API ECS Deploy

This document captures the currently working AWS deployment contract for the Spring Boot API.

## Live AWS configuration

- AWS Region: `us-east-1`
- ECS cluster: `default`
- ECS service: `programble-api-9045`
- ECS task definition family: `default-programble-api-9045`
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
- task CPU: `1024`
- task memory: `2048`
- execution role ARN: `arn:aws:iam::440373532734:role/service-role/ecsTaskExecutionRole`
- log group: `/aws/ecs/default/programble-api-9045-e07d`
- environment variables:
  - `SERVER_PORT=8080`
  - `PROGRAMBLE_API_BASE_PATH=/api/v1`
  - `PROGRAMBLE_ENV=main`

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

- `GET ${API_SMOKE_TEST_URL}/actuator/health`
- `GET ${API_SMOKE_TEST_URL}/api/v1`

The health endpoint must report `{"status":"UP"}` and the API root response must include `"service":"programble-api"`.

## Environment and profile notes

`PROGRAMBLE_ENV=main` is currently safe to keep. Today it drives application metadata and structured logs through `programble.environment`; it does not select a Spring profile on its own.

The container image default is `SPRING_PROFILES_ACTIVE=container`. There is currently no `application-container.properties`, so the active `container` profile does not introduce separate runtime overrides. That is why the service can run successfully even though `PROGRAMBLE_ENV=main`.

The only issue with the current setup is naming drift:

- structured logs and `/api/v1` metadata report environment `main`
- Spring reports active profile `container`

That is not a functional problem today, but it can become confusing once profile-specific config is added. When the service moves beyond the current Express setup, either:

- keep `PROGRAMBLE_ENV=main` and explicitly set `SPRING_PROFILES_ACTIVE=main`, or
- keep `SPRING_PROFILES_ACTIVE=container` and treat `container` as the canonical runtime profile

Until then, leaving `PROGRAMBLE_ENV=main` as-is is acceptable.
