# 0001. Choose website stack and architectural boundaries

Date: 2026-05-29

## Status

Accepted

## Context

ProGramble needs a website-oriented architecture that can grow beyond the current MLB workflow codebase. The initial MVP needs to support:

- home page
- global navigation
- sport landing pages
- game slate page
- player prop page
- simple SEO pages
- basic auth placeholder
- admin/data refresh placeholder
- yesterday results for `pitcher_k` and `pitcher_bb` sourced from `data/tracking`

The architecture needs to support future expansion across multiple sports, including MLB, NBA, NFL, ATP, WTA, PGA, and LPGA. It also needs clean boundaries between website delivery, backend APIs, background jobs, and the existing modeling/tracking workflows.

## Decision

Use the following stack and boundaries for the website platform:

- frontend: `Next.js` with `TypeScript`
- backend API: `Java 21` with `Spring Boot`
- primary database: `PostgreSQL`
- cache: `Redis`
- auth: public site with admin-only authenticated routes in the first phase
- hosting: prefer `AWS` as the primary hosting platform
- background jobs: scheduled or manually triggered Spring Boot jobs
- architecture style: modular monolith, not microservices

## Frontend framework

Choose `Next.js` with `TypeScript`.

Reasons:

- good fit for SEO-oriented public pages
- supports static generation, server rendering, and incremental updates
- routing model is well suited to sports, slates, events, and player pages
- TypeScript keeps the API integration contract explicit

Frontend responsibilities:

- public page rendering
- navigation and page composition
- SEO metadata and route structure
- authenticated admin UI

Frontend non-responsibilities:

- direct access to raw odds providers
- direct access to filesystem tracking artifacts
- business logic for normalization, grading, or modeling

## Backend API

Choose `Java 21` with `Spring Boot`.

Reasons:

- good fit for a long-lived backend with multiple integrations
- strong support for APIs, scheduling, auth, configuration, logging, and operational visibility
- reasonable long-term fit for multi-sport growth

Backend responsibilities:

- public APIs for sports, events, offers, and results
- admin APIs for refresh operations and diagnostics
- integration with database, cache, and tracking inputs
- orchestration of background jobs

Backend non-responsibilities:

- frontend rendering concerns
- direct training or experimentation workflows for models

## Database

Choose `PostgreSQL` as the system of record for website-serving data.

Primary stored data:

- sports and leagues/tours
- teams
- players
- events
- sportsbooks
- markets
- offers
- curated results
- admin/job status

The website should not depend on raw CSV files or raw third-party payloads as its primary serving layer.

## Cache

Choose `Redis` as a performance and coordination layer.

Intended uses:

- short-lived live slate caching
- hot API response caching
- rate-limit protection helpers
- lightweight job coordination or deduplication

Redis is not the source of truth.

## Auth approach

Phase 1 auth scope:

- public site remains open
- admin and operational routes require authentication

Implementation direction:

- frontend session-based login flow or equivalent
- backend validates session or token for protected endpoints

This keeps auth small in the MVP while preserving a clear path to user accounts later.

## Hosting choice

Preferred hosting choice: `AWS`.

Recommended AWS topology:

- frontend: `AWS Amplify Hosting`
- backend API: `Amazon ECS on Fargate`
- database: `Amazon RDS for PostgreSQL`
- cache: `Amazon ElastiCache for Redis`
- auth for admin routes: `Amazon Cognito`
- logs and operational visibility: `Amazon CloudWatch`

Reasons:

- better long-term fit for a multi-service sports platform
- natural fit for Spring Boot, scheduled jobs, managed database, managed cache, and operational monitoring
- stronger alignment with future production requirements than a simpler all-in-one host
- clean path to staging and production environments with controlled IAM, networking, and managed infrastructure

Operational tradeoff:

- AWS adds more setup and infrastructure overhead than Render for an MVP
- that overhead is acceptable because the preferred direction is long-term platform control on AWS

Fallback simpler alternative:

- `Render` remains an acceptable fallback if delivery speed becomes more important than AWS alignment

`AWS App Runner` is not the preferred backend anchor for this decision.

## Background jobs

Use Spring Boot scheduled or manually triggered jobs for the MVP.

Job responsibilities:

- refresh sports and odds data
- materialize website-facing read models
- ingest and normalize yesterday results from `data/tracking`
- publish admin-facing status and diagnostics

The website should consume normalized job outputs, not raw job internals.

## Domain model for multiple sports

Use a shared domain model with sport-specific extensions.

Shared core entities:

- `Sport`
- `LeagueOrTour`
- `Team`
- `Player`
- `Event`
- `Market`
- `Offer`
- `Result`
- `Projection`
- `TrackingArtifact`

Sport-specific extensions:

- MLB: probable starters, pitcher props
- NBA: lineup status, player points/rebounds/assists
- NFL: weekly game and player prop structures
- Tennis: match-level entities without team dependency
- Golf: tournament and round structures

This preserves one common website platform while allowing sport-specific behavior where needed.

## Boundary rules

### Frontend boundary

The frontend only consumes internal ProGramble APIs.

It does not:

- read `data/tracking` directly
- call The Odds API directly
- embed modeling or grading logic

### Backend boundary

The backend owns:

- API contracts
- provider integration logic
- normalization and serving logic
- adapters for tracking-backed results

### Modeling boundary

The existing modeling workflows remain separate from the website platform.

The website consumes outputs such as:

- normalized offers
- normalized projections
- normalized yesterday results

It does not own model training or experimentation workflows.

### Admin boundary

Admin actions must be mediated through authenticated backend endpoints.

The browser must not directly:

- access the filesystem
- mutate tracking artifacts
- trigger shell-like operations

## Initial API surface

Expected MVP endpoints:

- `GET /api/v1/sports`
- `GET /api/v1/sports/{sport}/slates?date=YYYY-MM-DD`
- `GET /api/v1/events/{eventId}`
- `GET /api/v1/events/{eventId}/offers`
- `GET /api/v1/results/yesterday?market=pitcher_k`
- `GET /api/v1/results/yesterday?market=pitcher_bb`
- `GET /api/v1/admin/status`
- `POST /api/v1/admin/refresh/{domain}`

## Consequences

Positive:

- clean split between public website, backend API, and modeling workflows
- good support for public SEO pages and future authenticated admin areas
- strong backend base for multi-sport growth
- avoids premature microservice complexity

Negative:

- two-application architecture adds some setup overhead compared to a single runtime
- Java/Spring Boot is a heavier backend stack than a smaller Node-only backend
- a future migration may still be needed if the website and modeling platform converge more tightly than expected

## Follow-up decisions

The following decisions remain open and should become separate MADRs:

- define AWS environment topology for development, staging, and production
- define API auth/session strategy
- define database schema for core multi-sport entities
- define how tracking artifacts are ingested into website-facing read models
- define frontend deployment and preview workflow
