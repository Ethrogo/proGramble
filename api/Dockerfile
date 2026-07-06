FROM maven:3.9.9-eclipse-temurin-21-alpine AS builder

WORKDIR /workspace

COPY api/.mvn/ api/.mvn/
COPY api/mvnw api/pom.xml api/

WORKDIR /workspace/api

RUN chmod +x mvnw
RUN ./mvnw -B -DskipTests dependency:go-offline

WORKDIR /workspace

COPY api/src/ api/src/
COPY MLB/data/tracking/pitcher_k_shadow_predictions.csv MLB/data/tracking/pitcher_k_shadow_predictions.csv
COPY MLB/data/tracking/official_picks_history.csv MLB/data/tracking/official_picks_history.csv

WORKDIR /workspace/api

RUN ./mvnw -B -DskipTests package

FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

ENV SERVER_PORT=8080
ENV PROGRAMBLE_ENV=container
ENV SPRING_PROFILES_ACTIVE=container

COPY --from=builder /workspace/api/target/*.jar /app/app.jar

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD wget --quiet --tries=1 --spider "http://127.0.0.1:${SERVER_PORT}/actuator/health/readiness" || exit 1

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
