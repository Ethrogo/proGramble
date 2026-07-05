package com.programble.api.web;

import java.io.IOException;
import java.util.Optional;
import java.util.UUID;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.slf4j.spi.LoggingEventBuilder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
public class RequestLoggingFilter extends OncePerRequestFilter {

	private static final Logger LOGGER = LoggerFactory.getLogger(RequestLoggingFilter.class);
	private static final String REQUEST_ID_HEADER = "X-Request-Id";

	@Override
	protected boolean shouldNotFilter(HttpServletRequest request) {
		var path = request.getRequestURI();
		return path.startsWith("/actuator/health")
				|| path.startsWith("/actuator/info")
				|| path.startsWith("/actuator/metrics");
	}

	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {
		var requestId = Optional.ofNullable(request.getHeader(REQUEST_ID_HEADER))
				.filter(header -> !header.isBlank())
				.orElseGet(() -> UUID.randomUUID().toString());
		var previousRequestId = MDC.get("requestId");
		var startedAt = System.nanoTime();

		response.setHeader(REQUEST_ID_HEADER, requestId);
		MDC.put("requestId", requestId);

		try {
			filterChain.doFilter(request, response);
		} finally {
			var durationMs = (System.nanoTime() - startedAt) / 1_000_000L;
			logForStatus(response.getStatus())
					.addKeyValue("method", request.getMethod())
					.addKeyValue("path", request.getRequestURI())
					.addKeyValue("status", response.getStatus())
					.addKeyValue("durationMs", durationMs)
					.addKeyValue("clientIp", resolveClientIp(request))
					.log("HTTP request completed");

			if (previousRequestId == null) {
				MDC.remove("requestId");
			} else {
				MDC.put("requestId", previousRequestId);
			}
		}
	}

	private LoggingEventBuilder logForStatus(int status) {
		if (status >= 500) {
			return LOGGER.atError();
		}
		if (status >= 400) {
			return LOGGER.atWarn();
		}
		return LOGGER.atInfo();
	}

	private String resolveClientIp(HttpServletRequest request) {
		var forwardedFor = request.getHeader("X-Forwarded-For");
		if (forwardedFor == null || forwardedFor.isBlank()) {
			return request.getRemoteAddr();
		}

		return forwardedFor.split(",", 2)[0].trim();
	}
}
