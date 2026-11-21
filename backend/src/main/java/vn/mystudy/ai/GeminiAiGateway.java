package vn.mystudy.ai;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import vn.mystudy.common.api.ErrorCode;
import vn.mystudy.common.exception.BusinessException;

/**
 * Gọi Gemini qua REST API {@code models/{model}:generateContent}. Đầu ra JSON được ràng buộc bằng
 * {@code responseJsonSchema} sinh từ record Java (cùng record với nhà cung cấp Claude), nên các tính năng AI không cần
 * biết đang dùng nhà cung cấp nào. Khóa API gửi ở header {@code x-goog-api-key}, không đặt trong URL để không lọt vào
 * log truy cập.
 */
class GeminiAiGateway implements AiGateway {

    private static final Logger log = LoggerFactory.getLogger(GeminiAiGateway.class);
    private static final Set<String> BLOCKED = Set.of("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
            "RECITATION", "IMAGE_SAFETY");

    private final HttpClient http;
    private final ObjectMapper json;
    private final String baseUrl;
    private final String apiKey;
    private final String model;

    GeminiAiGateway(ObjectMapper json, String baseUrl, String apiKey, String model) {
        this(HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build(), json, baseUrl, apiKey, model);
    }

    GeminiAiGateway(HttpClient http, ObjectMapper json, String baseUrl, String apiKey, String model) {
        this.http = http;
        this.json = json;
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.model = model.startsWith("models/") ? model.substring(7) : model;
    }

    @Override
    public boolean enabled() {
        return true;
    }

    @Override
    public String model() {
        return model;
    }

    @Override
    public <T> T generate(String systemPrompt, String userPrompt, Class<T> type, long maxTokens) {
        Map<String, Object> genConfig = new LinkedHashMap<>();
        genConfig.put("responseMimeType", "application/json");
        genConfig.put("responseJsonSchema", JsonSchemas.of(type));
        genConfig.put("maxOutputTokens", maxTokens);
        if (model.contains("gemini-3")) {
            // Giảm độ trễ suy nghĩ xuống mức LOW để phản hồi nhanh nhất (2-3 giây)
            genConfig.put("thinkingConfig", Map.of("thinkingLevel", "LOW"));
        }

        Map<String, Object> body = Map.of(
                "systemInstruction", Map.of("parts", List.of(Map.of("text", systemPrompt))),
                "contents", List.of(Map.of("role", "user", "parts", List.of(Map.of("text", userPrompt)))),
                "generationConfig", genConfig);
        String jsonPayload = json.writeValueAsString(body);
        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/models/" + model + ":generateContent"))
                .timeout(Duration.ofSeconds(35))
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = send(request, jsonPayload);
        JsonNode root = json.readTree(response.body());

        String blockReason = root.path("promptFeedback").path("blockReason").asString("");
        if (!blockReason.isEmpty()) {
            log.warn("Gemini blocked prompt: {}", blockReason);
            throw refused();
        }
        JsonNode candidate = root.path("candidates").path(0);
        String finishReason = candidate.path("finishReason").asString("");
        if ("MAX_TOKENS".equals(finishReason)) {
            log.warn("Gemini response truncated (maxOutputTokens={}) for {}", maxTokens, type.getSimpleName());
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI trả lời bị ngắt quãng. Vui lòng thử lại.");
        }
        if (BLOCKED.contains(finishReason)) {
            log.warn("Gemini stopped with finishReason={}", finishReason);
            throw refused();
        }
        StringBuilder text = new StringBuilder();
        for (JsonNode part : candidate.path("content").path("parts")) {
            if (part.path("thought").asBoolean(false)) continue; // bỏ phần "suy nghĩ" nếu mô hình trả về
            text.append(part.path("text").asString(""));
        }
        if (text.isEmpty()) {
            log.warn("Gemini returned no text (finishReason={})", finishReason);
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI chưa trả lời. Vui lòng thử lại.");
        }
        try {
            return json.readValue(text.toString(), type);
        } catch (RuntimeException e) {
            log.warn("Gemini returned JSON not matching {}: {}", type.getSimpleName(), e.getMessage());
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI trả lời sai định dạng. Vui lòng thử lại.");
        }
    }

    /** Gửi yêu cầu; tự động chuyển sang mô hình/endpoint tương thích nếu 404 (sai tên) hoặc 503 (quá tải). */
    private HttpResponse<String> send(HttpRequest request, String jsonPayload) {
        HttpResponse<String> response = executeWithRetry(request);
        int status = response.statusCode();
        if (status == 200) return response;

        if (status == 404 || status == 503) {
            String uriStr = request.uri().toString();
            List<String> fallbacks = new ArrayList<>();
            if (uriStr.contains("gemini-3")) {
                fallbacks.add("gemini-2.5-flash-lite");
                fallbacks.add("gemini-2.5-flash");
            } else if (uriStr.contains("gemini-2.5-flash-lite")) {
                fallbacks.add("gemini-2.5-flash");
                fallbacks.add("gemini-3.8-flash");
            } else {
                fallbacks.add("gemini-2.5-flash-lite");
                fallbacks.add("gemini-2.5-flash");
            }

            for (String fallbackModel : fallbacks) {
                String fallbackUri = uriStr.replaceAll("/models/[^:]+:", "/models/" + fallbackModel + ":");
                if (uriStr.contains("/v1/models/")) {
                    fallbackUri = fallbackUri.replace("/v1/models/", "/v1beta/models/");
                }
                if (fallbackUri.equals(uriStr)) continue;

                log.info("Gemini status {} on {}, trying fallback model {} ({})", status, uriStr, fallbackModel, fallbackUri);

                String fallbackPayload = jsonPayload;
                if (!fallbackModel.contains("gemini-3")) {
                    fallbackPayload = fallbackPayload.replaceAll(",\"thinkingConfig\":\\{[^}]*\\}", "")
                            .replaceAll("\"thinkingConfig\":\\{[^}]*\\},?", "");
                }

                HttpRequest fallbackReq = HttpRequest.newBuilder(URI.create(fallbackUri))
                        .timeout(Duration.ofSeconds(35))
                        .header("Content-Type", "application/json")
                        .header("x-goog-api-key", apiKey)
                        .POST(HttpRequest.BodyPublishers.ofString(fallbackPayload, StandardCharsets.UTF_8))
                        .build();

                HttpResponse<String> fallbackRes = executeWithRetry(fallbackReq);
                if (fallbackRes.statusCode() == 200) {
                    log.info("Gemini fallback to {} succeeded (200 OK)", fallbackModel);
                    return fallbackRes;
                }
                response = fallbackRes;
                status = response.statusCode();
            }
        }

        if (status == 400 && jsonPayload.contains("thinkingConfig")) {
            log.info("Gemini 400 with thinkingConfig, retrying without thinkingConfig");
            String strippedPayload = jsonPayload.replaceAll(",\"thinkingConfig\":\\{[^}]*\\}", "")
                    .replaceAll("\"thinkingConfig\":\\{[^}]*\\},?", "");
            HttpRequest retryReq = HttpRequest.newBuilder(request.uri())
                    .timeout(Duration.ofSeconds(35))
                    .header("Content-Type", "application/json")
                    .header("x-goog-api-key", apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(strippedPayload, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> retryRes = executeWithRetry(retryReq);
            if (retryRes.statusCode() == 200) {
                return retryRes;
            }
            response = retryRes;
            status = response.statusCode();
        }

        String detail = response.body() == null ? "" : response.body();
        log.error("Gemini API error status={} body={}", status,
                detail.substring(0, Math.min(500, detail.length())));
        if (status == 429) {
            throw new BusinessException(ErrorCode.RATE_LIMITED,
                    "Đã hết hạn mức gọi AI của khóa Gemini. Vui lòng thử lại sau ít phút.");
        }
        if (status == 503) {
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                    "Máy chủ Google Gemini đang bị quá tải tạm thời (503 High Demand). Vui lòng thử lại sau 1-2 phút hoặc chuyển sang model gemini-2.5-flash-lite.");
        }
        if (status == 400 || status == 401 || status == 403 || status == 404) {
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                    "Không gọi được Gemini (status " + status + "): khóa API hoặc tên mô hình (" + model + ") không hợp lệ. Hãy kiểm tra biến GEMINI_API_KEY hoặc GEMINI_MODEL.");
        }
        throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                "Dịch vụ AI tạm thời gián đoạn. Vui lòng thử lại.");
    }

    private HttpResponse<String> executeWithRetry(HttpRequest request) {
        for (int attempt = 1; ; attempt++) {
            try {
                HttpResponse<String> response = http.send(request,
                        HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                int status = response.statusCode();
                if (status == 200) return response;
                if ((status == 503 || status == 500) && attempt == 1) {
                    sleep(2000);
                    continue;
                }
                return response;
            } catch (IOException e) {
                if (attempt == 1) {
                    sleep(1000);
                    continue;
                }
                log.error("Gemini network error", e);
                throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Không kết nối được dịch vụ AI (timeout). Vui lòng thử lại.");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Yêu cầu AI bị hủy.");
            }
        }
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static BusinessException refused() {
        return new BusinessException(ErrorCode.BAD_REQUEST,
                "AI không thể trả lời yêu cầu này. Hãy xem lời giải có sẵn hoặc hỏi giáo viên.");
    }
}
