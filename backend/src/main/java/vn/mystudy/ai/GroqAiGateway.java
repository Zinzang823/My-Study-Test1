package vn.mystudy.ai;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import vn.mystudy.common.api.ErrorCode;
import vn.mystudy.common.exception.BusinessException;

/**
 * Gọi Groq API (tương thích chuẩn OpenAI chat completions) với tốc độ siêu nhanh (chip LPU).
 * Hỗ trợ response_format structured JSON schema và json_object.
 */
class GroqAiGateway implements AiGateway {

    private static final Logger log = LoggerFactory.getLogger(GroqAiGateway.class);

    private final HttpClient http;
    private final ObjectMapper json;
    private final String baseUrl;
    private final String apiKey;
    private final String model;

    GroqAiGateway(ObjectMapper json, String baseUrl, String apiKey, String model) {
        this(HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build(), json, baseUrl, apiKey, model);
    }

    GroqAiGateway(HttpClient http, ObjectMapper json, String baseUrl, String apiKey, String model) {
        this.http = http;
        this.json = json;
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.model = (model == null || model.isBlank()) ? "llama-3.3-70b-versatile" : model;
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
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("messages", List.of(
                Map.of("role", "system", "content", systemPrompt + "\n\nQuan trọng: Bắt buộc chỉ trả về định dạng JSON hợp lệ theo schema yêu cầu."),
                Map.of("role", "user", "content", userPrompt)
        ));
        body.put("max_tokens", maxTokens);
        body.put("temperature", 0.3);

        Map<String, Object> jsonSchema = new LinkedHashMap<>();
        jsonSchema.put("name", type.getSimpleName());
        jsonSchema.put("strict", true);
        jsonSchema.put("schema", JsonSchemas.of(type));
        body.put("response_format", Map.of("type", "json_schema", "json_schema", jsonSchema));

        String jsonPayload = json.writeValueAsString(body);
        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/chat/completions"))
                .timeout(Duration.ofSeconds(35))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = send(request, body);
        JsonNode root = json.readTree(response.body());
        JsonNode choice = root.path("choices").path(0);
        String finishReason = choice.path("finish_reason").asString("");
        if ("length".equalsIgnoreCase(finishReason)) {
            log.warn("Groq response truncated (max_tokens={}) for {}", maxTokens, type.getSimpleName());
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI trả lời bị ngắt quãng do quá dài. Vui lòng thử lại.");
        }
        String content = choice.path("message").path("content").asString("");
        if (content.isBlank()) {
            log.warn("Groq returned empty content (finishReason={})", finishReason);
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI chưa trả lời. Vui lòng thử lại.");
        }
        try {
            return json.readValue(content, type);
        } catch (RuntimeException e) {
            log.warn("Groq returned JSON not matching {}: {}", type.getSimpleName(), e.getMessage());
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI trả lời sai định dạng. Vui lòng thử lại.");
        }
    }

    private HttpResponse<String> send(HttpRequest request, Map<String, Object> bodyMap) {
        try {
            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            int status = response.statusCode();
            if (status == 200) return response;

            // Nếu model chưa hỗ trợ json_schema thì tự động chuyển sang json_object
            if (status == 400 && bodyMap.containsKey("response_format")) {
                log.info("Groq 400 with json_schema, retrying with json_object");
                Map<String, Object> retryBody = new LinkedHashMap<>(bodyMap);
                retryBody.put("response_format", Map.of("type", "json_object"));
                String retryPayload = json.writeValueAsString(retryBody);
                HttpRequest retryReq = HttpRequest.newBuilder(request.uri())
                        .timeout(Duration.ofSeconds(35))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + apiKey)
                        .POST(HttpRequest.BodyPublishers.ofString(retryPayload, StandardCharsets.UTF_8))
                        .build();
                HttpResponse<String> retryRes = http.send(retryReq, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                if (retryRes.statusCode() == 200) return retryRes;
                response = retryRes;
                status = response.statusCode();
            }

            String detail = response.body() == null ? "" : response.body();
            log.error("Groq API error status={} body={}", status, detail.substring(0, Math.min(500, detail.length())));
            if (status == 429) {
                throw new BusinessException(ErrorCode.RATE_LIMITED,
                        "Đã vượt hạn mức gọi API của Groq. Vui lòng thử lại sau ít giây.");
            }
            if (status == 401 || status == 403) {
                throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                        "Khóa GROQ_API_KEY không hợp lệ hoặc đã hết hạn. Hãy kiểm tra biến môi trường GROQ_API_KEY.");
            }
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                    "Không gọi được dịch vụ Groq AI (status " + status + "). Vui lòng thử lại.");
        } catch (IOException e) {
            log.error("Groq network timeout", e);
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Không kết nối được dịch vụ Groq (timeout). Vui lòng thử lại.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Yêu cầu AI bị hủy.");
        }
    }
}
