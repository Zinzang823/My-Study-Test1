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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import vn.mystudy.common.api.ErrorCode;
import vn.mystudy.common.exception.BusinessException;

/**
 * Gọi Groq API (tương thích chuẩn OpenAI chat completions) với tốc độ siêu nhanh (chip LPU).
 * Tự động truy vấn danh sách model khả dụng từ tài khoản (/models) và tự động thích ứng định dạng JSON.
 */
class GroqAiGateway implements AiGateway {

    private static final Logger log = LoggerFactory.getLogger(GroqAiGateway.class);

    private final HttpClient http;
    private final ObjectMapper json;
    private final String baseUrl;
    private final String apiKey;
    private final String model;
    private volatile List<String> cachedModels;

    GroqAiGateway(ObjectMapper json, String baseUrl, String apiKey, String model) {
        this(HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build(), json, baseUrl, apiKey, model);
    }

    GroqAiGateway(HttpClient http, ObjectMapper json, String baseUrl, String apiKey, String model) {
        this.http = http;
        this.json = json;
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.model = (model == null || model.isBlank()) ? "llama-3.1-8b-instant" : model;
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
        List<String> candidates = new ArrayList<>();
        if (model != null && !model.isBlank()) {
            candidates.add(model);
        }
        for (String m : fetchAvailableModels()) {
            if (!candidates.contains(m)) {
                candidates.add(m);
            }
        }
        if (candidates.isEmpty()) {
            candidates = List.of("openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant", "gemma2-9b-it");
        }

        Map<String, Object> jsonSchema = new LinkedHashMap<>();
        jsonSchema.put("name", type.getSimpleName());
        jsonSchema.put("strict", true);
        jsonSchema.put("schema", JsonSchemas.of(type));

        String systemWithJson = systemPrompt + "\n\nQuan trọng: Bắt buộc chỉ trả lời bằng một đối tượng JSON (json object) duy nhất theo cấu trúc sau:\n"
                + json.writeValueAsString(JsonSchemas.of(type)) + "\nKhông được viết bất kỳ lời dẫn giải nào ngoài JSON.";

        HttpResponse<String> lastResponse = null;
        for (String candidateModel : candidates) {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("model", candidateModel);
            body.put("messages", List.of(
                    Map.of("role", "system", "content", systemWithJson),
                    Map.of("role", "user", "content", userPrompt)
            ));
            body.put("max_tokens", maxTokens);
            body.put("temperature", 0.3);

            // Bước 1: Thử với json_schema
            body.put("response_format", Map.of("type", "json_schema", "json_schema", jsonSchema));
            HttpResponse<String> res = executePost(body);
            int status = res.statusCode();

            // Bước 2: Nếu 400 (model không hỗ trợ strict json_schema), thử lại với json_object trên chính model này
            if (status == 400) {
                log.info("Groq model {} returned 400 with json_schema, retrying with json_object", candidateModel);
                body.put("response_format", Map.of("type", "json_object"));
                res = executePost(body);
                status = res.statusCode();
            }

            // Bước 3: Nếu vẫn 400, thử lại không dùng response_format (dựa vào system prompt)
            if (status == 400) {
                log.info("Groq model {} returned 400 with json_object, retrying without response_format", candidateModel);
                body.remove("response_format");
                res = executePost(body);
                status = res.statusCode();
            }

            if (status == 200) {
                log.info("Groq model {} succeeded (200 OK)", candidateModel);
                return parseResponse(res.body(), type, maxTokens);
            }

            lastResponse = res;
            log.warn("Groq model {} returned status {}: {}", candidateModel, status,
                    res.body() == null ? "" : res.body().substring(0, Math.min(300, res.body().length())));

            // Nếu 404 (model không có trên tài khoản), chuyển sang model tiếp theo
            if (status == 404) {
                continue;
            }
        }

        handleFailure(lastResponse);
        throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Không thể kết nối dịch vụ Groq AI.");
    }

    private HttpResponse<String> executePost(Map<String, Object> body) {
        try {
            String jsonPayload = json.writeValueAsString(body);
            HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/chat/completions"))
                    .timeout(Duration.ofSeconds(35))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))
                    .build();
            return http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException e) {
            log.error("Groq network error: {}", e.getMessage());
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Không kết nối được dịch vụ Groq AI (timeout). Vui lòng thử lại.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "Yêu cầu AI bị hủy.");
        }
    }

    private <T> T parseResponse(String responseBody, Class<T> type, long maxTokens) {
        JsonNode root = json.readTree(responseBody);
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
        // Xóa markdown code block nếu model bao bọc JSON trong ```json ... ```
        String cleaned = content.trim();
        if (cleaned.startsWith("```json")) {
            cleaned = cleaned.substring(7);
        } else if (cleaned.startsWith("```")) {
            cleaned = cleaned.substring(3);
        }
        if (cleaned.endsWith("```")) {
            cleaned = cleaned.substring(0, cleaned.length() - 3);
        }
        cleaned = cleaned.trim();
        try {
            return json.readValue(cleaned, type);
        } catch (RuntimeException e) {
            log.warn("Groq returned JSON not matching {}: content={}", type.getSimpleName(), cleaned);
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE, "AI trả lời sai định dạng. Vui lòng thử lại.");
        }
    }

    private void handleFailure(HttpResponse<String> response) {
        if (response == null) return;
        int status = response.statusCode();
        String detail = response.body() == null ? "" : response.body();
        log.error("Groq all models failed, last status={} body={}", status, detail.substring(0, Math.min(500, detail.length())));
        if (status == 429) {
            throw new BusinessException(ErrorCode.RATE_LIMITED,
                    "Đã vượt hạn mức gọi API của Groq. Vui lòng thử lại sau ít giây.");
        }
        if (status == 401 || status == 403) {
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                    "Khóa GROQ_API_KEY không hợp lệ hoặc đã hết hạn. Hãy kiểm tra biến môi trường GROQ_API_KEY.");
        }
        if (status == 404) {
            throw new BusinessException(ErrorCode.SERVICE_UNAVAILABLE,
                    "Không tìm thấy model phù hợp trên Groq cho khóa API này.");
        }
    }

    private synchronized List<String> fetchAvailableModels() {
        if (cachedModels != null && !cachedModels.isEmpty()) {
            return cachedModels;
        }
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/models"))
                    .timeout(Duration.ofSeconds(10))
                    .header("Authorization", "Bearer " + apiKey)
                    .GET()
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (res.statusCode() == 200) {
                JsonNode root = json.readTree(res.body());
                List<String> list = new ArrayList<>();
                for (JsonNode m : root.path("data")) {
                    String id = m.path("id").asString("");
                    if (!id.isEmpty() && !id.contains("whisper") && !id.contains("guard") && !id.contains("orpheus")) {
                        list.add(id);
                    }
                }
                log.info("Groq models available for this API key: {}", list);
                cachedModels = list;
                return list;
            } else {
                log.warn("Failed to fetch Groq models, status={}: {}", res.statusCode(), res.body());
            }
        } catch (Exception e) {
            log.warn("Failed to fetch Groq models: {}", e.getMessage());
        }
        return List.of();
    }
}
