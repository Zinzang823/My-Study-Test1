package vn.mystudy.ai;

import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import java.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import tools.jackson.databind.ObjectMapper;

/**
 * Bật AI khi {@code app.ai.enabled=true} và có khóa của nhà cung cấp đã chọn: {@code AI_PROVIDER=anthropic} (mặc định,
 * dùng {@code ANTHROPIC_API_KEY}) hoặc {@code AI_PROVIDER=gemini} (dùng {@code GEMINI_API_KEY}). Khóa API chỉ nằm ở máy chủ (biến môi
 * trường), không bao giờ gửi xuống trình duyệt.
 */
@Configuration
class AiConfig {

    private static final Logger log = LoggerFactory.getLogger(AiConfig.class);

    AiGateway aiGateway(ObjectMapper json, boolean enabled, String provider, String anthropicKey, String anthropicModel,
            String geminiKey, String geminiModel, String geminiBaseUrl) {
        return aiGateway(json, enabled, provider, anthropicKey, anthropicModel, geminiKey, geminiModel, geminiBaseUrl,
                "", "llama-3.1-8b-instant", "https://api.groq.com/openai/v1");
    }

    @Bean
    @ConditionalOnMissingBean(AiGateway.class)
    AiGateway aiGateway(ObjectMapper json,
            @Value("${app.ai.enabled:false}") boolean enabled,
            @Value("${app.ai.provider:gemini}") String provider,
            @Value("${app.ai.api-key:}") String anthropicKey,
            @Value("${app.ai.model:claude-opus-5-5}") String anthropicModel,
            @Value("${app.ai.gemini-api-key:}") String geminiKey,
            @Value("${app.ai.gemini-model:gemini-2.5-flash-lite}") String geminiModel,
            @Value("${app.ai.gemini-base-url:https://generativelanguage.googleapis.com/v1beta}") String geminiBaseUrl,
            @Value("${app.ai.groq-api-key:}") String groqKey,
            @Value("${app.ai.groq-model:llama-3.1-8b-instant}") String groqModel,
            @Value("${app.ai.groq-base-url:https://api.groq.com/openai/v1}") String groqBaseUrl) {
        String p = provider != null ? provider.trim().toLowerCase() : "gemini";
        if ("groq".equals(p) || (groqKey != null && !groqKey.isBlank() && (geminiKey == null || geminiKey.isBlank()))) {
            if (!enabled || groqKey == null || groqKey.isBlank()) {
                log.info("AI features disabled (app.ai.enabled={}, provider=groq, api key present={})",
                        enabled, groqKey != null && !groqKey.isBlank());
                return new DisabledAiGateway();
            }
            log.info("AI features enabled with Groq model {}", groqModel);
            return new GroqAiGateway(json, groqBaseUrl, groqKey.trim(), groqModel);
        }
        boolean gemini = "gemini".equals(p);
        String key = gemini ? geminiKey : anthropicKey;
        if (!enabled || key == null || key.isBlank()) {
            log.info("AI features disabled (app.ai.enabled={}, provider={}, api key present={})", enabled,
                    gemini ? "gemini" : "anthropic", key != null && !key.isBlank());
            return new DisabledAiGateway();
        }
        if (gemini) {
            log.info("AI features enabled with Gemini model {}", geminiModel);
            return new GeminiAiGateway(json, geminiBaseUrl, key.trim(), geminiModel);
        }
        log.info("AI features enabled with model {}", anthropicModel);
        return new ClaudeAiGateway(AnthropicOkHttpClient.builder()
                .apiKey(key.trim())
                .timeout(Duration.ofSeconds(90))
                .maxRetries(2)
                .build(), anthropicModel);
    }
}
