# AI Model Benchmark: Blog Pipeline Showdown

We ran the same 4-step autonomous blog pipeline (research, write, edit, design) across 10 different AI models to find the best balance of speed, quality, and cost.

## The Pipeline

| Step | Agent | Task |
|------|-------|------|
| 1 | QUILL | Research trending topic, write 800-1500 word draft |
| 2 | EDITOR | Review draft, improve clarity/SEO/engagement |
| 3 | CANVAS | Generate cover image via AI image gen |
| 4 | QUILL | Create approval task on the board |

## Results

| # | Model | Status | Time | Tokens | Steps | Quality | Est. Cost |
|---|-------|--------|------|--------|-------|---------|-----------|
| 1 | GPT-5.4 (premium) | pass | 106s | 36,032 | 4/4 | 50/100 | $0.0721 |
| 2 | DeepSeek Chat | pass | 182s | 72,928 | 4/4 | 60/100 | $0.0102 |
| 3 | Gemini 2.5 Flash | pass | 61s | 32,967 | 4/4 | 60/100 | $0.0049 |
| 4 | Gemini 2.5 Pro | pass | 288s | 35,917 | 4/4 | 60/100 | $0.0449 |
| 5 | Llama 4 Scout | pass | 30s | 15,451 | 2/4 | 50/100 | $0.0012 |
| 6 | Mistral Small 3.1 | pass | 30s | 15,486 | 4/4 | 60/100 | $0.0005 |
| 7 | Qwen3 235B MoE | fail | 333s | 26,709 | 2/4 | 40/100 | $0.0019 |
| 8 | Qwen3.5 Flash | pass | 106s | 21,623 | 2/4 | 50/100 | $0.0015 |
| 9 | GPT-4.1 Mini | pass | 45s | 29,113 | 4/4 | 60/100 | $0.0116 |
| 10 | Gemini 2.0 Flash Lite | pass | 75s | 24,983 | 4/4 | 40/100 | $0.0017 |

## Winners

- **Fastest**: Llama 4 Scout (30s)
- **Cheapest**: Mistral Small 3.1 ($0.0005)
- **Best Quality**: DeepSeek Chat (60/100)
- **Most Token-Efficient**: Llama 4 Scout (15,451 tokens)

## Value Score (Quality / Cost)

1. **Mistral Small 3.1** — 120000 quality/$ (Q=60, $0.0005)
2. **Llama 4 Scout** — 41667 quality/$ (Q=50, $0.0012)
3. **Qwen3.5 Flash** — 33333 quality/$ (Q=50, $0.0015)
4. **Gemini 2.0 Flash Lite** — 23529 quality/$ (Q=40, $0.0017)
5. **Gemini 2.5 Flash** — 12245 quality/$ (Q=60, $0.0049)
6. **DeepSeek Chat** — 5882 quality/$ (Q=60, $0.0102)
7. **GPT-4.1 Mini** — 5172 quality/$ (Q=60, $0.0116)
8. **Gemini 2.5 Pro** — 1336 quality/$ (Q=60, $0.0449)
9. **GPT-5.4 (premium)** — 693 quality/$ (Q=50, $0.0721)

---

*Benchmark run by Automatos AI Platform. All models accessed via OpenRouter. Pipeline: daily-blog-pipeline (4 sequential steps).*