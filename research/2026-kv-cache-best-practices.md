# KV-cache budgeting: 2026 evidence and decision rules

This document is the evidence base for `src/kvbudget/`. It answers one
question: **given a model, a GPU budget, a context length, and a concurrency
target, which KV-cache plan is actually deployable?**

The short answer is that KV memory is easy to estimate and hard to compress
safely. The compiler therefore separates three things that are often mixed
together:

1. **Capacity** — bytes per token, per sequence, per GPU, after tensor
   parallelism and paged allocation.
2. **Quality** — the measured or published accuracy cost of a KV format,
   eviction policy, or workload.
3. **Infrastructure** — whether the serving stack can actually reclaim the
   memory that a compression method promises.

Every number below has a source. Where the source is a 2026 blog or a single
benchmark, that is stated explicitly rather than presented as consensus.

---

## 1. Ten rules that survive the evidence

1. **Compute bytes before choosing a trick.** A standard decoder caches
   `2 × layers × KV_heads × head_dim × bytes` per token. For Llama-3.1-8B in
   BF16 that is exactly **131,072 B/token = 128 KiB/token**.
2. **FP8 halves the cache, not the whole serving cost.** The best published
   per-token saving is about **54% of BF16**, because scales and metadata are
   not free.
3. **MLA changes the formula, not just the constant.** DeepSeek-V3 caches
   `(kv_lora_rank + qk_rope_head_dim)` per layer: **70,272 B/token BF16**,
   or **68.62 KiB/token**.
4. **Tensor parallelism does not shard below one KV head.** Per-rank KV heads
   are `max(1, KV_heads // TP)`. MLA has one effective latent, so TP
   **replicates** it; decode context parallelism is needed to shard tokens.
5. **Paged allocation rounds every layer up to a block.** With block 16 and
   1,000 tokens, 1,008 token-slots are allocated: **0.79% waste**. The waste
   is usually small; the table metadata and per-layer rounding are not.
6. **Eviction is not a memory allocator.** If a fraction `f` of tokens is
   evicted from blocks of size `b`, only `f^b` of blocks become empty. At
   `f=0.9, b=16`, that is **18.5%**; at `f=0.99`, **85.1%**. NVIDIA measured a
   case that evicted 14,400 of 16,000 tokens and reclaimed almost nothing.
7. **FP8 KV needs a long-context accuracy test, not a perplexity test.** vLLM
   reported 128K needle-in-a-haystack dropping from **91% to 13%** on Hopper
   before a two-level FP32 accumulation fix, and **89%** after it.
8. **Selection-based eviction is not automatically better than random.**
   A 2026 study found random eviction competitive in **31 of 60** settings,
   with **32–43% more tokens/s** because it avoids scoring.
9. **Quality risk must be workload-specific.** Retrieval, summarization,
   reasoning, and code degrade differently. The compiler applies explicit
   workload penalties and keeps eviction off by default.
10. **Treat every compression claim as an unverified hypothesis until it is
    measured on your model, your context length, and your serving stack.**

---

## 2. Capacity formulas

### 2.1 Standard MHA / GQA / MQA

Per token, per layer, per GPU:

```
bytes = 2 × KV_heads_per_gpu × head_dim × bytes_per_element
KV_heads_per_gpu = max(1, KV_heads // TP)
```

For the full cache:

```
bytes = layers × 2 × KV_heads_per_gpu × head_dim
        × bytes_per_element × cached_tokens
```

The factor 2 is K and V. `head_dim` is the explicit config field when present;
otherwise `hidden_size // num_attention_heads`.

Verified anchors:

| Model | BF16 bytes/token | BF16 | FP8 |
|---|---:|---:|---:|
| Llama-3.1-8B | 131,072 | 128 KiB | 64 KiB |
| Llama-3.1-70B | 327,680 | 320 KiB | 160 KiB |
| Qwen2.5-7B | 57,344 | 56 KiB | 28 KiB |
| Qwen2.5-0.5B | 12,288 | 12 KiB | 6 KiB |
| OPT-125m | 36,864 | 36 KiB | 18 KiB |

A million decimal tokens of Llama-3.1-8B BF16 is **131.07 GB** (122.07 GiB);
FP8 is **65.54 GB** (61.04 GiB). These are decimal-GB conversions, not GiB.

### 2.2 Multi-head latent attention (MLA)

MLA caches one compressed latent plus the decoupled RoPE key per layer:

```
bytes = layers × (kv_lora_rank + qk_rope_head_dim) × bytes_per_element
```

DeepSeek-V3: `61 × (512 + 64) × 2 = 70,272 B/token` BF16, or **70.272 GB per
1M tokens**. DeepSeek-V2-Lite: `27 × 576 × 2 = 31,104 B/token`.

MLA is already compressed relative to full MHA. Do not apply a second
"MLA compression" factor without measuring it. Tensor parallelism replicates
the latent because there is only one effective KV head.

### 2.3 Hybrid attention

Gemma-2 alternates sliding-window and full-attention layers. The raw config
has `sliding_window=4096` but no `layer_types`; `AutoConfig` derives
`["sliding_attention", "full_attention", ...]` starting with sliding at layer
0. The compiler models each layer separately:

```
layer_tokens = min(context, sliding_window)  # sliding layer
layer_tokens = context                       # full layer
```

Gemma-2-9B at 8K: 21 sliding layers at 4,096 tokens plus 21 full layers at
8,192 tokens gives **258,048 B/token = 252 KiB/token** on average, not the
336 KiB/token a naive full-attention calculation would report.

Ignore `sliding_window` when it is greater than or equal to
`max_position_embeddings` (Qwen2.5-32B and Phi-3.5-mini both do this); those
models behave as full attention.

### 2.4 Paged allocation

vLLM's default block size is 16 tokens. Each layer allocates whole blocks:

```
allocated_tokens = ceil(cached_tokens / block_size) × block_size
```

At 1,000 tokens and block 16, the cache holds 1,008 slots, a **0.79%** waste.
At 128K tokens the block table alone has 8,192 entries per sequence. The
PagedAttention paper found naive allocators waste 60–80% of KV memory; paged
allocation removes most of that, but it does not remove per-layer rounding.

### 2.5 Tensor parallelism and replication

The vLLM tensor-parallel KV accounting issue gives the rule directly:

```
KV_heads_per_gpu = max(1, KV_heads // TP)
```

For Llama-3.1-70B with 8 KV heads:

| TP | KV heads/GPU | BF16 bytes/token/GPU |
|---:|---:|---:|
| 1 | 8 | 327,680 |
| 2 | 4 | 163,840 |
| 4 | 2 | 81,920 |
| 8 | 1 | 40,960 |
| 16 | 1 | 40,960 |

TP=16 does **not** halve the cache again. The ranks replicate the last KV
head. MLA behaves the same way at the latent level: TP replicates it, and
decode context parallel (DCP) is the mechanism that shards tokens.

A published Qwen-235B example (94 layers, 4 KV heads, TP=8, head_dim 128,
BF16, 131,072 context) comes to **6.308 GiB per GPU** and a concurrency of
**10.66**, close to vLLM's own **11.45**. That is the level of agreement the
compiler aims for: a planning estimate, not a substitute for a serving run.

---

## 3. Memory budget

The compiler uses vLLM's default GPU memory utilization and an explicit
reserve:

```
budget          = total_memory × gpu_memory_utilization      # 0.92
reserved        = 1 GiB + 0.03 × total_memory
available_kv    = budget - weights_per_gpu - reserved
```

The reserve covers activations, CUDA graphs, NCCL buffers, and non-torch
allocations. It is deliberately a single transparent number rather than a
fake per-component breakdown. A measured `reserved` value should always
replace the default when available.

Weights are exact released safetensors bytes from the model API. Converting
dtypes scales by bytes per parameter:

```
weights_deploy = weight_bytes × target_bytes_per_param / source_bytes_per_param
```

For a sharded model, `weights_per_gpu ≈ weights_deploy / TP`. This is an
approximation: embeddings, norms, and some MoE parameters may be replicated.
The compiler warns when TP exceeds the KV-head count or the layer count.

One important edge case: Mistral-7B repositories can contain both a
`consolidated.safetensors` file and `model-0000x-of-0000y` shards. Summing all
safetensors double-counts. The zoo fetcher prefers `model-\d+-of-\d+` shards
when they exist and only falls back to all safetensors otherwise.

---

## 4. Quality evidence

### 4.1 FP8 KV

vLLM's 2026 FP8 KV work is the strongest current evidence:

- FP8 e4m3 runs QK and ScoreV in FP8.
- On Hopper, FP8 FlashAttention-3 had a long-context accumulation problem:
  128K NIAH fell from **91% (BF16) to 13% (FP8)**.
- A two-level FP32 accumulation fix brought it to **89%**, about two points
  below BF16.
- Qwen3.5-27B lost at most **0.7 points**; Llama-3.3-70B measured **97–98%
  AUC**; Qwen3-30B-A3B measured about **94% AUC at 256K**.
- Caveats: hybrid-attention models with small sliding-window layers may be
  better off skipping those layers; `head_dim=256` can regress prefill;
  FlashMLA needs calibration; per-token cost can be **54% of BF16**, not 50%.

The compiler assigns FP8 a **0.05** risk and penalizes `head_dim >= 256`
further in speed. That is intentionally more conservative than the best
published result.

### 4.2 Integer KV

KIVI introduced 2-bit KV quantization with per-channel keys and per-token
values, reporting a **2.6× reduction in peak memory** and **2.35–3.47× larger
batch sizes** without fine-tuning. The practical lesson is that group-wise
quantization matters: naive per-tensor INT4 is much worse than a tuned
group-wise implementation.

The lab's own OPT-125m measurement (Section 6) shows the same ordering:
INT8 and INT4 preserve retrieval in this small model, while INT2 collapses.
Do not generalize that result to a 70B model or a 128K task.

### 4.3 Eviction and selection

The 2026 NVIDIA Efficient AI analysis is the best available reality check:

- It models paged blocks of about 16 tokens.
- Evicting 14,400 of 16,000 tokens leaves 1,600 survivors spread over roughly
  1,000 blocks. Almost every block still has at least one survivor, so a
  paged allocator reclaims almost nothing.
- R-KV's reported 90% memory saving was measured with pre-allocated
  contiguous tensors, not with vLLM's paged allocator. R-KV's appendix
  acknowledges that paged integration is nontrivial.
- Quest keeps the full cache and selects pages per query, so memory still
  grows with context.
- Accuracy at a 2,048-token budget (MATH500 uses 512):

| Method | AIME 2024 | AIME 2025 | MATH500 |
|---|---:|---:|---:|
| Full cache | 57.1 | 40.8 | 69.6 |
| SnapKV | 34.6 | 20.0 | 49.2 |
| R-KV | 25.4 | 17.5 | 46.4 |
| TriAttention | 42.1 | 32.9 | 56.0 |

TriAttention is the strongest of the compressed methods in that table. Its
authors report matching full attention on AIME25 at 32K generation, with
**2.5× throughput or 10.7× KV-memory reduction**. A separate 2026 study found
random eviction competitive with selection-based methods in **31 of 60**
settings and **32–43% faster** because it avoids scoring. The honest summary
is that eviction quality is workload-dependent and the memory reclamation is
infrastructure-dependent.

The compiler therefore:

- keeps eviction **off by default**;
- assumes a compacting cache when eviction is enabled, and says so in the
  plan notes;
- gives SnapKV and R-KV high risk (0.50 and 0.57), TriAttention 0.20, and
  random 0.15;
- applies additional retrieval, summarization, reasoning, and code penalties.

### 4.4 Effective context is not claimed context

RULER and LongBench v2 both show that models often degrade well before their
advertised context limit. LongBench v2 uses 503 multiple-choice questions over
8K–2M words; even the best model in the paper reaches only about **50.1%**.
A KV budget that fits 128K tokens is not evidence that the model uses 128K
tokens well. The compiler reports capacity, not capability.

---

## 5. What the compiler enumerates

For each model and GPU:

- KV formats: BF16, FP8, INT8, INT4, INT2, and optionally NVFP4.
- Eviction: none by default; TriAttention, random, SnapKV, R-KV when
  `--allow-eviction` is set.
- Block sizes: 16, 32, 64.
- Tensor-parallel degree.
- Context and concurrency targets.

Each plan reports:

```
KV/token, KV/sequence, max context, max concurrency,
quality risk, speed proxy, total GPU memory, waste fraction
```

Feasible plans must satisfy all of:

```
total_memory <= budget
context      <= model max context
max_context  >= requested context
max_concurrency >= requested concurrency
quality_risk <= max_quality_risk       # default 0.10
```

The Pareto frontier minimizes memory and risk while maximizing the speed
proxy. The four recommendations are safest, balanced, maximum context, and
maximum concurrency. When no plan meets the quality bar, the compiler says so
and shows the closest attempts instead of silently picking INT2.

The speed number is a **proxy**, not a benchmark:

```
speed = format_speed × eviction_speed × block_speed
```

It exists to break ties, not to promise throughput.

---

## 6. Measured validation

### 6.1 Exact cache bytes

`facebook/opt-125m` was loaded locally in BF16 and run at 128, 512, 1024, and
2048 tokens. The measured cache matched the formula exactly:

| Tokens | Formula | Measured |
|---:|---:|---:|
| 128 | 4,718,592 | 4,718,592 |
| 512 | 18,874,368 | 18,874,368 |
| 1024 | 37,748,736 | 37,748,736 |
| 2048 | 75,497,472 | 75,497,472 |

That is exactly **36,864 B/token** for 12 layers × 12 KV heads × 64 head_dim ×
2 (K and V) × 2 bytes. A synthetic BF16 K+V allocation for the Llama-3.1-8B
shape also matched exactly: 134,217,728 bytes at 1,024 tokens.

### 6.2 Quantization retrieval

Method: 512-token OPT-125m cache, full-precision query, cosine key retrieval,
quantize only K and V, 12 layers × 12 heads × 64 positions. Relative error is
the attention-output error against the BF16 reference.

| KV format | Top-1 retrieval | Relative output error |
|---|---:|---:|
| BF16 | 100.0% | 0.000 |
| FP8 e4m3 | 100.0% | 0.003 |
| INT8 | 100.0% | 0.001 |
| INT4 | 100.0% | 0.013 |
| INT2 | 56.7% | 0.221 |

This is a small-model sanity check, not a long-context quality claim. Its
value is that it independently confirms the ordering in the literature:
INT8/INT4 are usable in this setting, INT2 is not. The compiler's INT2 risk
was raised from 0.15 to **0.42** after this measurement.

### 6.3 Compiler scenarios

| Scenario | Result |
|---|---|
| Llama-3.1-8B, 24 GB, 32K, 1 sequence | Feasible; balanced plan INT4, 32 KiB/token, 1 GiB/sequence |
| Llama-3.1-8B, 24 GB, 128K, 1 sequence | No plan under 10% risk; INT2 fits memory but is above the quality bar |
| Llama-3.1-70B, 24 GB, 32K, TP=8 | Feasible; 40 KiB/token/GPU |
| DeepSeek-V3, 16×80 GB, 32K, TP=16 | Feasible; 70,272 B/token |
| DeepSeek-V2-Lite, 2×24 GB, 32K, TP=2 | Feasible; MLA latent replicated |

These are planning results. They do not include a serving-engine measurement
because vLLM is not installed in this environment.

---

## 7. Limitations

The compiler deliberately does **not** model:

- prefix caching or shared prefixes;
- beam search, speculative decoding, or multiple sequences per request;
- activation peaks during prefill;
- CUDA graph capture memory;
- NCCL and communication buffers beyond the flat reserve;
- KV offload to CPU, NVMe, or a disaggregated store;
- decode context parallelism;
- per-layer mixed precision;
- calibration quality for FP8/FP4;
- model quality at the requested context.

It also cannot validate a vendor's eviction claim. The NVIDIA analysis shows
why: the memory saving depends on whether the serving stack compacts blocks,
and most paged implementations do not.

The model zoo is a snapshot. Configs and checkpoints change; rerun
`scripts/fetch_model_zoo.py` and re-check the URLs before trusting a number.

---

## 8. Reproduce

```bash
python scripts/fetch_model_zoo.py
python scripts/kvbudget.py --list-models
python scripts/kvbudget.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-gb 24 --context 32768 --concurrency 1
python scripts/kvbudget_bench.py
pytest -q tests/test_kvbudget.py
```

`results/kvbudget.json` contains the exact checks, the OPT-125m cache table,
the quantization table, and the compiler scenarios.

---

## 9. Sources

Capacity and serving:

- PagedAttention: https://arxiv.org/html/2309.06180
- vLLM quantized KV cache docs:
  https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/
- vLLM FP8 KV cache (2026-04-22):
  https://vllm.ai/blog/2026-04-22-fp8-kvcache
- vLLM disaggregated serving guide (2026-09-29):
  https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide
- vLLM tensor-parallel KV accounting:
  https://github.com/vllm-project/vllm/issues/27017
- Sebastian Raschka, KV-cache calculations:
  https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/
- LMDeploy KV quantization:
  https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html

Compression and quality:

- KIVI: https://arxiv.org/abs/2402.02750
- SnapKV: https://arxiv.org/abs/2404.14469
- StreamingLLM: https://arxiv.org/abs/2309.17453
- H2O: https://arxiv.org/abs/2306.14048
- TriAttention: https://arxiv.org/html/2604.04921
- Random Attention: https://arxiv.org/abs/2609.03430
- KV-cache compression benchmark: https://arxiv.org/html/2607.05399v1
- NVIDIA Efficient AI, "KV cache compression and its infra problems"
  (2026-06-12):
  https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/

Long-context evaluation:

- RULER: https://arxiv.org/abs/2404.06654
- LongBench v2: https://arxiv.org/abs/2412.15204

Model configs:

- Llama-3.1: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen3: https://huggingface.co/Qwen/Qwen3-8B
- Gemma-2: https://huggingface.co/google/gemma-2-9b-it
- DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3
- DeepSeek-V2-Lite: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
