# UE NLP & Generative AI (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE focuses on **Generative AI** (prompting → fine-tuning → alignment), **Data Embeddings** (text/multimodal retrieval & clustering), and **Factorization & Recommendation** (low-rank models + modern retrieval/reranking). It is **directly aligned with a PhD path in NLP/GenAI (low-resource)**: you will build publishable pipelines (RAG, LoRA fine-tuning, evaluation with confidence bands) and production-grade artifacts (indexes, APIs, dashboards).

---

## Learning outcomes
By the end of this UE, you will be able to:
- Design **LLM systems**: prompting strategies → retrieval-augmented generation (RAG) → **PEFT** fine-tuning (LoRA/QLoRA) → **alignment** (DPO/RLHF).
- Build and evaluate **text & multimodal embeddings** (Sentence-Transformers/e5/SimCSE/CLIP-style), index them with **ANN** (FAISS/HNSW) and ship **search/retrieval**.
- Deliver **recommendation** solutions mixing **matrix factorization** (ALS/NMF/SVD) with **embedding retrieval + reranking**; evaluate with NDCG@K/MAP@K/Recall@K and diversity/coverage.
- Report like a researcher: **ablations**, statistical **CIs**, effect sizes, and **decision cards** (when to use what, under which data regime).

---

## Modules (ECUEs), core skills & **strategic resources**

### 1) Generative AI

**What you’ll master**
- **Prompting → Tool-use/RAG**: instruction & few-shot prompting, chain-of-thought (responsibly), function/tool calling, **RAG** (chunking, hyDE/query-rewrite, reranking).
- **Fine-tuning (PEFT)**: **LoRA/QLoRA** with 4-/8-bit adapters; SFT vs. **DPO**/**RLHF (TRL)**; safety/guardrails; evaluation & calibration (toxicity/factuality).
- **Serving & efficiency**: vLLM/TGI-style high-throughput serving; **bitsandbytes** quantization; prompt caching; latency/throughput trade-offs.

**Do-first labs (pro level)**
- **RAG baseline → strong RAG**: BM25 + dense retrieval (**FAISS**) → add **cross-encoder reranker**; measure **EM/F1** (QA), **nDCG@K**, hallucination rate (with **ragas/TruLens**).
- **QLoRA SFT** on a low-resource task (classification/QA/NER); compare **zero-shot vs. SFT vs. RAG+SFT**; report CIs over ≥5 seeds.
- **DPO vs. SFT**: align a small model with preference data; analyze helpfulness/harmlessness trade-offs; add **guardrails** to block unsafe tool calls.

**Strategic resources**
- Hugging Face **Transformers**: https://huggingface.co/docs/transformers/  
- **PEFT** (LoRA, QLoRA): https://huggingface.co/docs/peft/ · bitsandbytes: https://github.com/TimDettmers/bitsandbytes  
- **TRL** (SFT/DPO/RLHF): https://github.com/huggingface/trl  
- **LangChain** / **LlamaIndex** (RAG stacks): https://python.langchain.com/ · https://docs.llamaindex.ai/  
- **FAISS** (ANN): https://github.com/facebookresearch/faiss · **sentence-transformers** rerankers: https://www.sbert.net/  
- **ragas** (RAG eval): https://github.com/explodinggradients/ragas · **TruLens**: https://www.trulens.org/  
- **vLLM** (fast serving): https://vllm.ai/ · TGI: https://github.com/huggingface/text-generation-inference  
- Guardrails (structured outputs/policies): https://www.guardrails.ai/

---

### 2) Data Embedding (text & multimodal)

**What you’ll master**
- **Text embeddings**: **Sentence-Transformers**, **e5**, **GTE**, **SimCSE**; pooling (CLS/MEAN/weighted), domain adaptation (continued pretraining, **TSDAE**).
- **Multimodal**: **CLIP-style** dual encoders (text–image/audio); alignment losses (InfoNCE); hard-negative mining; large-scale retrieval.
- **Evaluation at scale**: **MTEB/BEIR** style benchmarks; Recall@K/MRR/nDCG; clustering coherence & trustworthiness (for semantic maps).

**Do-first labs (pro level)**
- **Domain-adapted embeddings**: continue-pretrain a base model on your corpus (**TSDAE**), then **contrastive fine-tune** with hard negatives; evaluate on retrieval (**MRR/nDCG**) and clustering (**Silhouette/NMI**).
- **Multimodal mini-CLIP**: curate 50k pairs; train dual encoders; zero-shot classification on a held-out dataset; latency test with **FAISS** HNSW/IVF-PQ.

**Strategic resources**
- Sentence-Transformers: https://www.sbert.net/ · **e5**: https://arxiv.org/abs/2402.05680 · **SimCSE**: https://arxiv.org/abs/2104.08821  
- BEIR benchmark: https://github.com/beir-cellar/beir · **MTEB**: https://github.com/embeddings-benchmark/mteb  
- TSDAE (domain adaptation): https://arxiv.org/abs/2104.06979  
- CLIP paper: https://arxiv.org/abs/2103.00020  
- **FAISS** ANN: https://github.com/facebookresearch/faiss · **hnswlib**: https://github.com/nmslib/hnswlib

---

### 3) Factorization & Recommendation

**What you’ll master**
- **Matrix factorization**: **ALS** (implicit feedback), **NMF**, **SVD/SVD++**; regularization, confidence weights, cold-start with side features.
- **Modern retrieval + reranking**: two-tower (bi-encoder) retrieval with embeddings; **cross-encoder rerankers**; **recsys-RAG** for item knowledge grounding.
- **Evaluation & fairness**: **NDCG@K/MAP@K/Recall@K**, coverage/diversity (Gini, ILD); popularity bias mitigation; offline vs. online trade-offs.

**Do-first labs (pro level)**
- **Implicit-ALS vs. LightFM** (hybrid) on **MovieLens-20M**: report NDCG@{10,20} with CIs across 5 splits; compute **coverage** and **intra-list diversity**.  
- **Two-tower + reranker**: build an embedding retrieval index (FAISS) + cross-encoder reranker (Sentence-Transformers); measure end-to-end latency/quality.

**Strategic resources**
- **implicit** (ALS): https://implicit.readthedocs.io/ · **LightFM**: https://making.lyst.com/lightfm/docs/ · **Surprise** (SVD): https://surpriselib.com/  
- **RecBole** (DL recsys): https://recbole.io/ · NVIDIA **Merlin** (ETL + ranking): https://developer.nvidia.com/nvidia-merlin  
- **MovieLens** datasets: https://grouplens.org/datasets/movielens/  
- Rerankers (cross-encoders): https://www.sbert.net/examples/applications/cross-encoder/README.html

---

## Contact hours & key dates (from your calendar)

**Approx. total (UE): ~56 hours (8 × 7h)**

- **Generative AI sessions:** **21/11/2024**, **26/11/2024**, **03/12/2024**, **10/12/2024** (AM+PM each → ≈ 28 h)  
  - **Big Data/GenAI Exams:** **17/12/2024** — Big Data (AM), **Generative AI (PM)**  
- **Factorization & Recommendation (extended slots):** **31/10**, **05/11**, **07/11**, **12/11**, **21/11** (each 16:30–18:30 → ≈ 10 h)  
- **Data Embedding** overlaps with DEL days (cross-listed): **22/10 (AM+PM)** and **19/11 (AM+PM)** — use for NLP embeddings labs (≈ 14 h)

> Sessions: **09:00–12:30** (AM) · **14:00–17:30** (PM). Extended slots shown explicitly.

---

## Assessment & grading (suggested — align with official rubric)
- **Project A — Generative AI (RAG + PEFT)**: high-quality QA/summarization system with RAG, QLoRA SFT, and factuality eval — **40%**  
- **Project B — Embedding Retrieval**: domain-adapted embeddings + ANN index + dashboard; MTEB-style evaluation — **30%**  
- **Project C — Recommendation**: MF baseline + embedding retrieval + reranker; NDCG/coverage/diversity — **20%**  
- **Participation & lab check-offs** — **10%**

---

## Practice path (to **outperform**)
1. **RAG first**: BM25 → dense → reranking; automate eval with **ragas**; add query rewrite & self-consistency; ship a tiny **FastAPI**.  
2. **PEFT next**: QLoRA for your domain; compare **zero-shot vs. RAG vs. RAG+SFT** across tasks; track **CIs** and latency.  
3. **Embeddings at scale**: TSDAE continue-pretrain → contrastive fine-tune; compress index (IVF-PQ); build a retrieval **dashboard**.  
4. **Recsys fusion**: MF baseline + two-tower retrieval + cross-encoder reranker; measure **NDCG@K** and **diversity/coverage**; propose an online plan.

---

## Lab starters & exemplar repos
- **Transformers** quickstarts: https://huggingface.co/docs/transformers/quicktour  
- **PEFT/QLoRA** examples: https://huggingface.co/docs/peft/task_guides/peft_lora  
- **TRL (DPO/RLHF)** examples: https://github.com/huggingface/trl/tree/main/examples  
- **Sentence-Transformers** examples: https://www.sbert.net/examples/  
- **FAISS** demos: https://github.com/facebookresearch/faiss/tree/main/demos  
- **RecBole** cookbook: https://recbole.io/docs/cookbook/overview.html  
- **Merlin** examples: https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples

---

## Reference stack — quick links
**LLM & Training**  
Transformers — https://huggingface.co/docs/transformers/ · Datasets — https://huggingface.co/docs/datasets/ · Evaluate — https://huggingface.co/docs/evaluate/  
PEFT — https://huggingface.co/docs/peft/ · TRL — https://github.com/huggingface/trl · bitsandbytes — https://github.com/TimDettmers/bitsandbytes  
vLLM — https://vllm.ai/ · TGI — https://github.com/huggingface/text-generation-inference

**Embeddings & Retrieval**  
Sentence-Transformers — https://www.sbert.net/ · e5 — https://arxiv.org/abs/2402.05680 · SimCSE — https://arxiv.org/abs/2104.08821  
FAISS — https://github.com/facebookresearch/faiss · hnswlib — https://github.com/nmslib/hnswlib · BEIR — https://github.com/beir-cellar/beir · MTEB — https://github.com/embeddings-benchmark/mteb

**Recommendation**  
implicit — https://implicit.readthedocs.io/ · LightFM — https://making.lyst.com/lightfm/docs/ · Surprise — https://surpriselib.com/  
RecBole — https://recbole.io/ · NVIDIA Merlin — https://developer.nvidia.com/nvidia-merlin · MovieLens — https://grouplens.org/datasets/movielens/

---

## Deliverables & submission checklist
- ✅ **RAG+PEFT** project with clear eval (ragas/TruLens), latency profile, and **decision card** (when RAG vs. SFT vs. both).  
- ✅ **Embedding retrieval** demo: ANN index, dashboard, and MTEB-style results with CIs.  
- ✅ **Recommendation** demo: MF + embedding retrieval + reranking; NDCG@K + diversity/coverage; ablations.  
- ✅ Reproducibility: configs, fixed seeds, env lockfile; CI for style/tests; clean README with **how-to-run** and dataset links.

---

## Toolkit prerequisites
- **Python 3.10+**, GPU recommended.  
- Libraries: transformers, datasets, peft, accelerate, trl, bitsandbytes; sentence-transformers, faiss/hnswlib; implicit, LightFM/Surprise; scikit-learn, NumPy/SciPy, pandas; FastAPI/Streamlit/Gradio for demos.

---

## Folder plan (this UE)
```text
05-NLP-GenerativeAI/
├─ README.md # this file
├─ Generative-AI/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Data-Embedding/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
└─ Applications-Factorization-Recommendation/
├─ notes.md
├─ resources.md
└─ notebooks/
```
