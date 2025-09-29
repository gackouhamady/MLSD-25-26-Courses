
# EXAMS-PREP (Past Exams, Revision Sheets, Oral Preparation)

**Goal:** Turn scattered notes into **exam-ready mastery** and **confident orals**.  
This folder centralizes **past papers**, **high-yield revision sheets**, **problem-driven drills**, and **soutenances** prep (PPD & final defenses).

---

## Learning outcomes
By working through this folder, you will:
- Master **high-yield theory & methods** for each UE, with formula sheets and worked examples.
- Drill **exam-style problems** under time/constraints with **leakage-safe** reasoning and correct notation.
- Produce **executive-level slides/posters** and deliver a confident, structured **oral defense**.
- Track progress with a **checklist, spaced repetition**, and an **exam day protocol**.

---

## What to revise — per UE (high-yield map)

### 1) Data Engineering (Preprocessing • BI • Big Data • Packaging)
**High-yield theory**
- Data contracts & schemas (Avro/Protobuf), **partitioning**, **file formats** (Parquet), **table formats** (Delta/Iceberg/Hudi), **exactly-once** semantics.
- Streaming ETL: Kafka (+ Connect + Schema Registry), Spark Structured Streaming (triggers, watermarks), CDC (Debezium).
- **dbt** tests & lineage, **Airflow** DAG patterns, **Great Expectations/Soda** checks.
- Packaging & CI: `pyproject.toml`, tests, pre-commit, Docker multi-stage.

**Must-do drills**
- Design a streaming → lakehouse pipeline (bronze/silver/gold) with schema evolution + DQ checks; write the DAG outline & contracts.
- Diagnose a failing job from logs (checkpointing, offsets, backpressure).
- Write a minimal `pyproject.toml` + GitHub Actions matrix build.

**One-pager formulas / patterns**
- Watermarking rules, state TTL, window types; partitioning keys; dbt test taxonomy; CI quality gates.

---

### 2) Unsupervised Learning (Clustering • Mixtures/LBM • Dim. Reduction • Factorization/RecSys)
**High-yield theory**
- K-Means/K-Means++/MiniBatch; **GMM + EM** derivations; HDBSCAN/DBSCAN knobs; spectral clustering; **AIC/BIC/ICL**.
- PCA/KPCA, **t-SNE/UMAP** pitfalls; **trustworthiness/continuity**.
- NMF/SVD/ALS (implicit feedback), ranking metrics **NDCG/MAP/Recall@K**; coverage/diversity.

**Must-do drills**
- Derive EM updates for GMM; compute BIC; compare clusterings with **ARI/NMI**.
- UMAP vs t-SNE parameter card; compute **trustworthiness** on embeddings.
- Recsys: train ALS on a toy matrix; report NDCG@K and coverage.

**One-pager formulas**
- EM Q-function, silhouette/CH/DB, Laplacian eigenmaps, ranking metrics definitions.

---

### 3) Supervised • RL • Time Series
**Supervised (tabular)**
- Leakage-safe CV; imbalance (PR-AUC); calibration (Platt/Isotonic); SHAP & monotonic constraints.

**Reinforcement Learning**
- Bellman eqs; policy gradient; **PPO/SAC** mechanics; off-policy evaluation (IPS/DR); safety constraints (CPO idea).

**Time Series**
- ARIMA/SARIMA; ETS; **state-space/Kalman**; **rolling backtests**; probabilistic metrics (pinball loss); hierarchical **MinT**.

**Must-do drills**
- Calibrated classifier with threshold card (cost matrix).
- PPO clipping & GAE sanity checks; seed-averaged curves with CI.
- Rolling-origin CV; compute **MASE/RMSSE**; reconcile hierarchy (MinT).

---

### 4) Deep Learning & Graph Learning
**High-yield theory**
- Training recipes (One-Cycle, AMP, regularization, schedulers), transfer vs. linear probe, self-supervised (SimCLR/DINO/MAE).
- GNNs: **GCN/GAT/GraphSAGE**, oversmoothing fixes, hetero graphs (R-GCN), link prediction, **Graph Transformers**.

**Must-do drills**
- Build a robust training loop (Lightning/PyTorch) + seed control + confidence bands.
- PyG baseline on Cora/ogbn-arxiv; implement proper negative sampling for LP.

---

### 5) NLP & Generative AI
**High-yield theory**
- RAG pipeline (chunking, retrieval, reranking, grounding), **PEFT** (LoRA/QLoRA), **DPO** vs SFT, LLM eval pitfalls.
- Embeddings (ST/e5/SimCSE), ANN (FAISS/HNSW), **rerankers**.

**Must-do drills**
- RAG eval with EM/F1 + **ragas**-style metrics; latency vs. quality plot.
- QLoRA fine-tune; compare Zero-shot vs. RAG vs. RAG+SFT with CIs.

---

### 6) PPD, Soutenances & Final Defenses
**High-yield**
- **Executive story**: problem → method → evidence → decision → risk/impact.
- Anticipated questions matrix (stats validity, ethics, generalization, ablations).

**Must-do**
- 10-min talk, 5-min Q&A, backup slides; poster with constraints/risks panel; **demo fallback** (recorded run + logs).

---

## Exam calendar (from program)
- **Dim. Reduction / DEL:** **07/11/2024 PM**  
- **Supervised (L.L):** **28/11/2024 AM** · **Deep Learning (L.L):** **28/11/2024 PM**  
- **Big Data (S.M):** **17/12/2024 AM** · **Generative AI (S.M):** **17/12/2024 PM**  
- **Mixture Models (M.N):** **04/02/2025 AM** · **Packaging (S.M):** **04/02/2025 PM**  
- **Reinforcement Learning (B.B):** **10/02/2025 PM** · **Other Exam:** **12/02/2025 PM**  
- **Soutenances-PPD:** **26–27/03/2025** · **FA Defenses:** **22–25/06/2025** · **FA Jury:** **04/09/2025**  
- **FI Defenses:** **14–18/09/2025** · **FI Jury (Global end):** **22/09/2025**

> Sessions: **09:00–12:30** & **14:00–17:30** (Paris). Extended slots are indicated above.

---

## High-yield resources (quick access)

- **Sklearn user guide & examples:** https://scikit-learn.org/stable/auto_examples/  
- **Spark Structured Streaming:** https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html  
- **Kafka + Schema Registry:** https://kafka.apache.org/documentation/ · https://docs.confluent.io/platform/current/schema-registry/  
- **Delta / Iceberg / Hudi:** https://docs.delta.io/ · https://iceberg.apache.org/docs/latest/ · https://hudi.apache.org/docs/overview  
- **dbt Core:** https://docs.getdbt.com/  
- **Airflow best practices:** https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html  
- **Great Expectations:** https://docs.greatexpectations.io/docs/  
- **Clustering (sklearn):** https://scikit-learn.org/stable/modules/clustering.html · **Mixtures/EM:** https://scikit-learn.org/stable/modules/mixture.html  
- **UMAP / t-SNE:** https://umap-learn.readthedocs.io/ · https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html  
- **Implicit / LightFM / Surprise:** https://implicit.readthedocs.io/ · https://making.lyst.com/lightfm/docs/ · https://surpriselib.com/  
- **SHAP:** https://shap.readthedocs.io/ · **Optuna:** https://optuna.org/  
- **Gymnasium / SB3 / d3rlpy:** https://gymnasium.farama.org/ · https://stable-baselines3.readthedocs.io/ · https://d3rlpy.readthedocs.io/  
- **sktime / Nixtla (statsforecast, neuralforecast) / darts:** https://www.sktime.net/ · https://github.com/Nixtla · https://unit8co.github.io/darts/  
- **PyTorch / Lightning / timm:** https://pytorch.org/ · https://lightning.ai/ · https://github.com/huggingface/pytorch-image-models  
- **PyG / DGL / OGB:** https://pytorch-geometric.readthedocs.io/ · https://www.dgl.ai/ · https://ogb.stanford.edu/  
- **Transformers / PEFT / TRL / FAISS:** https://huggingface.co/docs/transformers/ · https://huggingface.co/docs/peft/ · https://github.com/huggingface/trl · https://github.com/facebookresearch/faiss  
- **Overleaf templates (ACM/IEEE):** https://www.overleaf.com/latex/templates

---

## Practice plans

### Theory → Problem sets (90–120 min blocks)
- **45 min**: read + derive (closed book) → **15 min**: formula recap → **45 min**: exam-style problems → **15 min**: error log.

### Coding sprints (120 min)
- **75 min**: implement (or fix) pipeline; **30 min**: tests/validation; **15 min**: write a mini report (plots + 5 bullets).

### Oral drills (30–45 min)
- **10-10-10**: 10-min talk → 10-min Q&A → 10-min debrief. Rotate topics across UEs.

---

## 21-Day countdown (example)
- **T-21 → T-15:** syllabus skim; collect **past papers**; draft all cheat sheets.  
- **T-14 → T-7:** daily problem sets; 1 coding sprint/day; 2 oral drills/week.  
- **T-6 → T-3:** full mock exams; tighten timing; finalize threshold/decision cards.  
- **T-2 → T-1:** light review; sleep; print formula sheets; pack exam kit.  
- **T-0 (Exam Day Protocol):** hydrate; read all questions first; allocate time; show work; **units & assumptions**; margin for checks.

---

## Deliverables & checklist (keep these in this folder)
- ✅ **Past-Exams/**: PDFs + your **worked solutions** (with timing & self-score).  
- ✅ **Revision-Sheets/**: one-pagers (formulas + traps + minimal examples) for each UE/topic.  
- ✅ **Oral-Soutenance/**: slide deck, poster, **anticipated Q&A**, demo fallback (recorded run + logs).  
- ✅ **Decision Cards**: *when to use what* (e.g., UMAP vs t-SNE; PPO vs SAC; ALS vs two-tower).  
- ✅ **Metric & Eval Bible**: PR-AUC vs ROC-AUC, pinball loss, MASE/RMSSE, NDCG/MAP definitions with tiny examples.  
- ✅ **Error Log**: mistakes & fixes (update after every mock).  
- ✅ **Exam Kit**: formula sheets, configs, seeds, runtime notes, citation snippets.

---

## Folder plan
```text
07-EXAMS-PREP/
├─ README.md # this file
├─ Past-Exams/
│ ├─ 2024-11-07-DimRed-DEL/
│ ├─ 2024-11-28-Supervised-AM/
│ ├─ 2024-11-28-DeepLearning-PM/
│ ├─ 2024-12-17-BigData-AM/
│ ├─ 2024-12-17-GenerativeAI-PM/
│ ├─ 2025-02-04-MixtureModels-AM/
│ ├─ 2025-02-04-Packaging-PM/
│ ├─ 2025-02-10-RL-PM/
│ └─ 2025-02-12-Other-PM/
├─ Revision-Sheets/
│ ├─ data-engineering.pdf
│ ├─ unsupervised.pdf
│ ├─ supervised-rl-ts.pdf
│ ├─ deep-graph.pdf
│ ├─ nlp-genai.pdf
│ └─ metrics-evaluation.pdf
├─ Oral-Soutenance/
│ ├─ slides/
│ ├─ poster/
│ ├─ demo-fallback/ # recordings, logs, metrics exports
│ └─ qa-matrix.md
├─ Decision-Cards/ # one-pagers to choose methods/params
├─ Error-Log.md
└─ Templates/
├─ cheat-sheet.tex # 2-page LaTeX compact sheet
├─ exam-report.md # problem → approach → result → check
├─ oral-outline.md # 10-min talk structure
└─ poster-outline.md
```
