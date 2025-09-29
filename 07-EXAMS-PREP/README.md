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
- Data contracts & schemas: **Avro** (spec: https://avro.apache.org/docs/1.11.1/specification/), **Protobuf** (lang guide: https://developers.google.com/protocol-buffers/docs/overview)  
- Storage & tables: **Parquet** (docs: https://parquet.apache.org/docs/), **Delta Lake** (https://docs.delta.io/), **Apache Iceberg** (https://iceberg.apache.org/docs/latest/), **Apache Hudi** (https://hudi.apache.org/docs/overview/)  
- Streaming ETL: **Apache Kafka** (https://kafka.apache.org/documentation/), **Kafka Connect** (https://kafka.apache.org/documentation/#connect), **Schema Registry** (https://docs.confluent.io/platform/current/schema-registry/), **Debezium** CDC (https://debezium.io/documentation/)  
- Stream processing: **Spark Structured Streaming** (https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)  
- Transformation & lineage: **dbt Core** (https://docs.getdbt.com/), **Airflow** best practices (https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html), **Great Expectations** (https://docs.greatexpectations.io/docs/), **Soda Core** (https://docs.soda.io/soda/core/)  
- Packaging & CI: Python packaging guide (https://packaging.python.org/), `pyproject.toml` PEPs (**PEP 517**: https://peps.python.org/pep-0517/, **PEP 518**: https://peps.python.org/pep-0518/), **Docker** multi-stage (https://docs.docker.com/build/building/multi-stage/), **GitHub Actions (Python)** (https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python)

**Must-do drills**  
- Lakehouse pipeline (bronze/silver/gold) with schema evolution + DQ checks → read: Delta `OPTIMIZE`/Z-Order (https://docs.delta.io/latest/optimizations-oss.html) and Iceberg compaction (https://iceberg.apache.org/docs/latest/maintenance/)  
- Debugging streaming: Spark checkpoints/backpressure (https://spark.apache.org/docs/latest/streaming-programming-guide.html#monitoring), Kafka consumer lag (https://docs.confluent.io/platform/current/kafka/post-deployment.html#monitor-consumer-lag)  
- Minimal packaging & CI: `build` (https://pypi.org/project/build/), `twine` (https://twine.readthedocs.io/), **pre-commit** (https://pre-commit.com/)

**One-pager formulas / patterns**  
- Watermarks & windows (Spark guide: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#handling-late-data-and-watermarking), dbt test taxonomy (https://docs.getdbt.com/docs/build/tests), CI quality gates with **black/ruff/mypy** (black: https://black.readthedocs.io/ · ruff: https://docs.astral.sh/ruff/ · mypy: https://mypy.readthedocs.io/)

---

### 2) Unsupervised Learning (Clustering • Mixtures/LBM • Dim. Reduction • Factorization/RecSys)
**High-yield theory**  
- Clustering: sklearn overview (https://scikit-learn.org/stable/modules/clustering.html); **K-Means++/MiniBatch**; **DBSCAN/HDBSCAN** (HDBSCAN: https://hdbscan.readthedocs.io/); **Spectral** (tutorial: Luxburg 2007 https://www.cs.upc.edu/~csplanas/teaching/pdf/Luxburg2007_tutorial_spectral_clustering.pdf)  
- Mixtures/EM: sklearn Mixture (https://scikit-learn.org/stable/modules/mixture.html), **AIC/BIC/ICL** (concepts: ISLR ch.6 https://www.statlearning.com/)  
- Dimensionality reduction: **PCA/KPCA** (https://scikit-learn.org/stable/modules/decomposition.html#pca), **t-SNE** pitfalls (Distill: https://distill.pub/2016/misread-tsne/), **UMAP** (https://umap-learn.readthedocs.io/)  
- Recommendation: **implicit ALS** (https://implicit.readthedocs.io/), **LightFM** (https://making.lyst.com/lightfm/docs/), **Surprise** (https://surpriselib.com/); ranking metrics **NDCG/MAP/Recall@K** (RecBole metrics guide: https://recbole.io/docs/user_guide/evaluation/metrics.html)

**Must-do drills**  
- Derive EM for GMM + compute BIC → sklearn example (https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html)  
- Trustworthiness/continuity for embeddings → sklearn metric (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html)  
- ALS on toy data → implicit quickstart (https://implicit.readthedocs.io/en/latest/quickstart.html)

**One-pager formulas**  
- Silhouette/CH/DB (https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation), Laplacian eigenmaps notes (https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering), ranking metric defs (RecSys survey: https://dl.acm.org/doi/10.1145/3285029)

---

### 3) Supervised • RL • Time Series
**Supervised (tabular)**  
- Leakage-safe CV (sklearn model selection: https://scikit-learn.org/stable/modules/cross_validation.html), imbalance (**imbalanced-learn**: https://imbalanced-learn.org/stable/), calibration (https://scikit-learn.org/stable/modules/calibration.html), **SHAP** (https://shap.readthedocs.io/)  
- Monotonic constraints: **LightGBM** (https://lightgbm.readthedocs.io/en/stable/Parameters.html#monotone-constraints), **XGBoost** (https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)

**Reinforcement Learning**  
- Concepts: **Spinning Up** (https://spinningup.openai.com/), **Stable-Baselines3** (https://stable-baselines3.readthedocs.io/)  
- PPO/SAC mechanics: PPO paper summary (https://spinningup.openai.com/en/latest/algorithms/ppo.html), SAC (https://spinningup.openai.com/en/latest/algorithms/sac.html)  
- Off-policy evaluation: IPS/DR tutorial (slides: https://hunch.net/~jmc/online_learning/OLSurvey.pdf), Safety RL (CPO paper: https://proceedings.mlr.press/v70/achiam17a.html)

**Time Series**  
- Classical: **statsmodels** tsa (https://www.statsmodels.org/stable/tsa.html), **Prophet** (https://facebook.github.io/prophet/)  
- Libraries: **sktime** (https://www.sktime.net/en/stable/), **statsforecast** (https://github.com/Nixtla/statsforecast), **neuralforecast** (https://github.com/Nixtla/neuralforecast), **darts** (https://unit8co.github.io/darts/)  
- Hierarchical reconciliation (**MinT**): Nixtla HierarchicalForecast (https://github.com/Nixtla/hierarchicalforecast)  
- Backtesting & metrics: Rolling-origin CV (sktime: https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.model_selection.ExpandingWindowSplitter.html), **MASE/RMSSE** (https://robjhyndman.com/hyndsight/mase/), pinball loss (https://scikit-learn.org/stable/modules/model_evaluation.html#quantile-loss)

**Must-do drills**  
- Calibrated classifier + threshold card (sklearn **CalibrationDisplay**: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html)  
- PPO with GAE sanity checks (SB3 PPO doc: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)  
- Rolling-origin CV + **MASE/RMSSE** implementation (sktime/darts examples above)

---

### 4) Deep Learning & Graph Learning
**High-yield theory**  
- Training recipes: **PyTorch** AMP (https://pytorch.org/docs/stable/amp.html), **One-Cycle LR** (https://arxiv.org/abs/1803.09820), **Lightning** loops (https://lightning.ai/docs/pytorch/stable/)  
- Self-supervised: **SimCLR** (https://arxiv.org/abs/2002.05709), **DINO** (https://arxiv.org/abs/2104.14294), **MAE** (https://arxiv.org/abs/2111.06377)  
- GNNs: **PyTorch Geometric** (https://pytorch-geometric.readthedocs.io/), **DGL** (https://www.dgl.ai/), OGB tasks (https://ogb.stanford.edu/), Graph Transformers (Graphormer: https://arxiv.org/abs/2106.05234)

**Must-do drills**  
- Robust training loop (Lightning + seed control): Lightning templates (https://lightning.ai/templates)  
- PyG baseline (Cora/ogbn-arxiv): PyG examples (https://github.com/pyg-team/pytorch_geometric/tree/master/examples)  
- Link prediction with proper negative sampling: DGL/PyG LP tutorials (DGL LP example: https://docs.dgl.ai/en/latest/guide_5_graph.html#link-prediction)

---

### 5) NLP & Generative AI
**High-yield theory**  
- RAG: **LangChain** (https://python.langchain.com/), **LlamaIndex** (https://docs.llamaindex.ai/); **FAISS** ANN (https://github.com/facebookresearch/faiss)  
- PEFT: **LoRA/QLoRA** (PEFT: https://huggingface.co/docs/peft/ · bitsandbytes: https://github.com/TimDettmers/bitsandbytes), **TRL** (DPO/RLHF: https://github.com/huggingface/trl)  
- Embeddings & reranking: **Sentence-Transformers** (https://www.sbert.net/), **e5** (https://arxiv.org/abs/2402.05680), cross-encoder rerankers (https://www.sbert.net/examples/applications/cross-encoder/README.html)  
- LLM evaluation & safety: **ragas** (https://github.com/explodinggradients/ragas), **TruLens** (https://www.trulens.org/), Guardrails (https://www.guardrails.ai/)

**Must-do drills**  
- RAG eval with EM/F1 + ragas; latency vs. quality plot (ragas quickstart above)  
- QLoRA fine-tune vs. zero-shot vs. RAG+SFT (PEFT/TRL guides above)

---

### 6) PPD, Soutenances & Final Defenses
**High-yield**  
- Executive story & slidecraft: **Presentation Zen** principles (https://www.presentationzen.com/), **Duarte** storytelling (https://www.duarte.com/)  
- Poster: **Better Poster** (https://osf.io/ef53g/), Beamer (https://ctan.org/pkg/beamer), reveal.js (https://revealjs.com/)  
- Viva/defense prep: **Vitae viva tips** (https://www.vitae.ac.uk/doing-research/ending-your-doctorate/defending-your-thesis-viva)

**Must-do**  
- 10-min talk + 5-min Q&A + backup slides; demo fallback (OBS for recording: https://obsproject.com/); Overleaf templates: **ACM** (https://www.acm.org/publications/proceedings-template) · **IEEE** (https://www.ieee.org/conferences/publishing/templates.html)

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

- Sklearn examples — https://scikit-learn.org/stable/auto_examples/  
- Spark Structured Streaming — https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html  
- Kafka + Schema Registry — https://kafka.apache.org/documentation/ · https://docs.confluent.io/platform/current/schema-registry/  
- Delta / Iceberg / Hudi — https://docs.delta.io/ · https://iceberg.apache.org/docs/latest/ · https://hudi.apache.org/docs/overview  
- dbt Core — https://docs.getdbt.com/ · Airflow best practices — https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html  
- Great Expectations — https://docs.greatexpectations.io/docs/  
- Clustering — https://scikit-learn.org/stable/modules/clustering.html · Mixtures/EM — https://scikit-learn.org/stable/modules/mixture.html  
- UMAP — https://umap-learn.readthedocs.io/ · t-SNE — https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html  
- Implicit / LightFM / Surprise — https://implicit.readthedocs.io/ · https://making.lyst.com/lightfm/docs/ · https://surpriselib.com/  
- SHAP — https://shap.readthedocs.io/ · Optuna — https://optuna.org/  
- Gymnasium / SB3 / d3rlpy — https://gymnasium.farama.org/ · https://stable-baselines3.readthedocs.io/ · https://d3rlpy.readthedocs.io/  
- sktime / statsforecast / darts — https://www.sktime.net/ · https://github.com/Nixtla/statsforecast · https://unit8co.github.io/darts/  
- PyTorch / Lightning / timm — https://pytorch.org/ · https://lightning.ai/ · https://github.com/huggingface/pytorch-image-models  
- PyG / DGL / OGB — https://pytorch-geometric.readthedocs.io/ · https://www.dgl.ai/ · https://ogb.stanford.edu/  
- Transformers / PEFT / TRL / FAISS — https://huggingface.co/docs/transformers/ · https://huggingface.co/docs/peft/ · https://github.com/huggingface/trl · https://github.com/facebookresearch/faiss  
- Overleaf templates (ACM/IEEE) — https://www.overleaf.com/latex/templates

---

## Practice plans

### Theory → Problem sets (90–120 min blocks)
- **45 min** read + derive (closed book) → **15 min** formula recap → **45 min** exam-style problems → **15 min** error log.  
  - Derivation helpers: Matrix Cookbook (https://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274), Matrix calculus cheat sheet (https://arxiv.org/abs/1802.01528)

### Coding sprints (120 min)
- **75 min** implement/fix pipeline → **30 min** tests/validation → **15 min** mini report (plots + 5 bullets).  
  - Testing refs: pytest (https://docs.pytest.org/), pre-commit (https://pre-commit.com/)

### Oral drills (30–45 min)
- **10-10-10**: 10-min talk → 10-min Q&A → 10-min debrief.  
  - Speaking guides: Toastmasters tips (https://www.toastmasters.org/resources/public-speaking-tips), Duarte (https://www.duarte.com/)

---

## 21-Day countdown (example)
- **T-21 → T-15:** syllabus skim; collect **past papers**; draft cheat sheets (LaTeX template: https://www.overleaf.com/latex/templates/cheat-sheet/mhctvxrzdtxy)  
- **T-14 → T-7:** daily problem sets; 1 coding sprint/day; 2 oral drills/week  
- **T-6 → T-3:** full mock exams; finalize **decision cards** (example: sklearn model selection guide https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)  
- **T-2 → T-1:** light review; sleep; print formula sheets; pack exam kit (checklist template: https://www.atlassian.com/software/confluence/templates/checklist)  
- **T-0 (Exam Day Protocol):** pacing tips (https://www.scotthyoung.com/blog/2018/03/27/exam-strategy/), units & assumptions prominently

---

## Deliverables & checklist (keep these in this folder)
- ✅ **Past-Exams/**: PDFs + your **worked solutions** (with timing & self-score).  
- ✅ **Revision-Sheets/**: one-pagers (formulas + traps + minimal examples) for each UE/topic.  
- ✅ **Oral-Soutenance/**: slide deck, poster, **anticipated Q&A**, demo fallback (recorded run + logs).  
  - Poster: Better Poster (https://osf.io/ef53g/) · Slides: reveal.js (https://revealjs.com/) / Beamer (https://ctan.org/pkg/beamer)  
- ✅ **Decision Cards**: *when to use what* (e.g., UMAP vs t-SNE; PPO vs SAC; ALS vs two-tower).  
  - Example inspiration: Google ML “Rules of ML” (https://developers.google.com/machine-learning/guides/rules-of-ml)  
- ✅ **Metric & Eval Bible**: PR-AUC vs ROC-AUC (https://dl.acm.org/doi/10.1145/1143844.1143874), pinball loss (https://scikit-learn.org/stable/modules/model_evaluation.html#quantile-loss), MASE/RMSSE (https://robjhyndman.com/hyndsight/mase/), NDCG/MAP (https://recbole.io/docs/user_guide/evaluation/metrics.html)  
- ✅ **Error Log**: mistakes & fixes (keep concise).  
- ✅ **Exam Kit**: formula sheets, configs, seeds, runtime notes, citation snippets.  
  - Citation quick help: ZoteroBib (https://zbib.org/)

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
