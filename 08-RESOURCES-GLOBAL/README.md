# Global Resources (MLSD 25–26)

This folder is your **always-on toolkit**: math/stats/optimization foundations, production-grade Python stack, and curated books/papers/courses. Use it to **unblock quickly**, **go deeper fast**, and **ship with quality**.

---

## How to use this folder

- **When stuck** on theory → jump to **ML-Theory/** (cheat sheets, books, lectures).  
- **When building** → jump to **Python-ML-Stack/** (APIs, recipes, MLOps, deployment).  
- Keep a living **Reading-List.md** (books/papers/courses you finish) and **Useful-Links.md** (shortcuts you actually use).  
- Add one-page **decision cards** (what to use when) as you learn.

---

## Learning outcomes
By leveraging these resources, you will be able to:
- Reason from **first principles** (probability, linear algebra, optimization) and derive/justify algorithmic choices.
- Apply a **modern Python ML stack** (NumPy → sklearn → PyTorch/Transformers → MLOps/serving) with reproducibility.
- Evaluate and communicate results with **credible metrics, confidence intervals, and visual narratives**.
- Build a sustainable **personal knowledge base** (reading log, decision cards, code snippets).

---

## Modules & **strategic resources**

### A) ML-Theory/ — Math, Stats, Optimization
**Linear algebra & matrix calculus**
- *The Matrix Cookbook* — https://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274  
- *Linear Algebra Done Right* (Axler) — https://linear.axler.net/  
- Matrix calculus cheat sheet — https://arxiv.org/abs/1802.01528

**Probability & statistics**
- *All of Statistics* (Wasserman) — https://www.stat.cmu.edu/~larry/all-of-statistics/  
- *Introduction to Probability* (Blitzstein & Hwang) — https://projects.iq.harvard.edu/stat110/home  
- *Statistical Rethinking* (McElreath, Bayesian) — https://xcelab.net/rm/statistical-rethinking/

**Optimization**
- *Convex Optimization* (Boyd & Vandenberghe) — https://web.stanford.edu/~boyd/cvxbook/  
- *Numerical Optimization* (Nocedal & Wright) — https://www.springer.com/gp/book/9780387303031  
- *First-Order Methods in Optimization* (Bubeck) — https://arxiv.org/abs/1405.4980

**Statistical learning & ML**
- *The Elements of Statistical Learning* (Hastie/Tibshirani/Friedman) — https://hastie.su.domains/ElemStatLearn/  
- *Pattern Recognition and Machine Learning* (Bishop) — https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book  
- *Probabilistic Machine Learning* (Murphy, 2023) — https://probml.github.io/pml-book/

**Deep learning**
- *Deep Learning* (Goodfellow/Bengio/Courville) — https://www.deeplearningbook.org/  
- MIT 6.S191 notes — http://introtodeeplearning.com/  
- Stanford CS231n (vision) — http://cs231n.stanford.edu/ · CS224n (NLP) — http://web.stanford.edu/class/cs224n/

**Evaluation & statistics for ML**
- Bootstrap & CIs for ML — https://robjhyndman.com/hyndsight/intervals/  
- *An Introduction to Statistical Learning* (ISLR) — https://www.statlearning.com/

> **Pro move:** keep short **derivation notes** (LaTeX/Typst) for: PCA from SVD, logistic loss gradient, EM steps for GMM, bias–variance.

---

### B) Python-ML-Stack/ — From Notebook to Production

**Core numerics & data**
- NumPy — https://numpy.org/doc/stable/ · SciPy — https://docs.scipy.org/doc/scipy/  
- pandas — https://pandas.pydata.org/docs/ · Polars — https://docs.pola.rs/  
- PyArrow — https://arrow.apache.org/docs/python/ · DuckDB — https://duckdb.org/docs/api/python/overview

**Modeling**
- scikit-learn — https://scikit-learn.org/stable/user_guide.html  
- XGBoost — https://xgboost.readthedocs.io/ · LightGBM — https://lightgbm.readthedocs.io/ · CatBoost — https://catboost.ai/en/docs/

**Deep learning & embeddings**
- PyTorch — https://pytorch.org/docs/stable/ · Lightning — https://lightning.ai/docs/pytorch/stable/  
- Hugging Face Transformers — https://huggingface.co/docs/transformers/ · Datasets — https://huggingface.co/docs/datasets/  
- Sentence-Transformers — https://www.sbert.net/ · FAISS — https://github.com/facebookresearch/faiss

**Graphs & time series**
- PyTorch Geometric — https://pytorch-geometric.readthedocs.io/ · DGL — https://www.dgl.ai/  
- sktime — https://www.sktime.net/ · statsforecast — https://github.com/Nixtla/statsforecast · darts — https://unit8co.github.io/darts/

**Experiment tracking & versioning**
- MLflow — https://mlflow.org/ · Weights & Biases — https://docs.wandb.ai/  
- DVC — https://dvc.org/doc · lakeFS — https://docs.lakefs.io/

**Orchestration, data quality & catalogs**
- Airflow — https://airflow.apache.org/docs/ · Prefect — https://docs.prefect.io/  
- Great Expectations — https://docs.greatexpectations.io/docs/  
- DataHub — https://www.datahubproject.io/docs/

**Serving & deployment**
- FastAPI — https://fastapi.tiangolo.com/ · BentoML — https://docs.bentoml.org/  
- ONNX Runtime — https://onnxruntime.ai/ · TorchScript — https://pytorch.org/docs/stable/jit.html  
- NVIDIA Triton — https://github.com/triton-inference-server/server · vLLM/TGI — https://vllm.ai/ · https://github.com/huggingface/text-generation-inference

**Big data & distributed**
- Apache Spark — https://spark.apache.org/docs/latest/  
- Ray — https://docs.ray.io/en/latest/  
- Parquet — https://parquet.apache.org/docs/ · Delta Lake — https://docs.delta.io/ · Iceberg — https://iceberg.apache.org/docs/latest/ · Hudi — https://hudi.apache.org/docs/overview/

**Quality, testing & CI**
- pytest — https://docs.pytest.org/ · pre-commit — https://pre-commit.com/  
- black — https://black.readthedocs.io/ · ruff — https://docs.astral.sh/ruff/ · mypy — https://mypy.readthedocs.io/  
- GitHub Actions (Python) — https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python

> **Pro move:** create a **src/** package with `pyproject.toml`, tests, and CI early. Add a **Makefile** (or `tox/nox`) for one-command workflows.

---

### C) Reading-List.md — Books, Papers, Courses (curated starter set)

**Books (readable & high-impact)**
- ISLR (Intro, hands-on) — https://www.statlearning.com/  
- ESL (advanced classical ML) — https://hastie.su.domains/ElemStatLearn/  
- PRML (probabilistic view) — see Bishop above  
- PML (Murphy, 2023) — https://probml.github.io/pml-book/  
- Convex Optimization (Boyd) — see link above  
- Deep Learning (Goodfellow) — see link above

**Courses (free, evergreen)**
- Stanford CS231n (Vision) — http://cs231n.stanford.edu/  
- Stanford CS224n (NLP) — http://web.stanford.edu/class/cs224n/  
- CMU 10-701/36-705 (statistical ML) — https://www.cs.cmu.edu/~10701/  
- MIT 6.S191 (DL) — http://introtodeeplearning.com/  

**Survey & practice**
- *A Few Useful Things to Know about ML* (Domingos) — https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf  
- *Deep Learning Tuning Playbook* — https://github.com/google-research/tuning_playbook

> Keep a table: **date · resource · why it matters · key takeaways · action you tried**.

---

### D) Useful-Links.md — Battle-tested Shortcuts

**Visualization & communication**
- *Data Viz* cheats: Matplotlib — https://matplotlib.org/stable/tutorials/ · ggplot2 — https://ggplot2.tidyverse.org/  
- Better Poster — https://osf.io/ef53g/ · Presentation Zen principles — https://www.presentationzen.com/

**Reproducibility & docs**
- The Turing Way — https://the-turing-way.netlify.app/  
- Model Cards — https://modelcards.withgoogle.com/about · Data Cards — https://ai.googleblog.com/2022/07/data-cards-templates-for-transparent.html  
- choosealicense.com — https://choosealicense.com/ · Zenodo DOI — https://guides.github.com/activities/citable-code/

**Ethics & privacy**
- EU GDPR — https://gdpr.eu/  
- *Fairness and ML* resources — https://fairmlbook.org/

**Datasets & benchmarks**
- Hugging Face Datasets — https://huggingface.co/datasets  
- Kaggle — https://www.kaggle.com/datasets  
- OGB (graphs) — https://ogb.stanford.edu/ · MTEB/BEIR (IR/embeddings) — https://github.com/embeddings-benchmark/mteb · https://github.com/beir-cellar/beir  
- UCI — https://archive.ics.uci.edu/  

**Performance & hardware**
- PyTorch performance tips — https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html  
- CUDA best practices — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

---

## Practice path (turn these links into **skills**)
1. **Week 1:** ML-Theory refresher → PCA from SVD, logistic loss gradient; write 2 derivation notes.  
2. **Week 2:** Python stack → create `src/` package, CI, and MLflow tracking in a toy project.  
3. **Week 3:** Embeddings + FAISS → build a tiny retrieval demo (FastAPI) with latency & quality metrics.  
4. **Week 4:** Reporting → make a 1-page poster and a Model Card for your toy project; share feedback.

---

## Deliverables & checklist (for this folder)
- ✅ **ML-Theory/**: at least 5 personal cheat sheets (PCA, EM, convex duality, gradient tricks, bootstrapping).  
- ✅ **Python-ML-Stack/**: template project with `pyproject.toml`, tests, CI, MLflow, Dockerfile.  
- ✅ **Reading-List.md**: living log (completed + rating + takeaway + application).  
- ✅ **Useful-Links.md**: only links you actually use; prune monthly.

---

## Folder plan
``` text
08-RESOURCES-GLOBAL/
├─ README.md # this file
├─ ML-Theory/ # math, stats, optimization, SL/BL/DL notes
│ ├─ linear-algebra.md
│ ├─ probability-stats.md
│ ├─ optimization.md
│ └─ ml-foundations.md
├─ Python-ML-Stack/ # APIs, patterns, MLOps, serving, big data
│ ├─ dataframes.md
│ ├─ modeling.md
│ ├─ deep-learning.md
│ ├─ experiment-tracking.md
│ ├─ deployment-serving.md
│ └─ big-data-distributed.md
├─ Reading-List.md # curated books/papers/courses with personal notes
└─ Useful-Links.md # battle-tested shortcuts & snippets
```
