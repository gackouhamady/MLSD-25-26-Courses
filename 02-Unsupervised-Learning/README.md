# UE Unsupervised Learning (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE develops mastery of **representation learning and structure discovery** without labels: clustering (partitioning, density, hierarchical, spectral), **mixture models** (EM/GMM, latent block models), **dimensionality reduction** (linear & manifold), and **matrix factorization & recommendation**. You will design **production-grade unsupervised pipelines** with rigorous validation, stability analysis, and clear business-facing deliverables.

---

## Learning outcomes
By the end of this UE, you will be able to:
- Build **end-to-end unsupervised pipelines**: preprocessing → representation → clustering/factorization → evaluation → reporting.
- Choose and tune methods: **K-Means/MBKMeans, GMM (EM), HDBSCAN/DBSCAN, Agglomerative, Spectral**, **PCA/KPCA/UMAP/t-SNE/Isomap**, **NMF/SVD/ALS**, **Latent Block Models (co-clustering)**.
- Evaluate with **internal metrics** (Silhouette, Calinski–Harabasz, Davies–Bouldin), **stability** (bootstrapping/consensus), and **external metrics** when labels exist (NMI/ARI).
- Apply **matrix factorization** to recommendations (implicit feedback, cold-start), measure **Recall@K / MAP@K / NDCG@K**, and argue trade-offs (accuracy vs. diversity/coverage).
- Communicate insights with **diagnostics & visuals** and ship **reproducible** notebooks + reports.

---

## Modules (ECUEs), core skills & **strategic resources**

### 1) Clustering (A. SAME)
**What you’ll master**
- Partitioning & init: K-Means++ / MiniBatchKMeans; Spherical K-Means for normalized text/audio embeddings.  
- Model-based: **Gaussian Mixture Models (EM)**; **mixture of t-distributions** for robustness.  
- Density: **DBSCAN/HDBSCAN** (epsilon/minPts vs. min_cluster_size/min_samples).  
- Hierarchical: Agglomerative (linkage, distance metrics) + dendrogram cuts.  
- Spectral: graph Laplacian, eigenmaps, affinity construction (RBF, mutual k-NN), scaling to large N.  
- Diagnostics: **silhouette profiles**, elbow/gap statistic, **cluster tendency (Hopkins)**, **stability via bootstrap/consensus**.

**Do-first labs (pro level)**
- **Consensus clustering**: run 50–100 bootstrap resamples across K-Means/GMM/HDBSCAN; compute co-association matrix; cut with hierarchical clustering; compare ARI/NMI vs. single runs.  
- **Scalable spectral**: Nyström approximation for large graphs; compare exact vs. approximated eigenmaps on ~100k samples.  
- **Spherical vs. Euclidean**: cluster sentence embeddings (all-MiniLM) with Spherical K-Means vs. HDBSCAN; evaluate topic coherence.

**Strategic resources**
- scikit-learn Clustering Guide: https://scikit-learn.org/stable/modules/clustering.html  
- HDBSCAN docs: https://hdbscan.readthedocs.io/  
- Spectral clustering tutorial (Luxburg): https://www.cs.upc.edu/~csplanas/teaching/pdf/Luxburg2007_tutorial_spectral_clustering.pdf  
- Gap Statistic (Tibshirani et al.): https://statweb.stanford.edu/~gwalther/gap  
- Mixture of t-distributions (Peel & McLachlan): https://www.sciencedirect.com/science/article/pii/S0167947300000920  
- Distill: “How to Use t-SNE Effectively” (for diagnostics that also inform clustering embeddings): https://distill.pub/2016/misread-tsne/  
- **Videos**  
  - Stanford CS229 (Clustering, EM): https://see.stanford.edu/Course/CS229  
  - StatQuest (intros but crisp math intuition): https://www.youtube.com/c/joshstarmer

---

### 2) Mixture Models → **Latent Block Models (LBM)** (January block)
**What you’ll master**
- **GMM + EM** from scratch; covariance types; model order via **AIC/BIC** and **ICL**.  
- Mixtures for **heavy-tailed** data (t-mixtures); **Dirichlet Process** mixtures (non-param K).  
- **Latent Block Models / co-clustering** for rectangular data (users × items, terms × docs).

**Do-first labs (pro level)**
- **EM from scratch** on synthetic data; verify monotonic log-likelihood. Add **t-mixture** variant and compare robustness under outliers.  
- **LBM on user×item**: Spectral CoClustering vs. Bregman co-clustering; analyze blocks; compare to MF baselines.

**Strategic resources**
- scikit-learn Mixture models: https://scikit-learn.org/stable/modules/mixture.html  
- DP-GMM concept (Bishop, PRML Chapter 10): https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book  
- Bregman co-clustering (Banerjee et al., KDD’04): https://www.cs.cornell.edu/~kilian/papers/bregman.pdf  
- Spectral CoClustering/Biclustering (sklearn): https://scikit-learn.org/stable/modules/clustering.html#biclustering  
- LBM reference (Govaert & Nadif): https://hal.science/hal-00434486/document  
- **Videos**  
  - EM & Mixtures (UVA/CS 6501 guest lecture style): https://www.youtube.com/watch?v=REypj2sy_5U  
  - AIC/BIC by StatQuest: https://www.youtube.com/watch?v=3u-_h2XHSEg

---

### 3) Dimensionality Reduction (M.N)
**What you’ll master**
- **Linear**: PCA/KPCA (kernels), whitening, explained variance; randomized SVD for scale.  
- **Manifold**: **UMAP**, **t-SNE**, **Isomap**, **LLE**; affinity choices; crowding problem; **trustworthiness/continuity**.  
- **Out-of-sample**: UMAP `.transform`, KPCA pre-image; **openTSNE** for adding new points.  
- Visual diagnostics: class overlap, local neighborhoods, **“holes”** indicating density effects, **param sweeps** (perplexity, n_neighbors, min_dist).

**Do-first labs (pro level)**
- **UMAP vs. t-SNE** on 100k points with FAISS k-NN; compute **trustworthiness** curves; produce a **decision card** for parameters by data regime.  
- **Embedding stability**: re-fit UMAP/t-SNE under re-sampling & noise; report Procrustes-aligned drift and neighborhood preservation.  
- **KPCA + linear model**: map to feature space, train linear classifier (for sanity), and project coefficients back.

**Strategic resources**
- UMAP: https://umap-learn.readthedocs.io/  
- openTSNE (fast t-SNE with transform): https://opentsne.readthedocs.io/  
- scikit-learn Manifold: https://scikit-learn.org/stable/modules/manifold.html  
- PCA/KPCA docs: https://scikit-learn.org/stable/modules/decomposition.html#pca  
- Distill t-SNE: https://distill.pub/2016/misread-tsne/  
- “Dimensionality Reduction: A Comparative Review” (Lee & Verleysen): https://arxiv.org/abs/0905.3968  
- **Videos**  
  - UMAP creator talk (McInnes): “UMAP: Uniform Manifold Approximation and Projection” — (PyData/YouTube) https://www.youtube.com/watch?v=nq6iPZVUxZU  
  - PCA deep dive (StatQuest): https://www.youtube.com/watch?v=FgakZw6K1QQ

---

### 4) Factorization & Recommendation
**What you’ll master**
- Low-rank models: **SVD/SVD++**, **NMF**, **ALS** for implicit feedback; regularization, cold-start with side features.  
- Ranking metrics (**NDCG@K / MAP@K / Recall@K**), **diversity** (intra-list), **coverage**; production considerations (approx nearest neighbors).

**Do-first labs (pro level)**
- **Implicit-ALS vs. LightFM (hybrid)** on **MovieLens 20M**; evaluate MAP/NDCG@K with **multiple random splits**, plot CIs; compute **coverage** and **Gini** diversity.  
- **Ann-accelerated recommenders**: build item embeddings, serve top-N with **FAISS** (IVF-PQ); compare latency vs. exact.

**Strategic resources**
- Implicit (ALS): https://implicit.readthedocs.io/  
- LightFM (hybrid MF): https://making.lyst.com/lightfm/docs/  
- Surprise (SVD baselines): https://surpriselib.com/  
- FAISS (ANN search): https://github.com/facebookresearch/faiss  
- Recommender evaluation survey (metrics pitfalls): https://dl.acm.org/doi/10.1145/3285029  
- MovieLens datasets: https://grouplens.org/datasets/movielens/  
- **Videos**  
  - “Matrix Factorization for Recsys” (Alex K): https://www.youtube.com/watch?v=ZspR5PZemcs  
  - RecSys tutorials (various talks): https://www.youtube.com/c/ACMRecSys

---

## Contact hours & key dates (from your calendar)
**Approx. total (UE): ~70 hours**

- **Clustering:** 10/10/2024 (full), 15/10/2024 (full), 17/10/2024 (full) → **≈ 21 h**  
- **Dimensionality Reduction:** 09/10/2024 (full), 23/10/2024 (full), 24/10/2024 (full) → **≈ 21 h**  
- **Factorization & Recommendation (extended slots):** 31/10, 05/11, 07/11, 12/11, 21/11 (16:30–18:30) → **≈ 10 h**  
- **Mixture Models → LBM:** 14/01 (PM), 15/01 (PM), 16/01 (PM), 21/01 (AM+PM) → **≈ 17.5 h**  
- **Exams (UE-related):** Dimensionality Reduction / DEL Exam — **07/11/2024 PM** · Mixture Models Exam — **04/02/2025 AM**  
> Sessions: **09:00–12:30** (AM) · **14:00–17:30** (PM). Extended slots as noted.

---

## Assessment & grading (suggested — adapt to official rubric)
- **Unsupervised capstone (team)** — end-to-end pipeline on a realistic dataset: **50%**  
  *Prep → representation → ≥2 clustering families → stability & internal metrics → factorization/recsys baseline → exec summary & risks.*  
- **Dimensionality Reduction / DEL Exam (07/11 PM):** **15%**  
- **Mixture Models Exam (04/02 AM):** **20%**  
- **Short graded notebook (manifold diagnostics):** **5%**  
- **Participation & lab check-offs:** **10%**

---

## Hands-on practice path (exact order to **outperform**)
1. **Data readiness & scaling** — Robust vs. Standard scaling; log/yeo-johnson; PCA sanity checks (variance/scree/loadings).  
2. **Core clustering sweep** — Grid over K, covariance, min_cluster_size; compare **Silhouette/CH/DB**, **stability** via bootstrap; visualize in PCA/UMAP.  
3. **Mixture models deep-dive** — EM from scratch; t-mixtures; AIC/BIC/ICL; responsibilities heatmaps.  
4. **Latent Block Models** — Spectral/Bregman co-clustering on user×item; compare with MF; interpret blocks.  
5. **Manifold learning lab** — UMAP vs. t-SNE with **trustworthiness**; create a parameter **decision card**.  
6. **Factorization & RecSys** — Implicit-ALS vs. LightFM; **NDCG@K/MAP@K/Recall@K**; add **FAISS** ANN for serving.  
7. **(Bonus) Anomaly detection** — Isolation Forest / One-Class SVM / LOF; evaluate precision@K on labeled anomalies.

---

## Lab starters & exemplar repos (ready-to-run)
- scikit-learn examples (clustering/mixture/manifold): https://scikit-learn.org/stable/auto_examples/index.html  
- HDBSCAN: https://hdbscan.readthedocs.io/ — UMAP: https://umap-learn.readthedocs.io/ — openTSNE: https://opentsne.readthedocs.io/  
- Spectral co-/bi-clustering: https://scikit-learn.org/stable/modules/clustering.html#biclustering  
- Implicit: https://implicit.readthedocs.io/ — LightFM: https://making.lyst.com/lightfm/docs/ — Surprise: https://surpriselib.com/  
- FAISS ANN: https://github.com/facebookresearch/faiss — MovieLens: https://grouplens.org/datasets/movielens/

---

## Reference stack — quick links
**Clustering & Mixtures**  
KMeans/MiniBatch: https://scikit-learn.org/stable/modules/clustering.html#k-means · GaussianMixture (EM): https://scikit-learn.org/stable/modules/mixture.html · Hierarchical: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering · DBSCAN/HDBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan — https://hdbscan.readthedocs.io/ · Spectral: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

**Dim. Reduction & Manifolds**  
PCA/KPCA: https://scikit-learn.org/stable/modules/decomposition.html#pca · t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html · UMAP: https://umap-learn.readthedocs.io/ · Isomap/LLE: https://scikit-learn.org/stable/modules/manifold.html · Review: https://arxiv.org/abs/0905.3968

**Factorization & Recsys**  
NMF/SVD: https://scikit-learn.org/stable/modules/decomposition.html · Implicit ALS: https://implicit.readthedocs.io/ · LightFM: https://making.lyst.com/lightfm/docs/ · Metrics pitfalls (survey): https://dl.acm.org/doi/10.1145/3285029

**Evaluation & Utilities**  
Internal metrics: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation · NMI/ARI: same page · Model selection: https://scikit-learn.org/stable/modules/grid_search.html

---

## Deliverables & submission checklist
- ✅ Reproducible notebooks + `src/unsup/*` utilities with **deterministic seeds** and config files.  
- ✅ Comparative study across **≥3 clustering families** with **stability** and **internal metrics** (and external when available).  
- ✅ Manifold learning **diagnostics** (trustworthiness/continuity) + **parameter decision card**.  
- ✅ **Factorization/Recsys** experiment with **NDCG@K / MAP@K / Recall@K**, CI over multiple splits, and **diversity/coverage**.  
- ✅ Executive summary: method choice rationale, risks (spurious clusters, curse of dimensionality), limitations & next steps.

---

## Toolkit prerequisites
- **Python 3.10+**, Jupyter/VS Code; **NumPy/SciPy/scikit-learn**, **umap-learn**, **hdbscan**, **implicit**/**LightFM**, optional **openTSNE**, **faiss** (ANN).  
- Optional deep reps: **PyTorch** or **TensorFlow** for SimCLR/VICReg/Barlow Twins embeddings before clustering.

---

## Folder plan (this UE)
``` text
02-Unsupervised-Learning/
├─ README.md # this file
├─ Clustering/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Mixture-Models/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Dimensionality-Reduction/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
└─ Factorisation-Recommandation/
├─ notes.md
├─ resources.md
└─ notebooks/

```

