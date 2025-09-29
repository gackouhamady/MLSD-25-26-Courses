# UE Unsupervised Learning (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE develops mastery of **representation learning and structure discovery** without labels: clustering (partitioning, density, hierarchical, spectral), **mixture models** (EM/GMM, latent block models), **dimensionality reduction** (linear & manifold), and **matrix factorization & recommendation**. You will design **production-grade unsupervised pipelines** with rigorous validation, stability analysis, and clear business-facing deliverables.

---

## Learning outcomes
By the end of this UE, you will be able to:
- Build **end-to-end unsupervised pipelines**: preprocessing → representation → clustering/factorization → evaluation → reporting.
- Choose and tune methods: **K-Means/MBKMeans, GMM (EM), HDBSCAN/DBSCAN, agglomerative, spectral**, **PCA/KPCA/UMAP/t-SNE/Isomap**, **NMF/SVD/ALS**, **Latent Block Models (co-clustering)**.
- Evaluate with **internal metrics** (Silhouette, Calinski–Harabasz, Davies–Bouldin), **stability** (bootstrapping), and **external metrics** when labels exist (NMI/ARI).
- Apply **matrix factorization** to recommendations (implicit feedback, cold-start), measure **Recall@K / MAP@K / NDCG@K**.
- Communicate insights with **well-designed diagnostics & visuals** and ship reproducible notebooks.

---

## Modules (ECUEs) and core skills

**Clustering (A. SAME)**  
- Partitioning & initialization (K-Means++, MiniBatchKMeans), model-based clustering (GMM/EM), hierarchical (linkage/merging), density (DBSCAN/HDBSCAN), spectral clustering.  
- Cluster diagnostics: silhouette profiles, elbow/gap, stability via resampling, cluster tendency (Hopkins).  

**Mixture Models → LBM (Jan block)**  
- Gaussian Mixture Models & EM; component selection (AIC/BIC).  
- **Latent Block Models (co-clustering)** for user×item/event×feature matrices; biclustering formulations; sparse regularization.  

**Dimensionality Reduction (M.N)**  
- **PCA/KPCA**, whitening, explained variance;  
- **Manifold learning**: t-SNE (perplexity/learning rate), **UMAP** (n_neighbors/min_dist), **Isomap/LLE**;  
- Embedding quality (trustworthiness/continuity), out-of-sample transforms.  

**Factorization & Recommendation**  
- Low-rank models: **SVD/SVD++**, **NMF**, **ALS** on implicit feedback; hybrid & content features;  
- Offline evaluation: **Recall@K, MAP@K, NDCG@K**, coverage & diversity; serving top-N.

---

## Contact hours & key dates (from your calendar)

**Approx. total (UE): ~70 hours**

- **Clustering:** 10/10/2024 (full), 15/10/2024 (full), 17/10/2024 (full) → **≈ 21 h**  
- **Dimensionality Reduction:** 09/10/2024 (full), 23/10/2024 (full), 24/10/2024 (full) → **≈ 21 h**  
- **Factorization & Recommendation (extended slots):**  
  31/10/2024 (16:30–18:30), 05/11/2024 (16:30–18:30), 07/11/2024 (16:30–18:30),  
  12/11/2024 (16:30–18:30), 21/11/2024 (16:30–18:30) → **≈ 10 h**  
- **Mixture Models → LBM:** 14/01/2025 (PM), 15/01/2025 (PM), 16/01/2025 (PM), 21/01/2025 (AM+PM) → **≈ 17.5 h**  
- **Exams (UE-related):**  
  - **Dimensionality Reduction / DEL Exam:** 07/11/2024 (PM)  
  - **Mixture Models Exam (M.N):** 04/02/2025 (AM)

> Sessions: **09:00–12:30** (AM) · **14:00–17:30** (PM). Extended slots noted above.

---

## Assessment & grading (suggested structure — adapt to official rubric)

- **Unsupervised capstone (team)** — end-to-end pipeline on a realistic, messy dataset: **50%**  
  *Data prep → representation → ≥2 clustering families → stability & internal metrics → factorization/recsys baseline → executive summary & risks.*  
- **Dimensionality Reduction / DEL Exam (07/11/2024 PM):** **15%**  
- **Mixture Models Exam (04/02/2025 AM):** **20%**  
- **Short graded notebook (manifold learning diagnostics):** **5%**  
- **Participation & lab check-offs (all sessions):** **10%**

---

## Hands-on practice path (do these in order)

1. **Data readiness & scaling**  
   - Robust scaling vs standardization; treat outliers & skew; PCA sanity checks (variance, scree, loadings).  
2. **Core clustering sweep**  
   - Grid over K-Means (K, init, batch size), GMM (components, covariance), HDBSCAN (min_cluster_size, min_samples), Agglomerative (linkage, distance).  
   - Compare with **Silhouette/CH/DB**, **stability** via bootstrap; visualize embeddings (PCA/UMAP).  
3. **Mixture models deep-dive**  
   - EM from scratch (toy) → scikit-learn GMM; model selection AIC/BIC; mixture of t-distributions (robust alt.); cluster posterior responsibilities.  
4. **Latent Block Models / Co-clustering**  
   - Try **SpectralCoclustering** / **SpectralBiclustering** on user×item; interpret blocks and compare with MF.  
5. **Manifold learning lab**  
   - t-SNE vs UMAP on the same dataset; measure **trustworthiness**; create a decision card (which to use when).  
6. **Factorization & Recommendation**  
   - **Implicit ALS** on sparse feedback; **LightFM** hybrid model; evaluate **NDCG@K/MAP@K/Recall@K**; report diversity & coverage.  
7. **Anomaly detection (bonus)**  
   - Isolation Forest / One-Class SVM / LOF; thresholding via top-K or extreme quantiles; precision@K on labeled anomalies.

---

## Lab starters & exemplar repos

- **scikit-learn examples** (clustering, mixture, manifold): https://scikit-learn.org/stable/auto_examples/index.html  
- **HDBSCAN**: https://hdbscan.readthedocs.io/  
- **UMAP-learn**: https://umap-learn.readthedocs.io/  
- **t-SNE docs (sklearn)**: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html  
- **Spectral co-clustering/biclustering (sklearn)**: https://scikit-learn.org/stable/modules/clustering.html#biclustering  
- **Implicit (ALS for recommendations)**: https://implicit.readthedocs.io/  
- **LightFM (hybrid MF)**: https://making.lyst.com/lightfm/docs/  
- **Surprise (SVD, baselines)**: https://surpriselib.com/

---

## Reference stack — quick links

**Clustering & Mixtures**  
- KMeans/MiniBatchKMeans: https://scikit-learn.org/stable/modules/clustering.html#k-means  
- GaussianMixture (EM): https://scikit-learn.org/stable/modules/mixture.html  
- Agglomerative: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering  
- DBSCAN/HDBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan — https://hdbscan.readthedocs.io/  
- Spectral clustering: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

**Dimensionality Reduction & Manifolds**  
- PCA/KPCA: https://scikit-learn.org/stable/modules/decomposition.html#pca  
- t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html  
- UMAP: https://umap-learn.readthedocs.io/  
- Isomap/LLE: https://scikit-learn.org/stable/modules/manifold.html

**Factorization & Recommendation**  
- NMF/SVD: https://scikit-learn.org/stable/modules/decomposition.html  
- Implicit ALS: https://implicit.readthedocs.io/  
- LightFM: https://making.lyst.com/lightfm/docs/  
- Metrics (ranking): MAP@K/NDCG@K/Recall@K references within libs above

**Evaluation & Utilities**  
- Internal clustering metrics: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation  
- External metrics (NMI/ARI): https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation  
- Model selection: https://scikit-learn.org/stable/modules/grid_search.html

---

## Deliverables & submission checklist

- ✅ Reproducible notebook(s) + utility module (`src/unsup/…`) with **deterministic seeds** and config files.  
- ✅ Comparative study across **≥3 clustering families** with **stability** and **internal metrics**.  
- ✅ Manifold learning **diagnostics** (trustworthiness) and **parameter card** (guidelines for future use).  
- ✅ **Factorization/recsys** experiment with **NDCG@K / MAP@K / Recall@K** and error bars over multiple splits.  
- ✅ Executive summary: method choice rationale, risks (spurious clusters, curse of dimensionality), and next steps.

---

## Toolkit prerequisites

- **Python 3.10+**, Jupyter/VS Code, **NumPy / SciPy / scikit-learn**, **umap-learn**, **hdbscan**, **implicit** and/or **LightFM**;  
- Optional: **PyTorch**/**TensorFlow** for deep embeddings; **faiss** for fast neighbor search on large data.

---

```text
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
## Folder plan (this UE)
- Keep each module self-contained (data → code → report). Track experiments with clear config files and versioned outputs for reproducibility.
