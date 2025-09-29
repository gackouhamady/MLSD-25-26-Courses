# UE Deep Learning & Graph Learning (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE consolidates advanced **Deep Learning (DL)**, **Representation Learning (Embeddings)**, and **Graph Learning (GNNs)**. You will train and evaluate modern neural architectures, design robust embedding pipelines, and build graph-based models for node/edge/graph-level tasks. The UE is tightly aligned with your **Hybrid DialogueGCN** project (ERC): aim for a research-grade pipeline and publishable results (e.g., *Hybrid GCN + adaptive temporal attention*).

---

## Learning outcomes
By the end of this UE, you will be able to:
- Implement and fine-tune **CNNs/RNNs/Transformers** with modern training recipes (mixed precision, schedulers, regularization, distributed training).
- Build **embedding pipelines** (contrastive/metric learning, text/audio/image embeddings), evaluate them for retrieval & downstream tasks.
- Design **graph ML** solutions (GCN/GAT/GraphSAGE/R-GCN, Graph Transformers), including hetero/dynamic graphs, with rigorous evaluation.
- Engineer **research-grade experiments**: ablations, confidence intervals, effect sizes, and cross-dataset generalization.

---

## Modules (ECUEs), core skills & **strategic resources**

### 1) Deep Learning II (B.H)

**What you’ll master**
- Architectures & training: **ResNets/ConvNeXt**, **BiLSTM/GRU**, **Transformers (ViT/Encoder-Decoder)**; normalization & regularization (LayerNorm, Dropout, Stochastic Depth).  
- Training systems: **PyTorch** (Autograd, AMP), **Lightning** (loops, callbacks, multi-GPU), gradient accumulation, **LR schedulers** (One-Cycle, Cosine), advanced optimizers (AdamW, Lion).  
- Transfer/self-supervised: **MAE**, **SimCLR/BYOL/VICReg**, **DINO**; task adaptation and linear probing vs. full fine-tuning.

**Do-first labs (pro level)**
- **Strong supervised baseline**: Image (CIFAR-100 or Tiny-ImageNet) → `timm` model + One-Cycle LR + CutMix/MixUp; report accuracy **with CIs** over ≥5 seeds.  
- **Self-supervised pretraining**: train SimCLR/DINO on a subset; linear probe vs. full fine-tune; measure transfer on a different domain.  
- **Efficient training**: AMP + gradient checkpointing + compiled mode; compare wall-clock vs. accuracy trade-offs.

**Strategic resources**
- PyTorch: https://pytorch.org/docs/stable/  
- Lightning: https://lightning.ai/docs/pytorch/stable/  
- `timm` (SOTA model zoo): https://huggingface.co/docs/timm/index  
- Transformers (HF): https://huggingface.co/docs/transformers/index  
- SimCLR: https://arxiv.org/abs/2002.05709 · BYOL: https://arxiv.org/abs/2006.07733 · DINO: https://arxiv.org/abs/2104.14294 · MAE: https://arxiv.org/abs/2111.06377  
- Stanford CS231n (vision): http://cs231n.stanford.edu/ · Stanford CS224n (NLP): http://web.stanford.edu/class/cs224n/

---

### 2) Data Embedding & Learning — **DEL** (I.K)

**What you’ll master**
- **Text embeddings**: static (GloVe/fastText) vs. contextual (**BERT/MPNet/e5**); pooling strategies; domain adaptation.  
- **Metric / contrastive learning**: **InfoNCE**, triplet margin, multi-positive pairs; hard-negative mining; temperature tuning.  
- **Multimodal**: **CLIP-style** alignment (text–image/audio); joint encoders vs. dual encoders; retrieval at scale with ANN.  
- **Evaluation**: retrieval metrics (**Recall@K, MRR, nDCG**), clustering coherence, probing (linear eval).

**Do-first labs (pro level)**
- **Sentence retrieval**: fine-tune **Sentence-Transformers** on STS/Quora; evaluate Recall@K/MRR; export to FAISS index and test latency vs. accuracy.  
- **Multimodal mini-CLIP**: train a small dual encoder on a curated text–image set; analyze zero-shot transfer to a held-out domain.  
- **Domain adaptation**: continue-pretrain a text embedding (MLM/TSDAE) on your target corpus; measure downstream improvement.

**Strategic resources**
- Sentence-Transformers: https://sbert.net/  
- Hugging Face Datasets/Evaluate: https://huggingface.co/docs/datasets/ · https://huggingface.co/docs/evaluate/  
- e5 embeddings: https://arxiv.org/abs/2402.05680  
- CLIP paper: https://arxiv.org/abs/2103.00020  
- FAISS (ANN at scale): https://github.com/facebookresearch/faiss  
- Metric learning survey: https://arxiv.org/abs/1811.12649

---

### 3) Graph Learning (S.A)

**What you’ll master**
- **GNN basics**: **GCN**, **GraphSAGE**, **GAT**; message passing, oversmoothing, residual/skip connections, normalization (PairNorm/BatchNorm).  
- **Relational & hetero graphs**: **R-GCN**, metapaths; knowledge graphs; **link prediction** (DistMult/ComplEx).  
- **Temporal/dynamic graphs**: TGAT/TGN ideas; sliding windows; edge time encoding.  
- **Graph Transformers**: positional encodings (Laplacian eigenvectors/RW), attention sparsity, k-NN graphs for non-relational data.  
- **Evaluation**: **OGB** leaderboards, Hits@K/MRR/AUC; transductive vs. inductive splits; leakage-safe negative sampling.

**Do-first labs (pro level)**
- **Node classification** (Cora/Citeseer/ogbn-arxiv): compare GCN vs. GraphSAGE vs. GAT; track compute/params; report mean±std over seeds.  
- **Link prediction** (ogbl-citation2): train relational decoder; do **proper negative sampling**; evaluate MRR/Hits@K.  
- **Hetero ERC graph**: build a conversation graph (utterance nodes, speaker edges, temporal edges); baseline **GCN/R-GCN** + **adaptive temporal attention** (aligns with Hybrid DialogueGCN).

**Strategic resources**
- PyTorch Geometric (PyG): https://pytorch-geometric.readthedocs.io/  
- DGL: https://www.dgl.ai/  
- Open Graph Benchmark: https://ogb.stanford.edu/  
- Graph Transformers (Graphormer): https://arxiv.org/abs/2106.05234  
- DialogueGCN (ERC): https://aclanthology.org/D19-1015/  
- TUDatasets: https://chrsmrrs.github.io/datasets/docs/datasets/  
- GraphGym (PyG experimentation): https://pytorch-geometric.readthedocs.io/en/latest/advanced/graph_gym.html

---

## Contact hours & key dates (from your calendar)

**Approx. total (UE): ~63 hours (9 × 7h)**  
- **DEL (I.K):** 16/10/2024 (AM+PM), 22/10/2024 (AM+PM), 19/11/2024 (AM+PM) → **≈ 21 h**  
- **Deep Learning II (B.H):** 13/11/2024 (AM+PM), 14/11/2024 (AM+PM), 20/11/2024 (AM+PM) → **≈ 21 h**  
- **Graph Learning (S.A):** 27/11/2024 (AM+PM), 04/12/2024 (AM+PM), 11/12/2024 (AM+PM) → **≈ 21 h**  
> Sessions: **09:00–12:30** (AM) · **14:00–17:30** (PM). Confirm any later changes with instructors.

---

## Assessment & grading (suggested — align with official rubric)
- **Project A — DEL**: Embedding pipeline with retrieval eval (Recall@K/MRR), ANN serving, and domain adaptation study — **30%**  
- **Project B — DL II**: Strong supervised/self-supervised baseline with ablations, efficiency report (AMP/throughput/latency) — **30%**  
- **Project C — Graph Learning**: Node/link/graph task on OGB or ERC graph with **Hybrid DialogueGCN** components — **30%**  
- **Participation & lab check-offs** — **10%**

---

## Practice path (to **outperform**)
1. **DL II first**: lock a reliable training recipe (One-Cycle, AMP, CutMix/MixUp, early stopping). Establish **seed-averaged** baselines with CIs.  
2. **DEL second**: fine-tune sentence embeddings + ANN serving; write a **decision card** (which model & params under which data regime).  
3. **Graph learning third**: replicate a **PyG** baseline; then integrate **adaptive temporal attention** and **hetero edges** for ERC (DialogueGCN++).  
4. **Research-grade reporting**: ablations, effect sizes, confidence intervals, cross-dataset tests; provide a clean **repro repo**.

---

## Lab starters & exemplar repos
- **PyTorch Tutorials**: https://pytorch.org/tutorials/ · **Lightning templates**: https://lightning.ai/templates  
- **timm training recipes**: https://github.com/huggingface/pytorch-image-models  
- **Sentence-Transformers examples**: https://www.sbert.net/examples/  
- **FAISS demos**: https://github.com/facebookresearch/faiss/tree/main/demos  
- **PyG examples**: https://github.com/pyg-team/pytorch_geometric/tree/master/examples  
- **DGL examples**: https://github.com/dmlc/dgl/tree/master/examples  
- **OGB baselines**: https://github.com/snap-stanford/ogb

---

## Reference stack — quick links

**Core DL & Training**  
PyTorch — https://pytorch.org/docs/stable/ · Lightning — https://lightning.ai/docs/pytorch/stable/ · timm — https://huggingface.co/docs/timm/index · HF Transformers — https://huggingface.co/docs/transformers/index

**Self-Supervised & Embeddings**  
SimCLR — https://arxiv.org/abs/2002.05709 · BYOL — https://arxiv.org/abs/2006.07733 · DINO — https://arxiv.org/abs/2104.14294 · MAE — https://arxiv.org/abs/2111.06377 · Sentence-Transformers — https://sbert.net/ · FAISS — https://github.com/facebookresearch/faiss

**Graph ML**  
PyG — https://pytorch-geometric.readthedocs.io/ · DGL — https://www.dgl.ai/ · OGB — https://ogb.stanford.edu/ · Graphormer — https://arxiv.org/abs/2106.05234 · DialogueGCN — https://aclanthology.org/D19-1015/

---

## Deliverables & submission checklist
- ✅ **Reproducible** projects for each module (configs, fixed seeds, env lockfile).  
- ✅ **DL II**: baseline + self-supervised variant; efficiency report (throughput, memory, latency).  
- ✅ **DEL**: embedding model + retrieval eval (Recall@K/MRR/nDCG); **FAISS** index & latency benchmarks.  
- ✅ **Graph**: GNN baseline + your **Hybrid DialogueGCN** extension; ERC metrics + ablations; clear diagrams.  
- ✅ Clean **README** with how-to-run, dataset links, and a **decision/ablation log**.

---

## Toolkit prerequisites
- **Python 3.10+**, CUDA-capable GPU recommended.  
- Libraries: PyTorch, Lightning, timm, Hugging Face (transformers/datasets/evaluate), Sentence-Transformers, FAISS, PyTorch Geometric and/or DGL, scikit-learn, NumPy/SciPy, matplotlib/plotly/seaborn.

---

## Folder plan (this UE)
```text
04-DeepLearning-Graph/
├─ README.md # this file
├─ Deep-Learning-II/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Data-Embedding-Learning/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Graph-Learning/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
└─ Advanced-Topics/
├─ notes.md # Graph Transformers, temporal GNNs, hybrid ERC models
└─ notebooks/
```

