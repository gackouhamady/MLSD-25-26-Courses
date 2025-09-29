# UE Professional Project (PPD) & Defenses (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE is your springboard to a **publishable, recruiter-ready portfolio** and a **convincing oral defense**. You’ll scope a meaningful problem, execute with MLOps-grade rigor (reproducibility, testing, CI/CD), communicate with **executive clarity**, and defend your work in **PPD soutenances** (Mar) and final **FA/FI defenses** (Jun/Sep).

---

## Learning outcomes
By the end of this UE, you will be able to:
- **Scope & design** an ML/NLP/DS project with clear KPIs, risks, and ethical guardrails; keep a defensible research log.
- Build a **reproducible pipeline** (data → modeling → eval → reporting) with experiments tracked, artifacts versioned, and results auditable.
- Communicate insights with **papers/posters/slides** that survive hard questioning; handle Q&A and live demos with poise.
- Package your work as a **portfolio**: a clean GitHub repo (license, tests, docs, model/data cards, DOI) that hiring managers and juries trust.

---

## Activities (ECUEs), core skills & **strategic resources**

### 1) Project Scoping & Research Design
**What you’ll master**
- Problem framing (user/business/science), **success criteria** (KPIs + guardrails), risks/assumptions, data access & governance.
- Method selection & baseline definition; ablations plan; preregistered acceptance criteria; decision logs.
- Planning: milestones, RACI, Gantt; communication cadence (weekly updates; demo days).

**Templates & resources**
- **CRISP-DM** overview — https://www.the-modeling-agency.com/crisp-dm.pdf  
- **Cookiecutter Data Science** — https://drivendata.github.io/cookiecutter-data-science/  
- **Design Docs (RFC/RFD)** — https://google.github.io/eng-practices/rfcs/  
- **The Turing Way** (reproducible research) — https://the-turing-way.netlify.app/  
- **Good Enough Practices in Scientific Computing** — https://doi.org/10.1371/journal.pcbi.1005510

---

### 2) MLOps & Reproducibility (Code, Data, Experiments)
**What you’ll master**
- Environment mgmt (conda/uv/poetry), Docker (multi-stage), `pre-commit` (black/ruff/mypy), tests/CI (GitHub Actions).
- **Data & model versioning**: **DVC** or **lakeFS**; experiment tracking (**MLflow** / **Weights & Biases**).
- Artifact packaging: wheels, model serialization (ONNX/SKOPS/TorchScript); API for demos (FastAPI/Gradio/Streamlit).

**Templates & resources**
- **MLflow** — https://mlflow.org/ · **Weights & Biases** — https://docs.wandb.ai/  
- **DVC** — https://dvc.org/doc · **lakeFS** — https://docs.lakefs.io/  
- **GitHub Actions (Python)** — https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python  
- **Docker best practices** — https://docs.docker.com/develop/dev-best-practices/  
- **skops (scikit-learn model packaging)** — https://skops.readthedocs.io/

---

### 3) Ethics, Compliance & Documentation
**What you’ll master**
- **GDPR-aware** data handling (PII minimization, consent, retention); risk register; model risk narratives.
- Transparent documentation: **Data Cards** & **Model Cards**; bias/fairness checks (where applicable).
- Licensing, attribution, and dataset governance; releasing non-sensitive artifacts.

**Templates & resources**
- **EU GDPR** portal — https://gdpr.eu/  
- **Model Cards** — https://modelcards.withgoogle.com/about  
- **Data Cards** — https://ai.googleblog.com/2022/07/data-cards-templates-for-transparent.html  
- **choosealicense.com** — https://choosealicense.com/  
- **Zenodo DOI for GitHub** — https://guides.github.com/activities/citable-code/

---

### 4) Writing: Paper, Report & Technical Docs
**What you’ll master**
- IMRaD structure; reproducibility appendix; negative results & limitations; effect sizes & confidence intervals.
- LaTeX/Overleaf workflow; citation mgmt (BibTeX); camera-ready polishing.

**Templates & resources**
- **Overleaf** — https://www.overleaf.com/latex/templates  
- **ACM Primary Article Template** — https://www.acm.org/publications/proceedings-template  
- **IEEE Manuscript Templates** — https://www.ieee.org/conferences/publishing/templates.html  
- **Tidy Data (Wickham)** — https://vita.had.co.nz/papers/tidy-data.pdf

---

### 5) Visual Communication: Poster & Slides
**What you’ll master**
- Poster storytelling (problem → method → results → impact); slide hygiene; executive summaries; live demo design.
- Graphics: color/contrast, typography, data-ink ratio, animation rules (sparingly).

**Templates & resources**
- **Better Poster** (M. Morrison) — https://osf.io/ef53g/  
- **Beamer** — https://ctan.org/pkg/beamer · **reveal.js** — https://revealjs.com/  
- **Presentation Zen** principles — https://www.presentationzen.com/  
- **Data visualization cheat sheets** (ggplot/Matplotlib) — https://ggplot2.tidyverse.org/ · https://matplotlib.org/stable/tutorials/

---

### 6) Oral Defense & Q&A (PPD, FA & FI)
**What you’ll master**
- Talk structure (hook → stakes → method → evidence → decision); **anticipated questions matrix**; live failure modes plan.
- Handling critiques; timeboxing; backup slides; **demo fallbacks** (recorded run + logs + metrics).

**Templates & resources**
- **OG viva/defense tips** (general) — https://www.vitae.ac.uk/doing-research/ending-your-doctorate/defending-your-thesis-viva  
- **Speaking checklists** — https://www.duarte.com/ (Resonate principles)  
- **Timer & rehearsal tools** — any presenter coach (PowerPoint/Keynote), OBS for run-through recordings.

---

## Contact hours & key dates (from your calendar)

**Approx. total (UE): ~98 hours (14 × 7h)**

- **Jan 2025**  
  - **23/01** — PROJECT **mandatory presence** (AM+PM)
- **Late Jan – Feb 2025**  
  - **29/01, 30/01** — PROJECT **mandatory presence** (AM+PM)  
  - **05/02, 06/02, 10/02, 11/02, 12/02, 13/02** — **MLSD FA (PPD) / MLSD FI (Soutenance PPD)** blocks (AM+PM on most days; 10/02 PM also RL exam)
- **March 2025 — PPD focus**  
  - **05/03, 06/03, 12/03, 13/03, 19/03, 20/03** — **PPD mandatory presence** (AM+PM)  
  - **26/03–27/03** — **Soutenances-PPD** (AM+PM)
- **Final Defenses & Juries**  
  - **FA Defenses:** **22–25/06/2025** · **FA Jury:** **04/09/2025**  
  - **FI Defenses:** **14–18/09/2025** · **FI Jury (global end date):** **22/09/2025**

> Sessions: **09:00–12:30** (AM) · **14:00–17:30** (PM). Extended exam/defense slots as announced.

---

## Assessment & grading (suggested — align with official rubric)
- **Project Delivery (Code + Experiments + MLOps)** — **40%**  
  Reproducible pipeline, tracked experiments, quality gates, proper docs.
- **Written Report / Paper (Overleaf)** — **20%**  
  IMRaD + reproducibility appendix + limitations/ethics.
- **Poster + Demo** — **15%**  
  Visual narrative, live demo or recorded fallback, Q&A fluency.
- **Oral Defense (PPD/Final)** — **15%**  
  Clarity, timing, evidence, handling objections.
- **Process & Portfolio Quality** — **10%**  
  Repo hygiene (README, LICENSE, CITATION.cff, CONTRIBUTING, Code of Conduct), issues/PRs, decision log.

---

## Practice path (to **outperform**)

1. **Week 0–1 — Framing & plan**  
   Design doc (problem, KPIs, risks), data access, milestones, ethics checklist, repo scaffold (cookiecutter).
2. **Week 2–4 — Baseline & infra**  
   Data EDA + baseline model; set up DVC/MLflow/W&B; Docker + CI; create decision log.
3. **Week 5–7 — Experiments**  
   Ablations & hyper-sweeps; fixed seeds; confidence intervals; failure analysis; update risk register.
4. **Week 8 — Reporting**  
   Draft Overleaf paper; build poster skeleton; record a 2-min demo; dry-run Q&A.
5. **Week 9 — PPD Soutenance**  
   Rehearse with timer; assemble backup slides; finalize poster/demo; deliver talk; capture feedback/actions.
6. **After PPD → Final Defenses**  
   Address feedback, polish paper, harden repo (tags/DOI), practice viva.

---

## Lab starters & exemplar repos
- **Cookiecutter DS** — https://drivendata.github.io/cookiecutter-data-science/  
- **DVC example-dvc-experiment** — https://github.com/iterative/example-dvc-experiments  
- **MLflow examples** — https://github.com/mlflow/mlflow/tree/master/examples  
- **W&B reports gallery** — https://docs.wandb.ai/guides/reports  
- **FastAPI template** — https://github.com/tiangolo/full-stack-fastapi-template  
- **Awesome research templates** — https://github.com/terryum/awesome-deep-learning-papers (reading) · https://github.com/boennemann/awesome-guidelines (writing/coding)

---

## Reference stack — quick links
**Reproducibility & Tracking**  
MLflow — https://mlflow.org/ · W&B — https://docs.wandb.ai/ · DVC — https://dvc.org/doc · lakeFS — https://docs.lakefs.io/

**DevEx & Packaging**  
GitHub Actions — https://docs.github.com/actions/ · pre-commit — https://pre-commit.com/ · black — https://black.readthedocs.io/ · ruff — https://docs.astral.sh/ruff/ · mypy — https://mypy.readthedocs.io/  
Docker — https://docs.docker.com/ · Poetry — https://python-poetry.org/docs/

**Docs & Publishing**  
Overleaf — https://www.overleaf.com/ · ACM — https://www.acm.org/publications/proceedings-template · IEEE — https://www.ieee.org/conferences/publishing/templates.html  
choosealicense — https://choosealicense.com/ · Zenodo DOI — https://guides.github.com/activities/citable-code/

---

## Deliverables & submission checklist
- ✅ **GitHub repo** (public or private) with: `README.md`, `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `docs/`, `Makefile`/`invoke` tasks.  
- ✅ **Repro**: `environment.yml` or `pyproject.toml` + lock; **Dockerfile**; `docker-compose.yml` if services.  
- ✅ **Data & model versioning**: DVC or lakeFS; storage remotes configured; sensitive data excluded.  
- ✅ **Tracking**: MLflow/W&B runs; `reports/` with plots & tables; `results.csv` + **decision log**.  
- ✅ **Quality gates**: tests, style, type checks in CI; pre-commit hooks.  
- ✅ **Docs**: Overleaf report (PDF), **poster**, **slides**, demo (**FastAPI/Gradio/Streamlit**) or recorded fallback.  
- ✅ **Ethics**: Model/Data Cards; GDPR checklist; risk register.  
- ✅ **Release**: git tag, changelog, Zenodo DOI (if open).

---

## Toolkit prerequisites
- **Python 3.10+**, Docker Desktop, Git, VS Code/JetBrains, Overleaf account.  
- Recommended libraries: scikit-learn/PyTorch + tracking (MLflow/W&B) + DVC/lakeFS + FastAPI/Gradio/Streamlit + testing & linting stack (pytest, black, ruff, mypy, pre-commit).

---

## Folder plan (this UE)
```text
06-PROJET-MLSD/
├─ README.md # project charter & quickstart
├─ docs/
│ ├─ paper/ # Overleaf export or TeX sources
│ ├─ poster/
│ └─ slides/
├─ data/ # DVC-tracked; raw/processed placeholders (no PII)
├─ models/ # serialized artifacts (DVC-tracked)
├─ src/ # package code
├─ notebooks/ # exploratory; keep light; move to src/ when stable
├─ configs/ # yaml/json configs for runs
├─ reports/ # metrics, plots, tables, decision log
├─ tests/ # unit/integration tests
├─ .github/workflows/ # CI pipelines
└─ docker/ # Dockerfile(s), compose, runtime configs
```
