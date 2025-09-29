# UE Data Engineering (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE trains you to build reliable, production-grade data pipelines—from raw ingestion to governed, analytics-ready datasets and BI dashboards—using both batch and streaming patterns. You’ll practice with modern table formats, enforce data quality, and package & ship your code like a professional team.

---

## Learning outcomes
By the end of this UE, you will be able to:
- **Ingest, transform, and serve data** in batch and streaming (Kafka/Spark/Beam) with durable formats (Parquet) and lakehouse tables (Delta/Iceberg/Hudi).
- **Model data for analytics** and ship warehouse-ready transformations with **dbt Core** and a BI layer in **R** (tidyverse/ggplot2/Shiny/Quarto) and/or **Power BI/Tableau**.
- **Orchestrate pipelines** and capture lineage & metadata for governance with **Airflow + OpenLineage/Marquez** and **DataHub**.
- **Validate data quality** with **Great Expectations (GX)** / **Soda** / **Deequ** and enforce contracts (Schema Registry) on streams.
- **Package & ship** code (Python packaging, CI/CD, Docker) with pre-commit quality gates and typed code (mypy).

---

## Modules (ECUEs) and core skills

**Data Pre-processing (A.F)**  
- pandas & Polars dataframes; Arrow memory model; DuckDB in-process analytics  
- scikit-learn `Pipeline`/`ColumnTransformer`; runtime validation with **pandera**  
- Links:  
  - pandas — https://pandas.pydata.org/docs/  
  - Polars — https://docs.pola.rs/  
  - PyArrow — https://arrow.apache.org/docs/python/  
  - DuckDB (Python) — https://duckdb.org/docs/api/python/overview  
  - scikit-learn Pipelines — https://scikit-learn.org/stable/modules/compose.html#pipeline  
  - ColumnTransformer — https://scikit-learn.org/stable/modules/compose.html#columntransformer  
  - pandera — https://pandera.readthedocs.io/

**BI (R) (S.A)**  
- Tidy data with **tidyverse**; visuals with **ggplot2**; interactive apps with **Shiny**; reproducible reports with **Quarto**  
- Optionally **Power BI** or **Tableau** for enterprise dashboards  
- Links:  
  - tidyverse — https://www.tidyverse.org/  
  - ggplot2 — https://ggplot2.tidyverse.org/  
  - Shiny — https://shiny.posit.co/  
  - Quarto — https://quarto.org/  
  - Power BI — https://learn.microsoft.com/power-bi/  
  - Tableau — https://www.tableau.com/learn/training

**Big Data Analytics (S.M)**  
- Streaming ETL/ELT: **Apache Kafka** (+ **Connect**, **Schema Registry**), **Debezium** CDC  
- **Spark Structured Streaming** & **Apache Beam**; lakehouse tables (**Delta Lake / Iceberg / Hudi**) over **Parquet**  
- Links:  
  - Kafka — https://kafka.apache.org/documentation/  
  - Kafka Connect — https://kafka.apache.org/documentation/#connect  
  - Confluent Schema Registry — https://docs.confluent.io/platform/current/schema-registry/index.html  
  - Debezium — https://debezium.io/documentation/  
  - Spark Structured Streaming — https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html  
  - Apache Beam — https://beam.apache.org/documentation/programming-guide/  
  - Delta Lake — https://docs.delta.io/  
  - Apache Iceberg — https://iceberg.apache.org/docs/latest/  
  - Apache Hudi — https://hudi.apache.org/docs/overview  
  - Apache Parquet — https://parquet.apache.org/docs/

**Packaging (S.M)**  
- Python packaging with `pyproject.toml` (PEP 517/518); build/publish; Poetry; semantic versioning  
- CI/CD (GitHub Actions), Docker multi-stage, code quality (pre-commit, black, ruff, mypy)  
- Links:  
  - Python Packaging User Guide — https://packaging.python.org/  
  - PEP 517 — https://peps.python.org/pep-0517/  
  - PEP 518 — https://peps.python.org/pep-0518/  
  - `build` — https://pypi.org/project/build/  
  - `twine` — https://twine.readthedocs.io/  
  - Poetry — https://python-poetry.org/docs/  
  - pre-commit — https://pre-commit.com/  
  - black — https://black.readthedocs.io/  
  - ruff — https://docs.astral.sh/ruff/  
  - mypy — https://mypy.readthedocs.io/  
  - Docker multi-stage — https://docs.docker.com/build/building/multi-stage/  
  - GitHub Actions (Python) — https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python

---

## Contact hours & key dates (from your program calendar)

- **Total contact hours (UE)**: **≈ 70 hours**
  - Data Pre-processing: **14 h** — 01/10/2024, 08/10/2024  
  - BI (R): **7 h** — 02/10/2024  
  - Big Data Analytics: **21 h** — 29/10/2024, 30/10/2024, 12/11/2024  
  - Packaging: **28 h** — 03/10/2024; 07/01/2025 (PM); 08/01/2025 (PM); 22/01/2025 (full); 28/01/2025 (full)
- **Assessments (UE scope)**  
  - Big Data **exam** — **17/12/2024 (AM)**  
  - Packaging **exam** — **04/02/2025 (PM)**
- **Rooms**: Lavoisier A / Salle Gley  
> Sessions run **09:00–12:30** and **14:00–17:30**.

---

## Assessment & grading (suggested)

> Align to your official rubric if announced later.

- **Team capstone pipeline** (end-to-end streaming + batch lakehouse, lineage & data quality, documented): **50%**  
  *Kafka → Spark/Beam → Delta/Iceberg/Hudi → dbt marts → BI dashboard; GX/Soda/Deequ checks; OpenLineage + Marquez; CI/CD & Docker.*
- **Big Data exam** (17/12/2024 AM): **15%**  
- **Packaging exam** (04/02/2025 PM): **15%**  
- **BI deliverable** (Shiny/Quarto or Power BI/Tableau) with narrative data story: **10%**  
- **Participation & lab check-offs**: **10%**

---

## Hands-on practice path (do these in order)

1. **Local lakehouse quickstart (Parquet + Delta/Iceberg/Hudi)**  
   - Create bronze/silver/gold; partitioning, clustering/Z-ordering, compaction; IO benchmarks.  
   - Docs: Delta https://docs.delta.io/ · Iceberg https://iceberg.apache.org/docs/latest/ · Hudi https://hudi.apache.org/docs/overview/  

2. **Streaming CDC to lakehouse**  
   - **Debezium** (Postgres/MySQL) → **Kafka Connect** (Avro/Protobuf/JSON) → **Schema Registry** → **Spark Structured Streaming** sink to Delta/Iceberg/Hudi.  
   - Debezium: https://github.com/debezium/debezium-examples · Schema Registry: https://docs.confluent.io/platform/current/schema-registry/index.html  

3. **Transformations with dbt Core**  
   - Build staging → marts; tests & docs; generate lineage site.  
   - dbt intro — https://docs.getdbt.com/docs/introduction · jaffle_shop — https://github.com/dbt-labs/jaffle_shop  

4. **Data quality & contracts**  
   - Add **GX** validations to batch/stream; complementary **Soda** checks or **Deequ** constraints; fail pipeline on violations.  
   - GX — https://docs.greatexpectations.io/docs/ · Soda — https://docs.soda.io/soda/core/ · Deequ — https://github.com/awslabs/deequ  

5. **Orchestration, lineage, catalog**  
   - Deploy DAGs on **Airflow**; emit **OpenLineage** events to **Marquez**; ingest metadata to **DataHub**.  
   - Airflow — https://airflow.apache.org/docs/ · Best practices — https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html  
   - OpenLineage — https://openlineage.io/docs/ · Marquez — https://marquezproject.github.io/marquez/  
   - DataHub — https://www.datahubproject.io/docs/ (quickstart in repo: https://github.com/datahub-project/datahub)

6. **BI productization**  
   - Publish an **R Shiny** app or **Quarto** site (or Power BI/Tableau) on top of gold tables; include reproducible scripts.  
   - Shiny — https://shiny.posit.co/ · Quarto — https://quarto.org/ · Power BI — https://learn.microsoft.com/power-bi/ · Tableau — https://www.tableau.com/learn/training  

7. **Packaging & CI/CD**  
   - Package shared Python libs (`pyproject.toml`), enforce **black/ruff/mypy** with **pre-commit**, build **Docker** multi-stage, run tests on **GitHub Actions**.  
   - Packaging guide — https://packaging.python.org/ · pre-commit — https://pre-commit.com/ · Docker multi-stage — https://docs.docker.com/build/building/multi-stage/  
   - GitHub Actions (Python) — https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python

---

## Lab starters & exemplar repos

- **Confluent “cp-all-in-one”** (Kafka + Connect + Schema Registry + ksqlDB via Docker Compose)  
  https://github.com/confluentinc/cp-all-in-one
- **Debezium examples** (CDC for Postgres/MySQL/Mongo/SQL Server/Oracle)  
  https://github.com/debezium/debezium-examples
- **dbt “jaffle shop”** (canonical starter project)  
  https://github.com/dbt-labs/jaffle_shop
- **Airflow best practices**  
  https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html

---

## Reference stack — quick links

**Batch/Streaming & Formats**  
Spark Structured Streaming — https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html  
Apache Beam — https://beam.apache.org/documentation/programming-guide/  
Parquet — https://parquet.apache.org/docs/  
Delta Lake — https://docs.delta.io/ · Iceberg — https://iceberg.apache.org/docs/latest/ · Hudi — https://hudi.apache.org/docs/overview

**Kafka Ecosystem**  
Kafka — https://kafka.apache.org/documentation/  
Kafka Connect — https://kafka.apache.org/documentation/#connect  
Schema Registry — https://docs.confluent.io/platform/current/schema-registry/index.html  
Debezium — https://debezium.io/documentation/

**Orchestration, Lineage, Catalog**  
Airflow — https://airflow.apache.org/docs/  
OpenLineage — https://openlineage.io/docs/  
Marquez — https://marquezproject.github.io/marquez/  
DataHub — https://www.datahubproject.io/docs/

**Data Quality**  
Great Expectations — https://docs.greatexpectations.io/docs/  
Soda Core — https://docs.soda.io/soda/core/  
AWS Deequ — https://github.com/awslabs/deequ

**Data Pre-processing**  
pandas — https://pandas.pydata.org/docs/  
scikit-learn pipelines — https://scikit-learn.org/stable/modules/compose.html#pipeline  
ColumnTransformer — https://scikit-learn.org/stable/modules/compose.html#columntransformer  
Polars — https://docs.pola.rs/  
PyArrow — https://arrow.apache.org/docs/python/  
DuckDB — https://duckdb.org/docs/api/python/overview  
pandera — https://pandera.readthedocs.io/

**BI & Communication**  
tidyverse — https://www.tidyverse.org/ · ggplot2 — https://ggplot2.tidyverse.org/  
Shiny — https://shiny.posit.co/ · Quarto — https://quarto.org/  
Power BI — https://learn.microsoft.com/power-bi/ · Tableau — https://www.tableau.com/learn/training

**Packaging, CI/CD, Containers**  
Packaging guide — https://packaging.python.org/ · PEP 517 — https://peps.python.org/pep-0517/ · PEP 518 — https://peps.python.org/pep-0518/  
`build` — https://pypi.org/project/build/ · `twine` — https://twine.readthedocs.io/ · Poetry — https://python-poetry.org/docs/  
pre-commit — https://pre-commit.com/ · black — https://black.readthedocs.io/ · ruff — https://docs.astral.sh/ruff/ · mypy — https://mypy.readthedocs.io/  
Docker best practices — https://docs.docker.com/develop/dev-best-practices/ · multi-stage — https://docs.docker.com/build/building/multi-stage/  
GitHub Actions (Python) — https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python

---

## Deliverables & submission checklist

- ✅ Reproducible **Dockerized** project (`docker compose up` runs the full stack)  
- ✅ `README.md` with **architecture diagram**, data contracts, and runbooks  
- ✅ **Automated tests** (unit + data quality) and **CI** passing  
- ✅ **Lineage & catalog** visible (Marquez/DataHub) and BI artifact published  
- ✅ Versioned **Python package** for shared utilities (built & published to internal index/artifacts)

---

## Toolkit prerequisites

- Laptop with **Docker Desktop**, **Python 3.10+**, **Git**, **R** (for BI), and **VS Code/RStudio**  
- GitHub account (Actions/Packages), and enough local RAM/CPU for Spark + containers

---

## Folder plan (this UE)

```text
01-Data-Engineering/
├─ README.md # this file
├─ Data-Preprocessing/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Business-Intelligence/
├─ Big-Data-Analytics/
└─ Packaging/
```

## ⏱ Time Budget & Strategic Schedule (70 h total)

- Goal: spend exactly 70 hours (same as contact hours) across all links, and align study blocks with your real timetable.

### A) Per-link time allocation (adds up to 70 h)
-Data Pre-processing (8.25 h)

- pandas — 2.0 h

- Polars — 1.25 h

-PyArrow — 1.0 h

- DuckDB (Python) — 1.0 h

- scikit-learn Pipelines — 1.5 h

- ColumnTransformer — 0.75 h

- pandera — 0.75 h

- BI (R) (7.0 h)

- tidyverse — 1.0 h

- ggplot2 — 1.0 h

- Shiny — 1.5 h

- Quarto — 1.25 h

- Power BI — 1.25 h

- Tableau — 1.0 h

- Big Data Analytics (23.0 h)

- Kafka — 3.5 h

- Kafka Connect — 1.25 h

- Schema Registry — 1.25 h

- Debezium — 1.75 h

- Spark Structured Streaming — 6.0 h

- Apache Beam — 2.0 h

- Delta Lake — 2.0 h

- Apache Iceberg — 2.0 h

- Apache Hudi — 2.0 h

- Apache Parquet — 1.25 h

- Packaging & CI/CD (11.5 h)

- Packaging guide — 2.0 h

- PEP 517 — 0.5 h

- PEP 518 — 0.25 h

- build — 0.5 h

- twine — 0.5 h

- Poetry — 1.25 h

- pre-commit — 0.75 h

- black — 0.25 h

- ruff — 0.25 h

- mypy — 0.5 h

- Docker multi-stage — 2.25 h

- GitHub Actions (Python) — 2.5 h

- Practice path & governance (17.75 h)

- Debezium examples — 1.0 h

- dbt intro — 2.5 h

- jaffle_shop — 1.5 h

- Great Expectations — 1.75 h

- Soda — 1.25 h

- Deequ — 1.25 h

- Airflow docs — 2.5 h

- Airflow best practices — 0.5 h

- OpenLineage — 1.75 h

- Marquez — 1.25 h

- DataHub docs — 2.0 h

- DataHub repo — 0.5 h

- Lab starters (2.5 h)

- Confluent cp-all-in-one — 2.5 h

- (Debezium examples, jaffle_shop, Airflow best practices — counted above, 0 h here)

- TOTAL = 8.25 + 7.0 + 23.0 + 11.5 + 17.75 + 2.5 = 70.0 h

## B) Timetable-aligned study plan (mirrors course days)

- Each block below sums to the session length (Full day 7 h, Half-day 3.5 h). Split long links across days if needed.

- 01 Oct — Data Pre-processing (7 h)

- pandas (1.75 h), Polars (1.0 h), PyArrow (0.75 h), DuckDB (0.75 h)

- scikit-learn Pipelines (1.25 h), ColumnTransformer (0.5 h), pandera (0.25 h)

## 02 Oct — BI (R) (7 h)

- tidyverse (1.0 h), ggplot2 (1.0 h), Shiny (1.5 h)

- Quarto (1.25 h), Power BI (1.25 h), Tableau (1.0 h)

## 03 Oct — Packaging I (7 h)

- Packaging guide (1.0 h), PEP 517 (0.5 h), PEP 518 (0.25 h)

- build (0.5 h), twine (0.5 h), Poetry (0.5 h)

- pre-commit (0.25 h), black (0.25 h), ruff (0.25 h), mypy (0.25 h)

- Docker multi-stage (1.0 h), GitHub Actions (1.0 h)

- 08 Oct — Data Pre-processing II (7 h)

- scikit-learn Pipelines (0.25 h, finish), ColumnTransformer (0.25 h, finish), pandera (0.5 h, finish)

- Parquet (1.25 h)

- Great Expectations (1.0 h), Soda (0.5 h), Deequ (0.5 h)

- dbt intro (1.0 h), jaffle_shop (0.5 h), DataHub docs (0.5 h)

## 29 Oct — Big Data I (7 h)

- Kafka (1.75 h), Kafka Connect (0.75 h), Schema Registry (0.75 h), Debezium (0.75 h)

- Spark Structured Streaming (2.0 h)

- Delta (0.75 h), cp-all-in-one (0.25 h)

## 30 Oct — Big Data II (7 h)

- Kafka (1.75 h, finish), Spark Structured Streaming (2.0 h)

- Apache Beam (1.0 h)

- Delta (0.5 h, finish), Iceberg (1.0 h), Hudi (0.75 h)

## 12 Nov — Big Data III (7 h)

- Spark Structured Streaming (2.0 h, finish), Apache Beam (1.0 h, finish)

- Iceberg (1.0 h, finish), Hudi (1.25 h, finish)

- Schema Registry (0.5 h, finish), Debezium (1.0 h, finish)

## 07 Jan (PM) — Packaging II (3.5 h)

- Packaging guide (1.0 h), Poetry (0.75 h), pre-commit (0.5 h)

## Docker multi-stage (0.75 h), GitHub Actions (0.5 h)

## 08 Jan (PM) — Packaging III (3.5 h)

- Docker multi-stage (0.5 h), GitHub Actions (0.5 h), mypy (0.25 h, finish)

- dbt intro (0.5 h), jaffle_shop (0.5 h), Great Expectations (0.5 h)

## 22 Jan — Orchestration & Lineage (7 h)

- cp-all-in-one (1.25 h), OpenLineage (0.75 h), Marquez (0.75 h)

- Airflow docs (1.5 h), Airflow best practices (0.5 h)

- DataHub docs (0.75 h), DataHub repo (0.5 h)

- Docker multi-stage (0.25 h, progress), GitHub Actions (0.25 h, progress)

## 28 Jan — Packaging IV + Governance (7 h)

- Docker multi-stage (0.75 h, finish), GitHub Actions (0.75 h, finish)

- dbt intro (1.0 h, finish), jaffle_shop (0.5 h, finish)

- Great Expectations (0.75 h, finish), Soda (0.75 h, finish), Deequ (0.75 h, finish)

- DataHub docs (0.75 h, finish)

- Total scheduled = 70 h. Adjust micro-splits as needed, but keep each link’s cumulative time within its allocation above.

## C) Strategic improvements (decided & scheduled)

- End-to-end demo day (30 Oct evening): bring up cp-all-in-one, run Debezium → Kafka → Spark → Delta, publish a tiny Shiny/Quarto view.

- Governance sprint (22 Jan): wire OpenLineage → Marquez, ingest to DataHub, attach GX validations to the DAG.

- Release day (28 Jan): publish your Python package (internal index), CI green, multi-stage Docker built, README with architecture diagram.
