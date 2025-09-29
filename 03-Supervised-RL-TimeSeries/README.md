# UE Supervised Learning, Reinforcement Learning & Time Series (MLSD 25–26)

**Attendance is mandatory (including all PROJECT sessions).**  
This UE builds the toolkit you’ll use in applied AI: **supervised learning** for predictive modeling and decisioning, **reinforcement learning** (RL) for sequential decision-making, and **time series** for temporal signals. You’ll design **production-grade pipelines** with leakage-safe validation, robust metrics, calibration, and business-ready reports.

---

## Learning outcomes
By the end of this UE, you will be able to:
- Ship **leakage-free supervised models** with calibrated probabilities, interpretability (global & local), and robust HPO.
- Design **RL agents** in simulated environments; select state/action/reward abstractions; conduct offline/online evaluation; apply safety constraints.
- Build **forecasting & anomaly pipelines** with rolling backtests, hierarchical reconciliation, probabilistic prediction, and exogenous drivers.
- Communicate model behavior with **well-chosen metrics**, uncertainty intervals, and risk/impact narratives.

---

## Modules (ECUEs), core skills & **strategic resources**

### 1) Supervised Learning (L.L)

**What you’ll master**
- **Tabular SOTA**: XGBoost, LightGBM, CatBoost (native categorical handling; monotonic constraints), calibrated classification/regression (Platt/Isotonic, quantile).
- **Evaluation**: stratified CV; **nested CV** for honest model selection; class imbalance (**PR-AUC**, focal loss, cost-sensitive learning); drift checks.
- **Interpretability**: **SHAP** (TreeExplainer/Kernel), permutation importance; monotonic constraints for policy compliance.
- **Production**: feature pipelines (sklearn `ColumnTransformer`), reproducible preprocessing, robust encodings (target/WOE with leakage guards).

**Do-first labs (pro level)**
- **Leakage hunt**: deliberately inject leakage (target enc., improper CV) and prove impact on PR-AUC vs. leakage-free baseline.  
- **Calibrated risk model**: train LGBM + **Isotonic**; audit **calibration curve/Brier**, **ECE**; deploy a **decision threshold card** by business costs.  
- **Monotone model**: enforce monotonicity in XGBoost/LGBM for sensitive features; verify partial dependence monotonicity.

**Strategic resources**
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html  
- XGBoost docs: https://xgboost.readthedocs.io/  
- LightGBM docs: https://lightgbm.readthedocs.io/  
- CatBoost docs: https://catboost.ai/en/docs/  
- Calibration (sklearn): https://scikit-learn.org/stable/modules/calibration.html  
- imbalanced-learn: https://imbalanced-learn.org/stable/  
- Optuna (HPO): https://optuna.org/  
- SHAP: https://shap.readthedocs.io/en/latest/  

---

### 2) Reinforcement Learning (B.B)

**What you’ll master**
- **Core RL**: MDPs, value-based (DQN/DDQN/Dueling/Distributional), policy-gradient (REINFORCE, A2C/A3C), actor-critic (PPO, SAC), entropy regularization.
- **Engineering**: environment design (**Gymnasium** API), reward shaping vs. credit assignment, curriculum learning, logging/rollouts/reproducibility.
- **Evaluation**: off-policy estimation (IPS/DR), confidence intervals, seeds & statistical testing; **safety & constraints** (Lagrangian/CPO).
- **Advanced** *(paths for your RL+NLP interest)*: offline RL (datasets, **d3rlpy**), imitation learning (DAgger, behavior cloning), RLHF/TRL for LLMs.

**Do-first labs (pro level)**
- **From scratch PPO** (CartPole → LunarLander): implement advantage normalization, GAE, clipping; sanity-check learning curves & entropy.  
- **SAC on continuous control** (Pendulum/HalfCheetah): tune target entropy, Q-function updates; show robustness to reward scaling.  
- **Offline RL**: train on a fixed dataset (D4RL-like) with **d3rlpy**; compare CQL vs. BC; evaluate with off-policy estimators.

**Strategic resources**
- Gymnasium: https://gymnasium.farama.org/  
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/  
- RLlib: https://docs.ray.io/en/latest/rllib/  
- OpenAI Spinning Up (concepts): https://spinningup.openai.com/  
- d3rlpy (offline RL): https://d3rlpy.readthedocs.io/  
- TRL (RL for LLMs): https://github.com/huggingface/trl  
- Safety RL (CPO idea): https://proceedings.mlr.press/v70/achiam17a.html  

---

### 3) Time Series (S.A, others)

**What you’ll master**
- **Classical & Probabilistic**: ETS/ARIMA/SARIMA; **state-space** (Kalman), TBATS; **Prophet** for additivity/seasonality; probabilistic intervals.
- **Modern**: **sktime**, **statsforecast**, **darts**, **GluonTS**, **neuralforecast** (TFT, N-BEATS, DeepAR), **PyTorch Forecasting**; global vs. local models.
- **Backtesting**: **rolling-origin** CV; horizon-wise metrics (**sMAPE, MAPE, MASE, RMSSE, Pinball loss**); exogenous drivers; holiday features.
- **Hierarchical & multivariate**: reconciliation (MinT/OLS), Granger causality, causality-aware modeling; change-point & anomaly detection.

**Do-first labs (pro level)**
- **Leakage-safe backtest**: implement rolling windows; compare ARIMA/ETS vs. TFT on multiple horizons; report **MASE/RMSSE** and prediction intervals.  
- **Hierarchical forecasting**: fit bottom-level models, reconcile with **MinT**; measure coherence & accuracy improvements.  
- **Anomaly & change-points**: seasonal-decompose + residual modeling; evaluate precision@K on labeled anomalies.

**Strategic resources**
- sktime: https://www.sktime.net/en/stable/  
- statsforecast (Nixtla): https://github.com/Nixtla/statsforecast  
- neuralforecast (Nixtla, TFT/N-BEATS/etc.): https://github.com/Nixtla/neuralforecast  
- darts: https://unit8co.github.io/darts/  
- GluonTS: https://ts.gluon.ai/  
- Prophet: https://facebook.github.io/prophet/  
- statsmodels tsa: https://www.statsmodels.org/stable/tsa.html  
- PyTorch Forecasting: https://pytorch-forecasting.readthedocs.io/en/stable/  
- HierarchicalForecast (Nixtla): https://github.com/Nixtla/hierarchicalforecast  

---

## Contact hours & key dates (from your calendar)
**Approx. total (UE): ~70 hours**

- **Supervised Learning:** 31/10/2024 (PM+ext), 05/11/2024 (AM+PM+ext), 06/11/2024 (AM+PM) → **≈ 21 h**  
  - **Exam (L.L):** **28/11/2024 AM**  
- **Reinforcement Learning:** 07/01/2025 (AM), 08/01/2025 (AM), 09/01/2025 (AM), 14/01/2025 (AM), 15/01/2025 (AM), 16/01/2025 (AM) → **≈ 21 h**  
  - **Exam (B.B):** **10/02/2025 PM**  
- **Time Series:** 05/12/2024 (AM+PM), 12/12/2024 (PM, TS II), 19/12/2024 (AM+PM, TS II) → **≈ 21–28 h**  
> Sessions: **09:00–12:30** (AM) · **14:00–17:30** (PM). Extended slots are noted in the October/November entries.

---

## Assessment & grading (suggested — adapt to official rubric)
- **Capstone A (Supervised)** — calibrated, interpretable model with **threshold card** and drift monitoring: **30%**  
- **Capstone B (Time Series)** — rolling backtest + probabilistic forecasts + business dashboard: **30%**  
- **RL Practical** — PPO/SAC study with proper eval protocol (seeds, confidence intervals): **20%**  
- **Exams:** Supervised (28/11 AM) **10%** · RL (10/02 PM) **10%**

---

## Hands-on practice path (to **outperform**)
1. **Baseline & leakage audit** (SL) → build a leakage-free, calibrated baseline with SHAP + threshold card.  
2. **Probabilistic forecasting** (TS) → rolling backtest with pinball loss; reconcile hierarchical series; stress-test exogenous features.  
3. **Policy learning** (RL) → PPO from scratch + SB3 baseline; add reward shaping and early stopping; report eval with CIs & ablations.  
4. **Production touches** → persist artifacts (skops/ONNX), fast inference (treelite for GBDT), and checks (model/feature drift).

---

## Lab starters & exemplar repos
- **scikit-learn examples**: https://scikit-learn.org/stable/auto_examples/  
- **XGBoost/LightGBM/CatBoost notebooks**: official repos & docs above  
- **Stable-Baselines3 zoo**: https://github.com/DLR-RM/rl-baselines3-zoo  
- **CleanRL (single-file implementations)**: https://github.com/vwxyzjn/cleanrl  
- **Nixtla templates (forecasting)**: https://nixtla.github.io/  
- **darts tutorial gallery**: https://unit8co.github.io/darts/examples/  

---

## Reference stack — quick links

**Supervised**  
scikit-learn — https://scikit-learn.org/ · XGBoost — https://xgboost.readthedocs.io/ · LightGBM — https://lightgbm.readthedocs.io/ · CatBoost — https://catboost.ai/en/docs/ · Optuna — https://optuna.org/ · SHAP — https://shap.readthedocs.io/

**Reinforcement Learning**  
Gymnasium — https://gymnasium.farama.org/ · SB3 — https://stable-baselines3.readthedocs.io/ · RLlib — https://docs.ray.io/en/latest/rllib/ · Spinning Up — https://spinningup.openai.com/ · d3rlpy — https://d3rlpy.readthedocs.io/ · TRL — https://github.com/huggingface/trl

**Time Series**  
sktime — https://www.sktime.net/ · statsforecast — https://github.com/Nixtla/statsforecast · neuralforecast — https://github.com/Nixtla/neuralforecast · darts — https://unit8co.github.io/darts/ · GluonTS — https://ts.gluon.ai/ · Prophet — https://facebook.github.io/prophet/ · statsmodels tsa — https://www.statsmodels.org/stable/tsa.html · PyTorch Forecasting — https://pytorch-forecasting.readthedocs.io/

---

## Deliverables & submission checklist
- ✅ **Supervised**: leakage-free, calibrated model; SHAP report; threshold card; cost/risk analysis.  
- ✅ **RL**: PPO/SAC experiments with seeds, eval curves, and **confidence bands**; ablation of reward shaping.  
- ✅ **Time Series**: rolling backtest; probabilistic intervals; hierarchical reconciliation; business-ready plots.  
- ✅ **Reproducibility**: config files, fixed seeds, environment lockfile; CI checks on metrics/plots.  

---

## Toolkit prerequisites
- **Python 3.10+**, Jupyter/VS Code.  
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn, Optuna, SHAP; Gymnasium, Stable-Baselines3, d3rlpy; sktime, statsforecast/neuralforecast, darts, statsmodels, Prophet, PyTorch Forecasting.

---

## Folder plan (this UE)

``` text
03-Supervised-RL-TimeSeries/
├─ README.md # this file
├─ Supervised-Learning/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Reinforcement-Learning/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
├─ Time-Series/
│ ├─ notes.md
│ ├─ resources.md
│ └─ notebooks/
└─ Evaluation-Metrics/
├─ notes.md # PR-AUC vs ROC-AUC, pinball loss, MASE/RMSSE, off-policy estimators
└─ notebooks/
```



