# Навигация

**Ноутбуки уже выполнены.** Все графики и выводы видны сразу при открытии на GitHub. 

| Что смотреть | Где |
|-------------|-----|
| **📄 Аналитический отчёт** (полная версия) | [ANALYSIS_REPORT.md](https://github.com/vokoder1990/ml_practic/blob/main/stroke-prediction/ANALYSIS_REPORT.md) |
| **📄 Executive Summary** (краткое саммари) | [EXECUTIVE_SUMMARY.md](https://github.com/vokoder1990/ml_practic/blob/main/stroke-prediction/EXECUTIVE_SUMMARY.md) |
| **📁 Полные результаты** (графики, отчёты, модели, все циклы) | [Google Drive](https://drive.google.com/drive/folders/1pfQ2ShgfPZ94OELHakzorucRvqmdv1vT?usp=sharing) |
| **Ноутбук 1: предобработка** | [preprocessing.ipynb](https://github.com/vokoder1990/ml_practic/blob/main/stroke-prediction/notebooks/preprocessing_v9_stroke.ipynb) |
| **Ноутбук 2: EDA** | [eda.ipynb](https://github.com/vokoder1990/ml_practic/blob/main/stroke-prediction/notebooks/eda_v8_stroke.ipynb) |
| **Ноутбук 3: моделирование** | [modelling.ipynb](https://github.com/vokoder1990/ml_practic/blob/main/stroke-prediction/notebooks/modeling_log_reg_v7_stroke.ipynb) |
| **Ноутбук 4: сравнение циклов и тестирование** | [cycle_comparison.ipynb](https://github.com/vokoder1990/ml_practic/blob/main/stroke-prediction/notebooks/cycle_comparison_v9_stroke.ipynb) |


# Stroke Prediction: навигация по ноутбукам

**Архитектура, логика и пайплайн** разработаны на основе курса [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course). 

**Реализация:** DeepSeek.

## Структура проекта (4 ноутбука)

Проект разбит на **4 независимых ноутбука**, которые запускаются строго последовательно.  
Каждый ноутбук читает данные предыдущего и создаёт новые артефакты.

<pre>
cycle_{N}/
├── preprocessing/       # ноутбук №1
├── eda/                 # ноутбук №2
├── modelling/           # ноутбук №3
└── cycle_comparison/    # ноутбук №4
</pre>

---

## 📋 Что делает каждый ноутбук

### 1. `preprocessing.ipynb`

**Задача:** загрузить сырые данные, очистить, разбить на train/val/test.

| Что делает | Что создаёт |
|-----------|-------------|
| загружает CSV | `checkpoints/` (контрольные точки) |
| приводит типы, чистит названия | `splits/` (X_train, X_val, X_test, y_*) |
| удаляет аномалии (age<1, bmi>60, gender=Other) | `metadata.json` (версии пакетов, сплиты) |
| разбивает на train/val/test | `preprocessing_report.txt` |

**Запускать первым.** Без него остальные ноутбуки не найдут данные.

---

### 2. `eda.ipynb`

**Задача:** анализ, создание признаков, кодирование, масштабирование, отбор признаков, Ablation Study и RF.

| Что делает | Что создаёт |
|-----------|-------------|
| загружает сплиты из preprocessing | `data/` (X_train_scaled, X_train_encoded и т.д.) |
| создаёт синтетические признаки | `transformers/` (encoders.pkl, scalers.pkl) |
| обрабатывает пропуски и выбросы | `plots/` (распределения, boxplot, корреляции) |
| кодирует категории (one-hot, binary_map) | `reports/` (eda_final_report.txt) |
| масштабирует числовые признаки | `production_metadata.json` |
| **Ablation Study**: удаляет признаки по одному/группам, смотрит на PR-AUC | `reports/ablation_study_tree.csv`, `ablation_study_linear.csv` |
| **Random Forest**: вычисляет важность признаков | `reports/feature_importance_rf.csv` |

**Запускать вторым.** Требует `splits/` и `metadata.json` из preprocessing.

---

### 3. `modelling.ipynb`

**Задача:** задать пороги метрик, обучить модели, отфильтровать, автоматически выбрать лучшую модель, проверить стабильность, сохранить.

| Что делает | Что создаёт |
|-----------|-------------|
| загружает масштабированные данные из EDA | `final_model.joblib` |
| перебирает сетку (FN, penalty, C) | `final_metadata.json` |
| находит порог через CV (без утечек) | `reports/experiments_comparison.csv` |
| **3 уровня фильтрации** (технический - качество и стабильность - клинический) | `reports/risk_stratification.csv` |
| **можно настраивать все пороги фильтров** (переменные в блоке 1.4) | `plots/` (confusion_matrix, calibration_curve) |
| **Проверка стабильности на 5 random_state** | `reports/finalist_stability.csv`, `plots/finalist_stability.png` |
| **Sensitivity Analysis**: удаление 5% данных | `reports/sensitivity_analysis.csv` |
| **Permutation Importance**: перемешивание признаков | `reports/permutation_importance.csv` |
| **Learning Curve**: хватает ли данных | `reports/learning_curve.csv`, `plots/learning_curve.png` |
| **Статистическая значимость коэффициентов** (p-values, CI) | `reports/coefficients_stats.csv`, `plots/forest_plot.png` |
| **SHAP-анализ** на валидации | `plots/shap_summary.png`, `plots/shap_importance.png` |

**Что можно настраивать в modelling (блок 1.4):**

| Уровень | Что настраиваем | Пример переменной |
|---------|----------------|-------------------|
| 1 (технический фильтр) | BSS, сходимость, Stability Gap | `FILTER_MIN_BSS`, `FILTER_REQUIRE_CONVERGED` |
| 2 (качество и стабильность) | PR-AUC, ECE, MCE, Reliability Slope, Medical Cost CV Coef | `FILTER_MIN_PR_AUC`, `FILTER_MAX_ECE` |
| 3 (клинические требования) | Recall, NNI, Selection Rate, TP, Recall Lower CI | `FILTER_MIN_RECALL_VAL`, `FILTER_MAX_NNI_VAL` |

**Запускать третьим.** Требует `data/` и `transformers/` из EDA.

---

### 4. `cycle_comparison.ipynb`

**Задача:** сравнить победителей циклов, протестировать лучшую модель только одни раз, проверить качество и стабильность, сохранить модель.

| Что делает | Что создаёт |
|-----------|-------------|
| находит все `cycle_*` с моделью (статус VALID) | `production/` (final_model.joblib, scalers.pkl) |
| выбирает лучший цикл (по Medical Cost CV + доп. метрики) | `winner_cycle.json` |
| запускает TEST (с кэшированием результатов) | `test_report.json` |
| проверяет критерии приёмки (Recall, NNI, дрейф метрик) | `production_report.json` |
| копирует модель в production | `clinical_conclusion.txt` |
| **SHAP-анализ на TEST** | `plots/shap_summary_test.png`, `plots/shap_importance_test.png` |
| **Сравнение SHAP VAL vs TEST** | `reports/shap_comparison.csv`, `plots/shap_comparison.png` |
| **PSI (дрейф признаков)** VAL vs TEST | `reports/psi_val_test.csv` |
| **Сравнение с Dummy** на TEST (4 стратегии) | `plots/dummy_comparison_test.png` |
| **PR-AUC кривая** на TEST | `plots/pr_curve_test.png` |
| **Анализ FP (ложные тревоги)**: вероятности, распределение, топ-признаки | `reports/fp_analysis.csv` |
| **Доверительные интервалы** для Recall, Precision, Specificity | вывод в консоль |
| **Сравнение калибровки** VAL vs TEST (ECE, MCE) | `plots/calibration_val_test.png` |
| **Анализ ошибок** (FN vs TP vs FP): сравнение профилей признаков | вывод в консоль |

**Запускать четвёртым.** Требует завершённые `cycle_*/modelling/`.

---

## Циклы экспериментов

Каждый новый эксперимент (изменение признаков, фильтров или сетки) - **новый номер цикла**.

Папки циклов не пересекаются. Это позволяет:
- сравнивать разные подходы;
- откатываться к старой модели;
- не бояться сломать работающий вариант.

---

## Где лежат результаты

- `cycle_N/preprocessing/splits/` - сырые сплиты;
- `cycle_N/eda/data/` - готовые данные для обучения;
- `cycle_N/eda/reports/ablation_study_*.csv` - результаты Ablation Study;
- `cycle_N/eda/reports/feature_importance_rf.csv` - важность признаков (RF);
- `cycle_N/modelling/final_model.joblib` - лучшая модель цикла;
- `cycle_N/modelling/reports/` - stability, sensitivity, permutation, learning curve;
- `cycle_N/modelling/plots/` - SHAP, forest plot, calibration;
- `production/` - финальная модель, скейлеры, метаданные;
- `production/plots/` - dummy_comparison, pr_curve, calibration, shap;
- `production/reports/` - fp_analysis, psi, shap_comparison.
