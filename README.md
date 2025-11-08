# 🏗️ Building the Omnichannel Growth Engine
**Data-Driven Sales Acceleration for 康师傅 Singapore (Mock FMCG Capstone Project)**  
Prepared by **Derrick Wong** | Graduate, NTUC LearningHub Associate Data Analyst Course (Cohort 36, 2025)

---

![Project Banner](presentation/dashboard_preview.png)

## 🚀 Executive Summary
This project demonstrates a complete end-to-end **data analytics solution** for the FMCG sector — simulating how **康师傅 Singapore** can leverage omnichannel insights (Retail, Shopee, Lazada, D2C) to accelerate sales growth and customer engagement.

It integrates **data cleaning, diagnostics, predictive modeling, and dashboarding** into one coherent analytical workflow — built entirely with Python and Jupyter.

---

## 🎯 Objectives
- Build a structured **data pipeline** from raw to clean datasets  
- Analyze **channel, territory, and product performance**  
- Forecast sales using **Prophet/ARIMA** models  
- Segment customers via **K-Means clustering**  
- Predict **promotion effectiveness** using logistic regression  
- Deliver a **Power BI–style executive dashboard**  

---

## 🧩 Project Architecture
```
omnichannel-growth-engine/
├── data/                         # Mock FMCG dataset (inputs)
├── clean/                        # Cleaned datasets (outputs)
├── notebooks/                    # Analytical workflow
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_predictive_models.ipynb
│   └── 04_visualization_dashboard.ipynb
├── presentation/                 # PPTX and visuals for storytelling
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## 📊 What Each Notebook Does

| Notebook | Purpose | Output |
|-----------|----------|---------|
| **01_data_cleaning** | Cleans, standardizes, fixes data types & removes outliers | `cleaned_fmcg_omnichannel_sales.csv` |
| **02_exploratory_analysis** | KPIs, channel trends, territory leaderboard, promo uplift | 5 visual charts + summary metrics |
| **03_predictive_models** | Forecasting (Prophet/ARIMA), segmentation (K-Means), promo prediction (LogReg) | Forecast plots, cluster visuals, confusion matrix |
| **04_visualization_dashboard** | Power BI–style dashboard (Plotly/Dash) | Interactive visuals for management storytelling |

---

## ⚙️ Tech Stack & Tools
**Programming & Analytics**
- Python 3.10+  
- Pandas, NumPy, Matplotlib, Seaborn  
- Plotly, Dash (for BI-style visuals)  
- Prophet / Statsmodels (ARIMA)  
- Scikit-learn (K-Means, Logistic Regression)

**Data & Platform**
- Jupyter Notebooks  
- Anaconda Environment  
- Mock FMCG Dataset (6 months of omnichannel sales)  

---

## 📈 Key Insights
- **Retail & D2C channels** show stable revenue growth — ideal for inventory scaling.  
- **Promotions uplift average order revenue by ~18%**, but need targeting to avoid cannibalization.  
- **Repeat buyers (~35%)** show strong retention opportunity for loyalty campaigns.  
- **East & North territories** dominate volume; **West** shows higher spend per order.  
- Predictive models highlight **future sales peaks** during promo-heavy months.

---

## 💻 Dashboard Preview (Power BI–Style)
![Dashboard Preview](presentation/dashboard_overview.png)
![Territory Performance](presentation/territory_chart.png)
![Promo Uplift](presentation/promo_uplift.png)

> Interactive dashboard built with Plotly/Dash, styled in red–gold theme inspired by 康师傅 brand identity.

---

## 🧠 Learning Highlights
- Mastered **end-to-end data lifecycle**: cleaning → analysis → prediction → visualization  
- Gained practical experience in **diagnostic, predictive, and prescriptive analytics**  
- Enhanced storytelling and **business insight presentation** for executive audiences  

---

## 📦 Deliverables
| Deliverable | Format | Description |
|--------------|---------|-------------|
| Cleaned Dataset | `.csv` | Final standardized FMCG data |
| Analysis Notebooks | `.ipynb` | 4 notebooks showing full data journey |
| Executive Deck | `.pptx` | Slide deck for management presentation |
| Dashboard | `.ipynb` / Dash app | Interactive Plotly-based analytics dashboard |
| Documentation | `.md` | GitHub README (this file) |

---

## 🧾 Author
**Derrick Wong**  
Graduate, NTUC LearningHub – Associate Data Analyst Course (Cohort 36, 2025)  
📍 Singapore  
💼 [LinkedIn](www.linkedin.com/in/daddylaughing)  
✉️ [Email](mailto:huang.derrick@gmail.com)

---

> “Data tells stories — analytics turns them into strategy.” ✨  
> *— Derrick Wong*
