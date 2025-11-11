# 🏗️ Building the Omnichannel Growth Engine
**Data-Driven Sales Acceleration for 康师傅 Singapore (Mock FMCG Capstone Project)**  
Prepared by **Derrick Wong** | Graduate, NTUC LearningHub Associate Data Analyst Course (Cohort 36, 2025)

---
![Dashboard Banner](Presentation/dashboard_preview.png)

---
## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://omnichannel-growth-engine-b3mqptt73h9dbe4scq4akd.streamlit.app/p)

> Click the badge to launch the live interactive dashboard hosted on Streamlit Cloud.
---


## 🧭 Problem Statement

In Singapore's fast-moving consumer goods (FMCG) market, retail and online channels operate in silos — leading to fragmented visibility and poor forecasting accuracy.  
To sustain growth, **康师傅 Singapore** must integrate omnichannel sales data, uncover channel patterns, and predict demand fluctuations to optimize promotions and inventory.  

This project aims to bridge that gap by transforming raw retail and e-commerce data into actionable insights for better channel strategy and forecasting accuracy.

---

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
- Plotly, Streamlit  
- Prophet / Statsmodels (ARIMA)  
- Scikit-learn (K-Means, Logistic Regression)

**Data & Platform**
- Jupyter Notebooks  
- Anaconda Environment  
- Mock FMCG Dataset (6 months of omnichannel sales) 
- Streamlit for hosting my Dasboard

---

## 📈 Key Insights
- **Retail & D2C channels** show stable revenue growth — ideal for inventory scaling.  
- **Promotions uplift average order revenue by ~18%**, but need targeting to avoid cannibalization.  
- **Repeat buyers (~35%)** show strong retention opportunity for loyalty campaigns.  
- **East & North territories** dominate volume; **West** shows higher spend per order.  
- Predictive models highlight **future sales peaks** during promo-heavy months.

---

### 💻 Developer vs Cloud Environments

This project runs in two setups:

| Environment | File | Description |
|--------------|------|-------------|
| **Cloud / Recruiter View** | `requirements.txt` | Lightweight setup for Streamlit Cloud (fast deploy) |
| **Local / Full Analytics** | `requirements_dev.txt` | Full environment for Prophet, Scikit-learn, Jupyter & analysis notebooks |

To install locally with full functionality:
```bash
pip install -r requirements_dev.txt

---

## 💻 Dashboard Preview (Streamlit Power BI–Style)

![Dashboard Overview](presentation/dashboard_overview.png)
![Channel Performance](presentation/channel_chart.png)
![Customer Insights](presentation/customer_tab.png)

> Interactive **Streamlit dashboard** built with Python and Plotly — styled in red–gold theme inspired by 康师傅 brand identity.

---

## 🌐 Live Dashboard Access

Once deployed via **Streamlit Community Cloud**, the dashboard will be accessible publicly:

[![View Dashboard](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://daddylaughing-omnichannel-growth-engine.streamlit.app)

📦 **Deployment Path**
---

## 🧠 Learning Highlights
- Mastered **end-to-end data lifecycle**: cleaning → analysis → prediction → visualization  
- Gained practical experience in **diagnostic, predictive, and prescriptive analytics**  
- Enhanced storytelling and **business insight presentation** for executive audiences  

---

## 🗓️ Timeline Alignment (Capstone C36)
| Day | Task | Deliverable |
|-----|------|--------------|
| D1 | Kickoff & Scoping | Problem Statement + Dataset link |
| D2 | Data Preparation | `.info()` / clean snapshot |
| D3 | EDA | Key visuals + insights |
| D4 | Model Dev | Prophet forecast draft |
| D5 | Evaluation | Metrics summary |
| D6 | Storytelling | Report narrative |
| D7 | Final Submission | Complete ZIP + README |
| D8 | Presentation | Live 10–12 min demo |

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

## 🏁 Conclusion

This capstone project delivers a comprehensive analytical engine to simulate real-world decision-making for omnichannel growth.  
By unifying retail and e-commerce datasets and embedding predictive intelligence, **康师傅 Singapore** can make faster, smarter, and more data-driven sales decisions.

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

---
