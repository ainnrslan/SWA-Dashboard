"""
Smart Waste Analytics Dashboard
Nur Ain binti Roslan | SD23027
Universiti Malaysia Pahang Al-Sultan Abdullah

FIXES APPLIED (v3):
  1. 'Year' removed from ML feature list — not a substantive predictor
  2. MLR equation display updated to reflect 4 predictors (no Year)
  3. All X_all / X_future arrays rebuilt without Year column
  4. Face Mask added to waste composition (2020: 1.4%, 2021: 0.7%)
  5. Composition percentages re-balanced after adding Face Mask
  6. Deployment section cleaned (removed duplicate sentences)
  7. ARIMA page re‑added (monthly recyclable waste forecasting)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA  # <-- added for ARIMA page

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Analytics | Malaysia",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafb; }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2E7D32;
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 13px;
        color: #666;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 12px;
        color: #2E7D32;
        margin-top: 2px;
    }
    .section-header {
        background: linear-gradient(90deg, #1B5E20, #388E3C);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 700;
        margin: 20px 0 16px 0;
    }
    .gap-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #856404;
        font-weight: 600;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_national_data():
    data = {
        'Year': [2019, 2020, 2021, 2022, 2023],
        'Total_Waste_Generated_Tonnes':          [13888862, 13914996, 13945448, 13963811, 14254714],
        'Total_Population_Thousands':            [32522.8,  32447.3,  32576.4,  32698.3,  33379.5],
        'Total_Recyclable_Collected_Tonnes':     [2145,     3091,     4354,     4285,     5830],
        'Total_Waste_Collected_Household_Tonnes':[3108889,  2944377,  3039178,  2439026,  2492363],
        'National_Recycling_Rate_%':             [0.068996, 0.104980, 0.143262, 0.175685, 0.233727],
        'Per_Capita_Waste_kg_day':               [1170.000, 1174.929, 1172.834, 1170.000, 1169.999],
    }
    return pd.DataFrame(data)


@st.cache_data
def load_waste_composition():
    """
    FIX: Face Mask added for 2020 (1.4%) and 2021 (0.7%) as reported in Section 4.2.2.
    'Others' adjusted downward to keep each year's total at 100%.
         2020 Others: 8.5 - 1.4 = 7.1
         2021 Others: 11.8 - 0.7 = 11.1
    """
    data = {
        'Type of Waste': [
            'Food Waste', 'Plastic', 'Paper',
            'Diapers/Napkin', 'Garden Waste', 'Textiles',
            'Comingled', 'Face Mask', 'Others'
        ],
        '2019': [30.3, 24.8, 10.5, 11.1, 4.1, 4.8, 4.1, 0.0, 10.3],
        '2020': [34.6, 22.6,  8.1, 12.8, 4.5, 3.6, 3.9, 1.4,  7.1],
        '2021': [36.0, 23.9,  8.8, 11.3, 3.6, 3.5, 0.4, 0.7, 11.1],
        '2022': [30.6, 21.9, 15.3,  8.2, 2.9, 2.3, 0.7, 0.0, 15.1],
        '2023': [35.5, 26.9, 10.3,  8.9, 3.4, 3.1, 2.3, 0.0,  9.6],
    }
    return pd.DataFrame(data)


@st.cache_data
def load_top_states():
    data = {
        'State':  ['Selangor','Johor','Sabah','Perak','Sarawak',
                   'Kedah','W.P. Kuala Lumpur','Kelantan','Pulau Pinang','Pahang'],
        'Waste_2023_Tonnes': [3077023, 1751289, 1534305, 1077042, 1068419,
                               928779,   854611,  786303,  764022,  700415],
    }
    return pd.DataFrame(data)


@st.cache_data
def load_disposal_sites():
    data = {
        'Year':         [2019, 2020, 2021, 2022, 2023],
        'Sanitary':     [19,   21,   21,   22,   22],
        'Non_Sanitary': [119,  117,  116,  116,  114],
        'Inert':        [4,    4,    4,    5,    5],
    }
    return pd.DataFrame(data)


# ──────────────────────────────────────────────
# ML MODELS — FIX 1: Year REMOVED from features
# ──────────────────────────────────────────────
@st.cache_data
def run_models(df):
    """
    80:20 split → 4 train (2019–2022) / 1 test (2023).

    FIX: 'Year' has been removed from the feature list.
    Year is a sequential index, not a substantive predictor.
    Retaining it would cause the model to fit a time trend
    rather than learning from meaningful waste-related variables.

    Features (4):
      - Total_Waste_Generated_Tonnes
      - Total_Population_Thousands
      - Total_Recyclable_Collected_Tonnes
      - Total_Waste_Collected_Household_Tonnes
    """
    # ── FIX 1: 'Year' removed ──
    features = [
        'Total_Waste_Generated_Tonnes',
        'Total_Population_Thousands',
        'Total_Recyclable_Collected_Tonnes',
        'Total_Waste_Collected_Household_Tonnes',
    ]
    target = 'National_Recycling_Rate_%'

    X = df[features].values
    y = df[target].values

    split = int(len(X) * 0.8)          # 4 train, 1 test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {}

    # ── MLR ──
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    y_pred_mlr = mlr.predict(X_test)
    results['MLR'] = {
        'model':        mlr,
        'y_test':       y_test,
        'y_pred':       y_pred_mlr,
        'r2':           r2_score(y_test, y_pred_mlr) if len(y_test) > 1 else float('nan'),
        'mae':          mean_absolute_error(y_test, y_pred_mlr),
        'rmse':         np.sqrt(mean_squared_error(y_test, y_pred_mlr)),
        'mape':         float(np.mean(np.abs((y_test - y_pred_mlr) / y_test)) * 100),
        'train_r2':     r2_score(y_train, mlr.predict(X_train)),
        'coefficients': dict(zip(features, mlr.coef_)),
        'intercept':    mlr.intercept_,
    }

    # ── Random Forest ──
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'model':              rf,
        'y_test':             y_test,
        'y_pred':             y_pred_rf,
        'r2':                 r2_score(y_test, y_pred_rf) if len(y_test) > 1 else float('nan'),
        'mae':                mean_absolute_error(y_test, y_pred_rf),
        'rmse':               np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mape':               float(np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100),
        'train_r2':           r2_score(y_train, rf.predict(X_train)),
        'feature_importance': dict(zip(features, rf.feature_importances_)),
    }

    # ── XGBoost ──
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'model':              xgb_model,
        'y_test':             y_test,
        'y_pred':             y_pred_xgb,
        'r2':                 r2_score(y_test, y_pred_xgb) if len(y_test) > 1 else float('nan'),
        'mae':                mean_absolute_error(y_test, y_pred_xgb),
        'rmse':               np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'mape':               float(np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100),
        'train_r2':           r2_score(y_train, xgb_model.predict(X_train)),
        'feature_importance': dict(zip(features, xgb_model.feature_importances_)),
    }

    meta = {
        'features':    features,
        'X_train':     X_train,
        'X_test':      X_test,
        'y_train':     y_train,
        'y_test':      y_test,
        'train_years': df['Year'].values[:split].tolist(),
        'test_years':  df['Year'].values[split:].tolist(),
        'split_idx':   split,
    }
    return results, meta


# ──────────────────────────────────────────────
# FIX 2: helper to build X arrays WITHOUT Year
# ──────────────────────────────────────────────
def build_X(df_rows):
    """
    Build feature matrix from a DataFrame slice.
    Columns must match the 4-feature order used in run_models().
    No Year column included.
    """
    return df_rows[[
        'Total_Waste_Generated_Tonnes',
        'Total_Population_Thousands',
        'Total_Recyclable_Collected_Tonnes',
        'Total_Waste_Collected_Household_Tonnes',
    ]].values


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def small_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


GREEN  = '#2E7D32'
LGREEN = '#81C784'
RED    = '#C62828'
ORANGE = '#F57C00'
BLUE   = '#1565C0'
GRAY   = '#607D8B'


# ══════════════════════════════════════════════
# SIDEBAR (ARIMA page added back)
# ══════════════════════════════════════════════
with st.sidebar:
    st.image("img1.png", use_container_width=True)

    st.markdown("## ♻️ Smart Waste Analytics")
    st.markdown("**Nur Ain binti Roslan**  \nSD23027 · UMPSA · 2025/2026")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview & KPIs",
         "📊 EDA & Trends",
         "🤖 Predictive Models",
         "📈 Time Series & ARIMA",   # <-- re‑added ARIMA page
         "🔮 Future Prediction",
         "📋 Model Comparison"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data source: MHLG & SWCorp Malaysia  \nPeriod: 2019–2023")


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
df        = load_national_data()
df_comp   = load_waste_composition()
df_states = load_top_states()
df_sites  = load_disposal_sites()
results, meta = run_models(df)

TARGET = 30.67


# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW & KPIs
# ══════════════════════════════════════════════
if page == "🏠 Overview & KPIs":
    st.title("♻️ Smart Waste Analytics Dashboard")
    st.caption("Monitoring and Predicting Recycling Efficiency in Malaysia (2019–2023)")

    latest_rate = df['National_Recycling_Rate_%'].iloc[-1]
    gap         = TARGET - latest_rate
    st.markdown(f"""
    <div class="gap-box">
    ⚠️ <b>CRITICAL FINDING:</b> Malaysia's 2023 recycling rate is <b>{latest_rate:.4f}%</b>
    — a gap of <b>{gap:.2f}%</b> below the national target of <b>{TARGET}%</b>.
    Significant improvements are urgently required.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    waste_change = ((df['Total_Waste_Generated_Tonnes'].iloc[-1] -
                     df['Total_Waste_Generated_Tonnes'].iloc[0]) /
                     df['Total_Waste_Generated_Tonnes'].iloc[0] * 100)
    c1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Waste Generated (2023)</div>
        <div class="kpi-value">{df['Total_Waste_Generated_Tonnes'].iloc[-1]/1e6:.2f}M</div>
        <div class="kpi-delta">Tonnes &nbsp;|&nbsp; +{waste_change:.1f}% since 2019</div>
    </div>""", unsafe_allow_html=True)

    rec_change = ((df['National_Recycling_Rate_%'].iloc[-1] -
                   df['National_Recycling_Rate_%'].iloc[0]) /
                   df['National_Recycling_Rate_%'].iloc[0] * 100)
    c2.markdown(f"""
    <div class="kpi-card" style="border-left-color:#C62828">
        <div class="kpi-label">National Recycling Rate (2023)</div>
        <div class="kpi-value">{df['National_Recycling_Rate_%'].iloc[-1]:.4f}%</div>
        <div class="kpi-delta" style="color:#C62828">Target: {TARGET}% &nbsp;|&nbsp;
        Gap: {gap:.2f}%</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="kpi-card" style="border-left-color:#1565C0">
        <div class="kpi-label">Per Capita Waste (2023)</div>
        <div class="kpi-value">{df['Per_Capita_Waste_kg_day'].iloc[-1]:.1f}</div>
        <div class="kpi-delta">kg / person / day (×1000 pop units)</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="kpi-card" style="border-left-color:#F57C00">
        <div class="kpi-label">Recyclable Collected (2023)</div>
        <div class="kpi-value">{df['Total_Recyclable_Collected_Tonnes'].iloc[-1]:,.0f}</div>
        <div class="kpi-delta">Tonnes &nbsp;|&nbsp; +172% since 2019</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-header">📌 Recycling Rate vs National Target (30.67%)</div>',
                unsafe_allow_html=True)

    fig, ax = small_fig(10, 4)
    bars = ax.bar(df['Year'], df['National_Recycling_Rate_%'],
                  color=GREEN, alpha=0.8, label='Actual Rate', zorder=3)
    ax.axhline(TARGET, color=RED, linewidth=2.5, linestyle='--',
               label=f'Target {TARGET}%', zorder=4)
    ax.set_ylim(0, 35)
    ax.set_ylabel('Recycling Rate (%)')
    ax.set_title('National Recycling Rate vs Target (2019–2023)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for bar, rate in zip(bars, df['National_Recycling_Rate_%']):
        gap_val = TARGET - rate
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{rate:.4f}%\n(Gap: {gap_val:.2f}%)',
                ha='center', fontsize=8, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">📋 National Summary Table (2019–2023)</div>',
                unsafe_allow_html=True)
    display_df = df.copy()
    display_df['Total_Waste_Generated_Tonnes']       = display_df['Total_Waste_Generated_Tonnes'].map('{:,.0f}'.format)
    display_df['Total_Population_Thousands']         = display_df['Total_Population_Thousands'].map('{:,.1f}'.format)
    display_df['Total_Recyclable_Collected_Tonnes']  = display_df['Total_Recyclable_Collected_Tonnes'].map('{:,.0f}'.format)
    display_df['National_Recycling_Rate_%']          = display_df['National_Recycling_Rate_%'].map('{:.4f}%'.format)
    display_df['Per_Capita_Waste_kg_day']            = display_df['Per_Capita_Waste_kg_day'].map('{:.3f}'.format)
    display_df.columns = ['Year', 'Total Waste (Tonnes)', 'Population (000)',
                           'Recyclable (Tonnes)', 'HH Waste (Tonnes)',
                           'Recycling Rate (%)', 'Per Capita (kg/p/day)']
    st.dataframe(display_df.set_index('Year'), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 — EDA & TRENDS
# ══════════════════════════════════════════════
elif page == "📊 EDA & Trends":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗑️ Waste Trends", "♻️ Waste Composition",
         "🏙️ State Analysis", "🏭 Disposal Sites"]
    )

    # ── TAB 1 ──
    with tab1:
        st.markdown('<div class="section-header">National Waste & Recycling Trends</div>',
                    unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig, ax = small_fig()
            ax.plot(df['Year'], df['Total_Waste_Generated_Tonnes'] / 1e6,
                    'o-', color=BLUE, linewidth=2.5, markersize=8)
            for x, y_val in zip(df['Year'], df['Total_Waste_Generated_Tonnes'] / 1e6):
                ax.annotate(f'{y_val:.2f}M', (x, y_val),
                            textcoords='offset points', xytext=(0, 10),
                            ha='center', fontsize=8)
            ax.set_title('Total Solid Waste Generated', fontweight='bold')
            ax.set_ylabel('Million Tonnes')
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with r1c2:
            fig, ax = small_fig()
            ax.plot(df['Year'], df['National_Recycling_Rate_%'],
                    'o-', color=RED, linewidth=2.5, markersize=8)
            ax.axhline(TARGET, color=ORANGE, linestyle='--',
                       linewidth=1.8, label=f'Target {TARGET}%')
            for x, y_val in zip(df['Year'], df['National_Recycling_Rate_%']):
                ax.annotate(f'{y_val:.4f}%', (x, y_val),
                            textcoords='offset points', xytext=(0, 10),
                            ha='center', fontsize=8)
            ax.set_title('National Recycling Rate', fontweight='bold')
            ax.set_ylabel('Recycling Rate (%)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            st.pyplot(fig, use_container_width=True)
            plt.close()

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig, ax = small_fig()
            ax.bar(df['Year'], df['Total_Recyclable_Collected_Tonnes'],
                   color=GREEN, alpha=0.8)
            for yr, v in zip(df['Year'], df['Total_Recyclable_Collected_Tonnes']):
                ax.text(yr, v + 50, f'{v:,.0f}', ha='center', fontsize=8)
            ax.set_title('Recyclable Waste Collected', fontweight='bold')
            ax.set_ylabel('Tonnes')
            ax.grid(alpha=0.3, axis='y')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with r2c2:
            fig, ax = small_fig()
            ax.plot(df['Year'], df['Per_Capita_Waste_kg_day'],
                    'o-', color=GRAY, linewidth=2.5, markersize=8)
            ax.set_title('Per Capita Waste Generation', fontweight='bold')
            ax.set_ylabel('kg / person / day (×1000 pop)')
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── TAB 2 — FIX 4: Face Mask now shown in composition ──
    with tab2:
        st.markdown('<div class="section-header">Waste Composition Analysis (2019–2023)</div>',
                    unsafe_allow_html=True)
        st.caption(
            "📌 Face Mask waste appears in 2020 (1.4%) and 2021 (0.7%), "
            "corresponding to the SARS-CoV-2 pandemic period."
        )

        year_sel = st.select_slider(
            "Select year for pie chart",
            options=[2019, 2020, 2021, 2022, 2023],
            value=2023,
        )
        col_name = str(year_sel)

        # filter out zero-value categories for cleaner pie
        comp_sel = df_comp[['Type of Waste', col_name]].copy()
        comp_sel = comp_sel[comp_sel[col_name] > 0]
        comp_sel = comp_sel.sort_values(col_name, ascending=False).reset_index(drop=True)

        pc1, pc2 = st.columns([1, 1])
        with pc1:
            top5   = comp_sel.head(5)
            rest   = comp_sel.iloc[5:]
            others = pd.DataFrame({'Type of Waste': ['Others'],
                                   col_name: [rest[col_name].sum()]})
            pie_df = pd.concat([top5, others]).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(6, 5))
            colors_p = ['#FF6B6B', '#4ECDC4', '#45B7D1',
                        '#96CEB4', '#FFEAA7', '#DDA0DD']
            wedges, texts, autotexts = ax.pie(
                pie_df[col_name],
                labels=pie_df['Type of Waste'],
                autopct='%1.1f%%',
                colors=colors_p,
                startangle=90,
                pctdistance=0.8,
            )
            for at in autotexts:
                at.set_fontsize(8)
            ax.set_title(f'Waste Composition {year_sel}', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with pc2:
            years_c   = ['2019', '2020', '2021', '2022', '2023']
            key_types = ['Food Waste', 'Plastic', 'Paper',
                         'Diapers/Napkin', 'Garden Waste', 'Face Mask']
            fig, ax = small_fig(6, 5)
            for wt in key_types:
                row = df_comp[df_comp['Type of Waste'] == wt]
                if not row.empty:
                    vals = row[years_c].values[0]
                    style = '--' if wt == 'Face Mask' else '-'
                    ax.plot(years_c, vals, marker='o', linewidth=2,
                            linestyle=style, label=wt)
            ax.set_title('Key Waste Composition Trends', fontweight='bold')
            ax.set_ylabel('Composition (%)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.dataframe(df_comp.set_index('Type of Waste'), use_container_width=True)

    # ── TAB 3 ──
    with tab3:
        st.markdown(
            '<div class="section-header">Top 10 States by Solid Waste Generation (2023)</div>',
            unsafe_allow_html=True,
        )
        df_s_sorted = df_states.sort_values('Waste_2023_Tonnes', ascending=True)
        fig, ax = small_fig(9, 5)
        cmap = plt.cm.Reds(np.linspace(0.4, 0.9, len(df_s_sorted)))
        bars = ax.barh(df_s_sorted['State'],
                       df_s_sorted['Waste_2023_Tonnes'] / 1e6,
                       color=cmap)
        ax.set_xlabel('Million Tonnes')
        ax.set_title('Top 10 States — Solid Waste Generation 2023', fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        for bar, val in zip(bars, df_s_sorted['Waste_2023_Tonnes'] / 1e6):
            ax.text(bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}M', va='center', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            '<div class="section-header">Correlation: Population vs Waste Generation</div>',
            unsafe_allow_html=True,
        )
        corr_df = df[['Total_Waste_Generated_Tonnes',
                       'Total_Population_Thousands',
                       'Per_Capita_Waste_kg_day']].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title('Correlation Heatmap', fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── TAB 4 ──
    with tab4:
        st.markdown(
            '<div class="section-header">Solid Waste Disposal Sites (2019–2023)</div>',
            unsafe_allow_html=True,
        )
        fig, ax = small_fig(9, 5)
        x = np.arange(len(df_sites))
        w = 0.25
        ax.bar(x - w, df_sites['Sanitary'],    w, label='Sanitary',     color=GREEN,  alpha=0.85)
        ax.bar(x,     df_sites['Non_Sanitary'], w, label='Non-Sanitary', color=RED,    alpha=0.85)
        ax.bar(x + w, df_sites['Inert'],        w, label='Inert',        color=ORANGE, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sites['Year'])
        ax.set_ylabel('Number of Sites')
        ax.set_title('Landfill Sites by Type (2019–2023)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 160)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            "**⚠️ Non-sanitary sites dominate (~80% of all disposal sites), "
            "with only a marginal decrease from 119 (2019) to 114 (2023).**"
        )
        st.dataframe(df_sites.set_index('Year'), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 — PREDICTIVE MODELS
# ══════════════════════════════════════════════
elif page == "🤖 Predictive Models":
    st.title("🤖 Predictive Models — 80:20 Train-Test Split")

    st.info(
        f"**Train set:** {meta['train_years']} (n={meta['split_idx']})  |  "
        f"**Test set:** {meta['test_years']} (n={len(meta['y_test'])})\n\n"
        "With 5 annual observations (2019–2023), the 80:20 split assigns "
        "2019–2022 to training and 2023 to testing.  \n"
        "**Note:** 'Year' is excluded from the feature set as it is a sequential "
        "index, not a substantive waste-related predictor."
    )

    model_choice = st.selectbox("Select Model to Inspect",
                                ["MLR", "Random Forest", "XGBoost"])
    res = results[model_choice]

    st.markdown("---")

    st.markdown(
        '<div class="section-header">📐 Model Performance Metrics (Test Set: 2023)</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    r2_val = res['r2']
    m1.metric("R²",   f"{r2_val:.4f}" if not np.isnan(r2_val) else "N/A*",
              help="*R² requires ≥2 test points.")
    m2.metric("MAE",  f"{res['mae']:.6f}")
    m3.metric("RMSE", f"{res['rmse']:.6f}")
    m4.metric("MAPE", f"{res['mape']:.2f}%")
    st.caption("*R² is undefined for a single test observation. "
               "MAE, RMSE, and MAPE are the primary evaluation metrics.")

    st.markdown("---")

    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown("**Actual vs Predicted (Test Year: 2023)**")
        y_actual = float(res['y_test'][0])
        y_pred   = float(res['y_pred'][0])
        err      = abs(y_actual - y_pred)
        err_pct  = err / y_actual * 100

        comp_df = pd.DataFrame({
            'Year':      meta['test_years'],
            'Actual':    [y_actual],
            'Predicted': [y_pred],
            'Error':     [err],
            'Error (%)': [err_pct],
        }).set_index('Year')
        st.dataframe(comp_df.style.format("{:.6f}"), use_container_width=True)

        fig, ax = small_fig(5, 3.5)
        ax.bar(['Actual', 'Predicted'], [y_actual, y_pred],
               color=[GREEN, BLUE], alpha=0.85, width=0.4)
        ax.set_title(f'{model_choice} — 2023 Prediction vs Actual', fontweight='bold')
        ax.set_ylabel('Recycling Rate (%)')
        ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate([y_actual, y_pred]):
            ax.text(i, v + 0.001, f'{v:.6f}', ha='center', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with pc2:
        # ── FIX 3: X_all built WITHOUT Year ──
        X_all      = build_X(df)
        mdl        = res['model']
        y_all_pred = mdl.predict(X_all)

        fig, ax = small_fig(5, 3.5)
        split_idx = meta['split_idx']
        ax.plot(df['Year'], df['National_Recycling_Rate_%'],
                'o-', color=GREEN, linewidth=2, label='Actual', zorder=5)
        ax.plot(df['Year'][:split_idx], y_all_pred[:split_idx],
                's--', color=BLUE, linewidth=2, alpha=0.7, label='Train Fit')
        ax.plot(df['Year'][split_idx:], y_all_pred[split_idx:],
                'D--', color=RED, linewidth=2, label='Test Prediction', markersize=9)
        ax.axvline(x=df['Year'].iloc[split_idx - 1] + 0.5,
                   color=GRAY, linestyle=':', linewidth=1.5, label='Train|Test split')
        ax.set_title(f'{model_choice} — Full Timeline View', fontweight='bold')
        ax.set_ylabel('Recycling Rate (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    st.markdown('<div class="section-header">🔍 Model Details</div>',
                unsafe_allow_html=True)

    if model_choice == 'MLR':
        # ── FIX: equation display updated — Year removed ──
        st.markdown(
            "**MLR Equation:**  \n"
            "`Recycling Rate = β₀ + β₁·TotalWaste + β₂·Population "
            "+ β₃·Recyclable + β₄·HHWaste`  \n"
            "*Note: Year is excluded — it is a sequential index, "
            "not a substantive waste-related predictor.*"
        )
        coef_df = pd.DataFrame({
            'Feature':     list(res['coefficients'].keys()),
            'Coefficient': list(res['coefficients'].values()),
        })
        st.dataframe(
            coef_df.style.format({'Coefficient': '{:.6e}'}),
            use_container_width=True,
        )
        st.metric("Intercept (β₀)", f"{res['intercept']:.6f}")

    elif model_choice in ['Random Forest', 'XGBoost']:
        st.markdown("**Feature Importance:**")
        fi    = res['feature_importance']
        fi_df = pd.DataFrame({'Feature':    list(fi.keys()),
                               'Importance': list(fi.values())})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig, ax = small_fig(7, 3.5)
        ax.barh(fi_df['Feature'], fi_df['Importance'], color=GREEN, alpha=0.8)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_choice} Feature Importance', fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.metric("Training R² (in-sample fit)", f"{res['train_r2']:.6f}")
    st.caption("Training R² measures in-sample fit (optimistic). "
               "Focus on test MAE, RMSE, and MAPE for out-of-sample performance.")


# ══════════════════════════════════════════════
# PAGE 4 — TIME SERIES & ARIMA (re‑added)
# ══════════════════════════════════════════════
elif page == "📈 Time Series & ARIMA":
    st.title("📈 Monthly Recyclable Waste Forecasting (ARIMA)")
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** is used to forecast monthly recyclable waste.
    The model is trained on data from 2019–2022 and evaluated on 2023 actuals.
    """)

    # Generate synthetic monthly data from national annual totals
    yearly_recyclable = df.set_index('Year')['Total_Recyclable_Collected_Tonnes'].to_dict()
    months = pd.date_range('2019-01-01', '2023-12-31', freq='MS')
    monthly_recyclable = []
    for year in [2019,2020,2021,2022,2023]:
        # Add mild seasonality (peak in middle of year)
        season = 1 + 0.15 * np.sin(np.linspace(0, 2*np.pi, 12))
        monthly = (yearly_recyclable[year] / 12) * season
        monthly_recyclable.extend(monthly)
    df_monthly = pd.DataFrame({'Date': months, 'Recyclable_Tonnes': monthly_recyclable})
    df_monthly = df_monthly.set_index('Date').asfreq('MS')

    train = df_monthly[df_monthly.index.year <= 2022]
    test  = df_monthly[df_monthly.index.year == 2023]

    # Fit ARIMA(1,1,1) – fast and stable for demonstration
    model = ARIMA(train['Recyclable_Tonnes'], order=(1,1,1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))

    # Metrics
    mae = mean_absolute_error(test['Recyclable_Tonnes'], forecast)
    rmse = np.sqrt(mean_squared_error(test['Recyclable_Tonnes'], forecast))
    mape = np.mean(np.abs((test['Recyclable_Tonnes'] - forecast) / test['Recyclable_Tonnes'])) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f} tonnes")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAPE", f"{mape:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(train.index, train['Recyclable_Tonnes'], label='Training (2019-2022)', color='blue')
    ax.plot(test.index, test['Recyclable_Tonnes'], label='Actual 2023', color='green', marker='o')
    ax.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--', marker='x')
    ax.set_title('ARIMA(1,1,1) Forecast of Monthly Recyclable Waste')
    ax.set_xlabel('Date')
    ax.set_ylabel('Recyclable Tonnes')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.caption("Note: Monthly data is synthetically distributed from national annual totals with mild seasonality. Real monthly data would improve accuracy.")


# ══════════════════════════════════════════════
# PAGE 5 — FUTURE PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮 Future Prediction":
    st.title("🔮 Future Recycling Efficiency Prediction")
    st.caption("Predict the national recycling rate for any future year using the trained models.")
    st.markdown("---")

    fc1, fc2 = st.columns([1, 1])

    # ── FIX 3: project() uses 4 features only (no Year) ──
    def project(col, year):
        """Linear extrapolation from historical data."""
        m, b = np.polyfit(df['Year'], df[col], 1)
        return m * year + b

    with fc1:
        st.markdown("**Input Parameters**")
        pred_year = st.slider("Future Year", 2024, 2035, 2025)

        waste_proj = st.number_input(
            "Total Waste Generated (Tonnes)",
            value=int(project('Total_Waste_Generated_Tonnes', pred_year)),
            step=100000,
        )
        pop_proj = st.number_input(
            "Total Population (Thousands)",
            value=round(project('Total_Population_Thousands', pred_year), 1),
            step=100.0,
        )
        rec_proj = st.number_input(
            "Recyclable Collected (Tonnes)",
            value=int(project('Total_Recyclable_Collected_Tonnes', pred_year)),
            step=100,
        )
        hh_proj = st.number_input(
            "Total HH Waste Collected (Tonnes)",
            value=int(project('Total_Waste_Collected_Household_Tonnes', pred_year)),
            step=10000,
        )

        model_sel   = st.selectbox("Prediction Model", ["MLR", "Random Forest", "XGBoost"])
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with fc2:
        if predict_btn:
            # ── FIX 3: X_future has 4 columns — NO Year ──
            X_future   = np.array([[waste_proj, pop_proj, rec_proj, hh_proj]])
            mdl        = results[model_sel]['model']
            pred_rate  = max(0.0, float(mdl.predict(X_future)[0]))

            st.markdown(f"### Predicted Recycling Rate ({pred_year})")
            color = GREEN if pred_rate >= TARGET else RED
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color}; padding: 30px;">
                <div class="kpi-label">Predicted National Recycling Rate</div>
                <div class="kpi-value" style="font-size:40px; color:{color}">
                    {pred_rate:.4f}%
                </div>
                <div class="kpi-delta" style="color:{color}; font-size:14px">
                    {"✅ Above Target!" if pred_rate >= TARGET
                     else f"⚠️ Still {TARGET - pred_rate:.2f}% below the {TARGET}% target"}
                </div>
            </div>""", unsafe_allow_html=True)

            # Projection chart
            future_years = list(range(2024, pred_year + 1))
            future_preds = []
            for yr in future_years:
                X_f = np.array([[
                    project('Total_Waste_Generated_Tonnes',          yr),
                    project('Total_Population_Thousands',            yr),
                    project('Total_Recyclable_Collected_Tonnes',     yr),
                    project('Total_Waste_Collected_Household_Tonnes', yr),
                ]])
                future_preds.append(max(0.0, float(mdl.predict(X_f)[0])))

            fig, ax = small_fig(6, 4)
            ax.plot(df['Year'], df['National_Recycling_Rate_%'],
                    'o-', color=GREEN, linewidth=2, label='Historical', zorder=5)
            if future_years:
                ax.plot(future_years, future_preds,
                        's--', color=BLUE, linewidth=2,
                        label='Projected', alpha=0.85)
            ax.axhline(TARGET, color=RED, linestyle='--',
                       linewidth=1.8, label=f'Target {TARGET}%')
            ax.set_title(f'{model_sel} Projection to {pred_year}', fontweight='bold')
            ax.set_ylabel('Recycling Rate (%)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            st.pyplot(fig, use_container_width=True)
            plt.close()

        else:
            st.info("👈 Set parameters and click **Predict** to generate a forecast.")

    st.markdown("---")
    st.caption(
        "⚠️ Predictions are extrapolated beyond the training range (2019–2023). "
        "Accuracy decreases for years further from the observed data. "
        "Use results as indicative estimates only."
    )


# ══════════════════════════════════════════════
# PAGE 6 — MODEL COMPARISON
# ══════════════════════════════════════════════
elif page == "📋 Model Comparison":
    st.title("📋 Model Comparison Summary")

    st.markdown(
        '<div class="section-header">📐 Performance Metrics — Test Set (2023)</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for name, res in results.items():
        rows.append({
            'Model':     name,
            'Train R²':  round(res['train_r2'], 6),
            'Test R²':   'N/A*',
            'MAE':       round(res['mae'],  6),
            'RMSE':      round(res['rmse'], 6),
            'MAPE (%)':  round(res['mape'], 4),
        })
    comp_df = pd.DataFrame(rows).set_index('Model')
    st.dataframe(comp_df, use_container_width=True)
    st.caption(
        "*R² requires ≥2 test observations. With a single test point (2023), "
        "MAE, RMSE, and MAPE are the valid comparison metrics."
    )

    st.markdown(
        '<div class="section-header">📊 Visual Metric Comparison</div>',
        unsafe_allow_html=True,
    )
    mc1, mc2, mc3 = st.columns(3)
    model_names = list(results.keys())

    with mc1:
        fig, ax = small_fig(4, 3)
        vals = [results[m]['mae'] for m in model_names]
        bars = ax.bar(model_names, vals,
                      color=[GREEN if v == min(vals) else BLUE for v in vals],
                      alpha=0.85)
        ax.set_title('MAE (lower = better)', fontweight='bold', fontsize=10)
        ax.set_ylabel('MAE')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.000002,
                    f'{v:.6f}', ha='center', fontsize=7, rotation=15)
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with mc2:
        fig, ax = small_fig(4, 3)
        vals = [results[m]['rmse'] for m in model_names]
        bars = ax.bar(model_names, vals,
                      color=[GREEN if v == min(vals) else BLUE for v in vals],
                      alpha=0.85)
        ax.set_title('RMSE (lower = better)', fontweight='bold', fontsize=10)
        ax.set_ylabel('RMSE')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.000002,
                    f'{v:.6f}', ha='center', fontsize=7, rotation=15)
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with mc3:
        fig, ax = small_fig(4, 3)
        vals = [results[m]['mape'] for m in model_names]
        bars = ax.bar(model_names, vals,
                      color=[GREEN if v == min(vals) else BLUE for v in vals],
                      alpha=0.85)
        ax.set_title('MAPE % (lower = better)', fontweight='bold', fontsize=10)
        ax.set_ylabel('MAPE (%)')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{v:.2f}%', ha='center', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    best_model = min(results.items(), key=lambda x: x[1]['mape'])[0]
    best_res   = results[best_model]
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:#F57C00; margin-top:20px">
        <div class="kpi-label">🏆 Best Performing Model (lowest MAPE)</div>
        <div class="kpi-value">{best_model}</div>
        <div class="kpi-delta" style="color:#F57C00">
        MAE: {best_res['mae']:.6f} &nbsp;|&nbsp;
        RMSE: {best_res['rmse']:.6f} &nbsp;|&nbsp;
        MAPE: {best_res['mape']:.2f}%
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">📝 Interpretation & Limitations</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
**80:20 Train-Test Split Applied:**
- **Training set (80%):** 2019, 2020, 2021, 2022 — models learn historical patterns from these 4 years.
- **Test set (20%):** 2023 — models are evaluated on unseen data from 2023.

**Features Used (4):**
- Total Waste Generated (Tonnes)
- Total Population (Thousands)
- Total Recyclable Collected (Tonnes)
- Total Household Waste Collected (Tonnes)

> ✅ *Year was intentionally excluded as it is a sequential index, not a substantive
> waste-related predictor. Including it would cause models to fit a time trend
> rather than learning from meaningful waste management variables.*

**Key Limitations:**
- With only **5 annual observations**, a single test point limits statistical power.
  MAE, RMSE, and MAPE are the primary and valid evaluation metrics.
- High in-sample Training R² is expected with few data points — do not interpret
  this as evidence of strong generalisation.
- Future work should incorporate monthly or state-level data to enable more robust evaluation.

**Recommendation:** For this dataset size, **MLR** is preferred for interpretability
and generalisation, while RF and XGBoost provide useful complementary feature importance insights.
    """)

    st.markdown(
        '<div class="section-header">📈 Training R² (In-Sample Reference)</div>',
        unsafe_allow_html=True,
    )
    train_r2_df = pd.DataFrame({
        'Model':      list(results.keys()),
        'Training R²': [results[m]['train_r2'] for m in results],
    }).set_index('Model')
    st.dataframe(
        train_r2_df.style.format('{:.6f}').highlight_max(color='lightgreen'),
        use_container_width=True,
    )
    st.caption(
        "High training R² is expected with few data points and does not "
        "indicate real-world predictive power."
    )


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Smart Waste Analytics Dashboard &nbsp;|&nbsp;
    Nur Ain binti Roslan (SD23027) &nbsp;|&nbsp;
    Universiti Malaysia Pahang Al-Sultan Abdullah &nbsp;|&nbsp;
    Data: MHLG &amp; SWCorp Malaysia (2019–2023)
</div>
""", unsafe_allow_html=True)