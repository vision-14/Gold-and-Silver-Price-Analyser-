import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# CONFIG


st.set_page_config(page_title="Gold vs Silver Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #FFD700; letter-spacing: -0.5px; }
    .section-tag { background: #1a1a2e; color: #FFD700; padding: 4px 12px; border-radius: 20px;
                   font-size: 0.75rem; font-weight: 600; display: inline-block; margin-bottom: 8px; }
    .stat-box { background: #0f0f23; border: 1px solid #333; border-radius: 10px;
                padding: 16px; text-align: center; }
    .stat-val { font-size: 1.4rem; font-weight: 700; color: #FFD700; }
    .stat-lbl { font-size: 0.78rem; color: #aaa; margin-top: 4px; }
    .insight-box { background: #0d2137; border-left: 4px solid #FFD700;
                   padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & WRANGLING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    # ── Gold ──
    df_gold = pd.read_csv('gold_data.csv')
    df_gold['Date'] = pd.to_datetime(df_gold['Date'])
    df_gold['Price'] = df_gold['Price'].astype(str).str.replace(',', '').astype(float)
    df_gold['Open']  = df_gold['Open'].astype(str).str.replace(',', '').astype(float)
    df_gold['High']  = df_gold['High'].astype(str).str.replace(',', '').astype(float)
    df_gold['Low']   = df_gold['Low'].astype(str).str.replace(',', '').astype(float)
    df_gold['Change %'] = df_gold['Change %'].astype(str).str.replace('%', '').astype(float)
    df_gold = df_gold.sort_values('Date').reset_index(drop=True)

    # ── Silver ──
    df_silver = pd.read_csv('silver_data.csv')
    df_silver.columns = df_silver.columns.str.strip()
    df_silver['Date'] = pd.to_datetime(df_silver['Date'])
    df_silver['Price'] = pd.to_numeric(df_silver['Price'].astype(str).str.replace(',', ''), errors='coerce')
    df_silver['Change %'] = df_silver['Change %'].astype(str).str.replace('%', '').astype(float)
    df_silver = df_silver.sort_values('Date').reset_index(drop=True)

    # ── Handle Missing Values ──
    #if 'Vol' in df_silver.columns:
       # df_silver['Vol'] = df_silver['Vol'].fillna(method='ffill')
    for col in df_gold.select_dtypes(include=np.number).columns:
        df_gold[col].fillna(df_gold[col].median(), inplace=True)
    for col in df_silver.select_dtypes(include=np.number).columns:
        df_silver[col].fillna(df_silver[col].median(), inplace=True)

    # ── Feature Engineering ──
    for df in [df_gold, df_silver]:
        df['Daily Return'] = df['Price'].pct_change() * 100
        df['MA_10']        = df['Price'].rolling(10).mean()
        df['MA_30']        = df['Price'].rolling(30).mean()
        df['Volatility']   = df['Daily Return'].rolling(10).std()
        df['Year']         = df['Date'].dt.year
        df['Month']        = df['Date'].dt.month
        df['Quarter']      = df['Date'].dt.quarter

    # Label Encoding
    def label_market(ret):
        if pd.isna(ret): return 'Neutral'
        if ret < -2: return 'Bear'
        elif ret > 2: return 'Bull'
        return 'Neutral'

    le = LabelEncoder()
    df_gold['Market_Label']   = df_gold['Daily Return'].apply(label_market)
    df_silver['Market_Label'] = df_silver['Daily Return'].apply(label_market)
    df_gold['Market_Encoded']   = le.fit_transform(df_gold['Market_Label'])
    df_silver['Market_Encoded'] = le.fit_transform(df_silver['Market_Label'])

    # Normalization
    scaler = MinMaxScaler()
    df_gold['Price_MinMax']   = scaler.fit_transform(df_gold[['Price']])
    df_silver['Price_MinMax'] = scaler.fit_transform(df_silver[['Price']])

    # ── Merged ──
    df = pd.merge(
        df_gold[['Date','Price','Daily Return','Volatility','Market_Label']],
        df_silver[['Date','Price','Daily Return','Volatility']],
        on='Date', suffixes=('_gold','_silver')
    ).sort_values('Date').set_index('Date').dropna()

    df['Gold Return']      = df['Daily Return_gold']
    df['Silver Return']    = df['Daily Return_silver']
    df['Gold Volatility']  = df['Volatility_gold']
    df['Silver Volatility'] = df['Volatility_silver']
    df['GS_Ratio']         = df['Price_gold'] / df['Price_silver']
    df['Month']            = df.index.month
    df['Year']             = df.index.year

    return df, df_gold, df_silver


df, df_gold, df_silver = load_data()

gold_ret   = df['Gold Return'].dropna()
silver_ret = df['Silver Return'].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER & KPIs
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📊 Gold vs Silver — Analytics Dashboard</div>', unsafe_allow_html=True)
st.caption("Covers: Python • NumPy • Pandas • EDA • Statistics ")
st.markdown("---")

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    ("Gold Avg Return", f"{gold_ret.mean():.4f}%"),
    ("Silver Avg Return", f"{silver_ret.mean():.4f}%"),
    ("Gold Volatility", f"{df['Gold Volatility'].mean():.4f}"),
    ("Silver Volatility", f"{df['Silver Volatility'].mean():.4f}"),
    ("Pearson r", f"{gold_ret.corr(silver_ret):.3f}"),
    ("GS Ratio", f"{df['GS_Ratio'].mean():.1f}"),
]
for col, (label, val) in zip([c1,c2,c3,c4,c5,c6], kpis):
    col.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-lbl">{label}</div></div>',
                 unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🗂 Navigation")
st.sidebar.markdown("Select an analysis module:")

sections = {
    "🏠 Overview & Dataset": "overview",
    "📈 Price Trend & MA": "price",
    "📊 Histograms & Distributions": "hist",
    "📦 Boxplots & Outliers": "box",
    "🔵 Scatter Plots": "scatter",
    "🌡 Correlation Heatmap": "corr",
    "⚡ Volatility Analysis": "vol",
    "📉 Crash Detection": "crash",
    "🔄 Normalized Comparison": "norm",
    "⚖️ Gold-Silver Ratio": "ratio",
    "🧹 Missing Values Report": "missing",
    "📐 Normalization Demo": "normalization",
    "🔢 NumPy Operations": "numpy",
    "📊 Descriptive Statistics": "desc",
    "🧪 T-Test": "ttest",
    "χ² Chi-Square Test": "chi2",
    "🔗 Pearson & Spearman": "corr_tests",
    "📈 ANOVA": "anova",
    
}

analysis = st.sidebar.radio("", list(sections.keys()), label_visibility="collapsed")
key = sections[analysis]

st.sidebar.markdown("---")
st.sidebar.info("💡 Data: Gold & Silver historical daily prices. All ML models trained on closing prices and returns.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def show_tag(label): st.markdown(f'<span class="section-tag">{label}</span>', unsafe_allow_html=True)
def insight(text): st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

if key == "overview":
    show_tag("MODULE 1 — Python Basics & Pandas")
    st.subheader("Dataset Overview")

    tab1, tab2, tab3 = st.tabs(["Gold", "Silver", "Merged"])
    with tab1:
        st.dataframe(df_gold.head(20), use_container_width=True)
        st.write("Shape:", df_gold.shape)
        st.write("dtypes:", df_gold.dtypes)
    with tab2:
        st.dataframe(df_silver.head(20), use_container_width=True)
        st.write("Shape:", df_silver.shape)
    with tab3:
        st.dataframe(df.head(20), use_container_width=True)
        st.write("Shape:", df.shape)
    insight("Pandas is used to load, clean, sort, merge, and index time-series data by Date.")


elif key == "price":
    show_tag("MODULE 1 — Pandas / Time-Series Visualization")
    st.subheader("Price Trend with Moving Averages")

    commodity = st.radio("Select", ["Gold", "Silver", "Both"], horizontal=True)
    fig, axes = plt.subplots(2 if commodity == "Both" else 1, 1,
                             figsize=(14, 8 if commodity == "Both" else 4))

    def plot_price(ax, df_, color, name):
        ax.plot(df_['Date'], df_['Price'], color=color, lw=1.2, label='Price')
        ax.plot(df_['Date'], df_['MA_10'], color='orange', lw=1, ls='--', label='MA-10')
        ax.plot(df_['Date'], df_['MA_30'], color='red',    lw=1, ls='--', label='MA-30')
        ax.set_title(f"{name} Price + Moving Averages")
        ax.set_ylabel("Price (USD)")
        ax.legend()

    if commodity == "Gold":
        plot_price(axes, df_gold, '#FFD700', 'Gold')
    elif commodity == "Silver":
        plot_price(axes, df_silver, '#C0C0C0', 'Silver')
    else:
        plot_price(axes[0], df_gold, '#FFD700', 'Gold')
        plot_price(axes[1], df_silver, '#C0C0C0', 'Silver')

    plt.tight_layout()
    st.pyplot(fig)
    insight("MA-10 reacts faster to price changes; MA-30 captures the longer-term trend. Crossovers indicate trend reversals.")


elif key == "hist":
    show_tag("MODULE 3 — EDA: Histograms & Distribution")
    st.subheader("Daily Return Distributions")

    bins = st.slider("Bins", 20, 100, 60)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, series, color, label in [
        (axes[0], gold_ret, '#FFD700', 'Gold'),
        (axes[1], silver_ret, '#C0C0C0', 'Silver')
    ]:
        ax.hist(series.dropna(), bins=bins, color=color, edgecolor='black', alpha=0.85)
        ax.axvline(series.mean(), color='red', ls='--', lw=1.5, label=f'Mean: {series.mean():.2f}')
        ax.axvline(series.median(), color='blue', ls=':', lw=1.5, label=f'Median: {series.median():.2f}')
        ax.set_title(f"{label} Daily Return Distribution")
        ax.set_xlabel("Return (%)")
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.write(f"Gold — Skewness: `{gold_ret.skew():.4f}`, Kurtosis: `{gold_ret.kurtosis():.4f}`")
    c2.write(f"Silver — Skewness: `{silver_ret.skew():.4f}`, Kurtosis: `{silver_ret.kurtosis():.4f}`")
    insight("Both distributions show leptokurtosis (fat tails) — extreme returns occur more often than a normal distribution predicts.")


elif key == "box":
    show_tag("MODULE 3 — EDA: Boxplots & Outlier Detection")
    st.subheader("Boxplots & IQR Outlier Detection")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(y=gold_ret.dropna(),   ax=axes[0], color='#FFD700')
    axes[0].set_title("Gold Daily Return")
    sns.boxplot(y=silver_ret.dropna(), ax=axes[1], color='#C0C0C0')
    axes[1].set_title("Silver Daily Return")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("IQR-Based Outlier Summary")
    method = st.selectbox("Outlier Method", ["IQR (1.5×)", "Z-Score (|z|>3)", "Percentile (95th)"])

    rows = []
    for label, s in [("Gold", gold_ret), ("Silver", silver_ret)]:
        if method == "IQR (1.5×)":
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = Q3 - Q1
            out = s[(s < Q1 - 1.5*iqr) | (s > Q3 + 1.5*iqr)]
        elif method == "Z-Score (|z|>3)":
            z = np.abs(stats.zscore(s.dropna()))
            out = s.dropna()[z > 3]
        else:
            out = s[s > s.quantile(0.95)]
        rows.append({"Commodity": label, "Outlier Count": len(out),
                     "% of Data": f"{100*len(out)/len(s):.2f}%",
                     "Max Outlier": f"{out.max():.2f}%" if len(out) else "—"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


elif key == "scatter":
    show_tag("MODULE 3 — EDA: Scatter Plots")
    st.subheader("Scatter Plots")

    option = st.radio("Plot type", ["Price vs Price", "Return vs Return", "Volatility vs Volatility"], horizontal=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    if option == "Price vs Price":
        ax.scatter(df['Price_gold'], df['Price_silver'], alpha=0.4, s=8, color='steelblue')
        ax.set_xlabel("Gold Price"); ax.set_ylabel("Silver Price")
    elif option == "Return vs Return":
        ax.scatter(df['Gold Return'], df['Silver Return'], alpha=0.4, s=8, color='coral')
        ax.set_xlabel("Gold Daily Return (%)"); ax.set_ylabel("Silver Daily Return (%)")
        # Regression line
        m, b, r, p, _ = stats.linregress(df['Gold Return'].dropna(), df['Silver Return'].dropna())
        x_line = np.linspace(df['Gold Return'].min(), df['Gold Return'].max(), 200)
        ax.plot(x_line, m*x_line + b, color='red', lw=2, label=f'y = {m:.2f}x + {b:.2f}  (r={r:.2f})')
        ax.legend()
    else:
        ax.scatter(df['Gold Volatility'], df['Silver Volatility'], alpha=0.4, s=8, color='purple')
        ax.set_xlabel("Gold Volatility"); ax.set_ylabel("Silver Volatility")

    ax.set_title(option)
    st.pyplot(fig)


elif key == "corr":
    show_tag("MODULE 3 — Correlation Heatmap")
    st.subheader("Correlation Heatmaps")

    tab1, tab2, tab3 = st.tabs(["Gold Internal", "Silver Internal", "Cross-Commodity"])
    with tab1:
        cols = [c for c in ['Price','Open','High','Low','Daily Return','Volatility','MA_10','Change %'] if c in df_gold.columns]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_gold[cols].corr(), annot=True, fmt='.2f', cmap='YlOrBr', ax=ax, linewidths=0.5)
        ax.set_title("Gold — Correlation Matrix")
        st.pyplot(fig)
    with tab2:
        cols = [c for c in ['Price','Daily Return','Volatility','MA_10','Change %'] if c in df_silver.columns]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_silver[cols].corr(), annot=True, fmt='.2f', cmap='Greys', ax=ax, linewidths=0.5)
        ax.set_title("Silver — Correlation Matrix")
        st.pyplot(fig)
    with tab3:
        cross_cols = ['Gold Return','Silver Return','Gold Volatility','Silver Volatility','GS_Ratio']
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[cross_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
        ax.set_title("Cross-Commodity Correlation")
        st.pyplot(fig)
    insight("High correlation between Gold & Silver returns confirms they respond similarly to macro events.")


elif key == "vol":
    show_tag("MODULE 3 — Time-Series Visualization")
    st.subheader("Volatility Over Time")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Gold Volatility'],   color='#FFD700', lw=1.2, label='Gold')
    ax.plot(df.index, df['Silver Volatility'], color='#888',    lw=1.2, label='Silver')
    ax.fill_between(df.index, df['Silver Volatility'], df['Gold Volatility'],
                    alpha=0.15, color='grey')
    ax.set_title("10-Day Rolling Volatility")
    ax.set_ylabel("Std of Daily Return (%)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.write("**Volatility Statistics:**")
    col1, col2 = st.columns(2)
    col1.dataframe(df['Gold Volatility'].describe().round(4).rename("Gold Volatility"))
    col2.dataframe(df['Silver Volatility'].describe().round(4).rename("Silver Volatility"))


    st.dataframe(pd.DataFrame({'Gold Std': yearly_gold, 'Silver Std': yearly_silver}).round(4),
                 use_container_width=True)


elif key == "crash":
    show_tag("MODULE 3 — EDA: Anomaly Detection")
    st.subheader("Market Crash Detection")

    threshold = st.slider("Crash threshold (%)", -5.0, -1.0, -2.0, 0.1)

    gold_crash   = df[df['Gold Return'] < threshold]
    silver_crash = df[df['Silver Return'] < threshold]

    col1, col2, col3 = st.columns(3)
    col1.metric("Gold Crashes",   len(gold_crash))
    col2.metric("Silver Crashes", len(silver_crash))
    col3.metric("Simultaneous",   len(gold_crash.index.intersection(silver_crash.index)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, series_col, crash_df, color, label in [
        (axes[0], 'Gold Return',   gold_crash,   '#FFD700', 'Gold'),
        (axes[1], 'Silver Return', silver_crash, '#888',    'Silver'),
    ]:
        ax.plot(df.index, df[series_col], color=color, lw=0.8, alpha=0.9, label=label)
        ax.scatter(crash_df.index, crash_df[series_col], color='red', s=15, zorder=5, label='Crash')
        ax.axhline(threshold, color='red', ls='--', lw=0.8)
        ax.set_title(f"{label} — Crash Days")
        ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)


elif key == "norm":
    show_tag("MODULE 3 — Comparative Visualization")
    st.subheader("Normalized Price Comparison")

    df_n = df[['Price_gold','Price_silver']].copy()
    df_n = df_n / df_n.iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_n.index, df_n['Price_gold'],   label='Gold',   color='#FFD700', lw=1.5)
    ax.plot(df_n.index, df_n['Price_silver'], label='Silver', color='#888',    lw=1.5)
    ax.axhline(1, color='black', ls=':', lw=0.8)
    ax.fill_between(df_n.index, df_n['Price_gold'], df_n['Price_silver'], alpha=0.1, color='grey')
    ax.set_title("Normalized Price (Base = 1.0 at Start)")
    ax.set_ylabel("Relative Price")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    final_gold   = df_n['Price_gold'].iloc[-1]
    final_silver = df_n['Price_silver'].iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("Gold Total Return",   f"{(final_gold-1)*100:.1f}%")
    col2.metric("Silver Total Return", f"{(final_silver-1)*100:.1f}%")


elif key == "ratio":
    show_tag("MODULE 1 — Feature Engineering")
    st.subheader("Gold-to-Silver Price Ratio")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df['GS_Ratio'], color='teal', lw=1.2, label='GS Ratio')
    ax.axhline(df['GS_Ratio'].mean(), color='orange', ls='--', lw=1.2, label=f'Mean: {df["GS_Ratio"].mean():.1f}')
    ax.fill_between(df.index,
                    df['GS_Ratio'].mean() + df['GS_Ratio'].std(),
                    df['GS_Ratio'].mean() - df['GS_Ratio'].std(),
                    alpha=0.15, color='orange', label='±1 Std')
    ax.set_title("Gold / Silver Price Ratio")
    ax.set_ylabel("Ratio")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(df['GS_Ratio'].describe().round(2).rename("GS Ratio Stats"))
    insight("A historically high ratio (e.g. >80) may signal silver is undervalued relative to gold.")


elif key == "missing":
    show_tag("MODULE 2 — Data Wrangling")
    st.subheader("Missing Value Report")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Gold — Missing Values:**")
        mv_gold = df_gold.isnull().sum().reset_index()
        mv_gold.columns = ['Column','Missing']
        mv_gold['% Missing'] = (mv_gold['Missing'] / len(df_gold) * 100).round(2)
        st.dataframe(mv_gold, use_container_width=True)
    with col2:
        st.write("**Silver — Missing Values:**")
        mv_silver = df_silver.isnull().sum().reset_index()
        mv_silver.columns = ['Column','Missing']
        mv_silver['% Missing'] = (mv_silver['Missing'] / len(df_silver) * 100).round(2)
        st.dataframe(mv_silver, use_container_width=True)

    st.write("**Strategies applied:** forward-fill for `Vol`, median imputation for all numeric columns.")
    insight("Missing values in financial data often follow patterns (e.g. weekends/holidays). Forward-fill preserves temporal continuity.")


elif key == "normalization":
    show_tag("MODULE 2 — Normalization & Encoding")
    st.subheader("Normalization Demo")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df_gold['Date'], df_gold['Price'], color='#FFD700')
    axes[0].set_title("Original Price")

    axes[1].plot(df_gold['Date'], df_gold['Price_MinMax'], color='steelblue')
    axes[1].set_title("Min-Max Normalized [0, 1]")

    z_score = (df_gold['Price'] - df_gold['Price'].mean()) / df_gold['Price'].std()
    axes[2].plot(df_gold['Date'], z_score, color='coral')
    axes[2].set_title("Z-Score Standardized")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Label Encoding — Market Regime")
    st.write(df_gold[['Date','Daily Return','Market_Label','Market_Encoded']].dropna().tail(20))
    insight("Min-Max scales to [0,1]; Z-Score centers at 0 with std=1. Both are important for ML models that are sensitive to scale.")


elif key == "numpy":
    show_tag("MODULE 1 — NumPy Operations")
    st.subheader("NumPy Array Operations on Price Data")

    gold_np = df_gold['Price'].dropna().values
    silver_np = df_silver['Price'].dropna().values

    st.write("#### Descriptive NumPy Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.json({"mean": round(float(np.mean(gold_np)),2), "std": round(float(np.std(gold_np)),2),
                 "var": round(float(np.var(gold_np)),2), "min": round(float(np.min(gold_np)),2),
                 "max": round(float(np.max(gold_np)),2), "median": round(float(np.median(gold_np)),2)})
        st.caption("Gold")
    with col2:
        st.json({"mean": round(float(np.mean(silver_np)),2), "std": round(float(np.std(silver_np)),2),
                 "var": round(float(np.var(silver_np)),2), "min": round(float(np.min(silver_np)),2),
                 "max": round(float(np.max(silver_np)),2), "median": round(float(np.median(silver_np)),2)})
        st.caption("Silver")

    st.write("#### Log Returns (NumPy)")
    log_ret = np.log(gold_np[1:] / gold_np[:-1])
    st.write(f"Cumulative Log Return: **{np.sum(log_ret):.4f}** | Mean: `{np.mean(log_ret):.6f}` | Std: `{np.std(log_ret):.6f}`")

    st.write("#### Percentile Bounds")
    for p in [5, 25, 50, 75, 95]:
        st.write(f"  Gold {p}th percentile: `{np.percentile(gold_np, p):.2f}`")

    min_len = min(len(gold_np), len(silver_np))
    gs_ratio_np = gold_np[:min_len] / silver_np[:min_len]
    st.write(f"#### Gold/Silver Ratio (NumPy division)")
    st.write(f"Mean: `{np.mean(gs_ratio_np):.2f}`, Max: `{np.max(gs_ratio_np):.2f}`, Min: `{np.min(gs_ratio_np):.2f}`")


elif key == "desc":
    show_tag("MODULE 5 — Descriptive Statistics")
    st.subheader("Summary Statistics")

    tab1, tab2 = st.tabs(["Gold", "Silver"])
    with tab1:
        st.dataframe(df_gold.describe().round(4), use_container_width=True)
    with tab2:
        st.dataframe(df_silver.describe().round(4), use_container_width=True)

    st.write("#### Return Comparison")
    comp = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Variance', 'Skewness', 'Kurtosis', 'Min', 'Max'],
        'Gold Return': [gold_ret.mean(), gold_ret.median(), gold_ret.std(), gold_ret.var(),
                        gold_ret.skew(), gold_ret.kurtosis(), gold_ret.min(), gold_ret.max()],
        'Silver Return': [silver_ret.mean(), silver_ret.median(), silver_ret.std(), silver_ret.var(),
                          silver_ret.skew(), silver_ret.kurtosis(), silver_ret.min(), silver_ret.max()],
    }).set_index('Metric').round(5)
    st.dataframe(comp, use_container_width=True)


elif key == "ttest":
    show_tag("MODULE 5 — Statistical Test: T-Test")
    st.subheader("Independent T-Test: Gold vs Silver Returns")

    st.write("**Hypothesis:**")
    st.write("- H₀: Mean daily return of Gold = Mean daily return of Silver")
    st.write("- H₁: The means differ significantly")

    t_stat, p_val = stats.ttest_ind(gold_ret, silver_ret, equal_var=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("T-Statistic", f"{t_stat:.4f}")
    col2.metric("P-Value", f"{p_val:.4f}")
    col3.metric("Result", "Reject H₀" if p_val < 0.05 else "Fail to Reject H₀")

    st.markdown(f"""
    **Interpretation:** {'The means are **statistically significantly different** (p < 0.05). Gold and silver have different average return profiles.' 
    if p_val < 0.05 else 'No significant difference in average returns between Gold and Silver.'}
    """)
    insight("Welch's T-test is used here (equal_var=False) since the two datasets may have different variances.")


elif key == "chi2":
    show_tag("MODULE 5 — Statistical Test: Chi-Square")
    st.subheader("Chi-Square Test: Market Regime Independence")

    st.write("**Are Gold and Silver market regimes (Bull/Bear/Neutral) independent of each other?**")
    st.write("- H₀: Market labels are independent")
    st.write("- H₁: Market labels are associated")

    df_chi = pd.merge(
        df_gold[['Date','Market_Label']],
        df_silver[['Date','Market_Label']],
        on='Date', suffixes=('_gold','_silver')
    ).dropna()

    contingency = pd.crosstab(df_chi['Market_Label_gold'], df_chi['Market_Label_silver'])
    st.write("**Contingency Table:**")
    st.dataframe(contingency, use_container_width=True)

    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi² Statistic", f"{chi2:.4f}")
    col2.metric("P-Value", f"{p_chi:.4f}")
    col3.metric("Degrees of Freedom", dof)

    st.markdown(f"**Result:** {'Reject H₀ — regimes are **associated**.' if p_chi < 0.05 else 'Fail to Reject H₀ — regimes appear independent.'}")


elif key == "corr_tests":
    show_tag("MODULE 5 — Correlation: Pearson & Spearman")
    st.subheader("Pearson & Spearman Correlation")

    pearson_r, pearson_p   = stats.pearsonr(gold_ret.dropna(), silver_ret.dropna())
    spearman_r, spearman_p = stats.spearmanr(gold_ret.dropna(), silver_ret.dropna())

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Pearson (Linear Correlation)")
        st.metric("r", f"{pearson_r:.4f}")
        st.metric("p-value", f"{pearson_p:.4f}")
        st.caption("Measures linear relationship; sensitive to outliers.")
    with col2:
        st.write("#### Spearman (Rank Correlation)")
        st.metric("ρ", f"{spearman_r:.4f}")
        st.metric("p-value", f"{spearman_p:.4f}")
        st.caption("Rank-based; robust to outliers and non-normal distributions.")

    strength = "Strong" if abs(pearson_r) > 0.7 else "Moderate" if abs(pearson_r) > 0.4 else "Weak"
    insight(f"{strength} positive correlation ({pearson_r:.2f}) — Gold and Silver tend to move together, reflecting shared exposure to macro and inflation factors.")


elif key == "anova":
    show_tag("MODULE 5 — ANOVA")
    st.subheader("One-Way ANOVA: Monthly Return Variation")

    target = st.radio("Commodity", ["Gold", "Silver"], horizontal=True)
    ret_col = 'Gold Return' if target == 'Gold' else 'Silver Return'

    monthly_groups = [df[df['Month'] == m][ret_col].dropna() for m in range(1, 13)]
    monthly_groups = [g for g in monthly_groups if len(g) > 1]

    f_stat, p_anova = stats.f_oneway(*monthly_groups)

    col1, col2, col3 = st.columns(3)
    col1.metric("F-Statistic", f"{f_stat:.4f}")
    col2.metric("P-Value", f"{p_anova:.4f}")
    col3.metric("Result", "Significant" if p_anova < 0.05 else "Not Significant")

    monthly_means = df.groupby('Month')[ret_col].mean()
    monthly_stds  = df.groupby('Month')[ret_col].std()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(months[:len(monthly_means)], monthly_means.values, 
           yerr=monthly_stds.values, capsize=4,
           color='#FFD700' if target=='Gold' else '#C0C0C0', edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(f"{target} — Mean Monthly Return ± Std")
    ax.set_ylabel("Mean Daily Return (%)")
    plt.tight_layout()
    st.pyplot(fig)
    insight("ANOVA tests whether the mean return differs significantly across months (seasonal effect).")
