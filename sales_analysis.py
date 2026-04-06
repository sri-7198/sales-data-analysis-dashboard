# =============================================================================
# SALES DATA ANALYSIS DASHBOARD
# Author: Data Analyst | Python + Pandas + Matplotlib + Seaborn
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── PLOT STYLE ────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "figure.titlesize": 16,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})
ACCENT   = "#58a6ff"
PALETTE  = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff","#56d364","#ff7b72"]

# =============================================================================
# STEP 1 — GENERATE REALISTIC DATASET  (10 000+ records)
# =============================================================================
np.random.seed(42)
N = 12_000

REGIONS    = ["North", "South", "East", "West", "Central"]
CATEGORIES = {
    "Electronics":   ["Laptop","Smartphone","Tablet","Monitor","Headphones","Webcam","Keyboard","Mouse"],
    "Furniture":     ["Office Chair","Standing Desk","Bookshelf","Filing Cabinet","Sofa","Lamp","Whiteboard"],
    "Clothing":      ["Jacket","Jeans","T-Shirt","Sneakers","Dress Shirt","Blazer","Boots"],
    "Office Supplies":["Pen Set","Notebook","Stapler","Printer Paper","Binder","Highlighters","Tape"],
    "Food & Bev":    ["Coffee Beans","Energy Drink","Protein Bar","Green Tea","Sparkling Water","Granola Mix"],
}

# Base prices per product
BASE_PRICE = {
    "Laptop":130,"Smartphone":80,"Tablet":60,"Monitor":40,"Headphones":25,
    "Webcam":20,"Keyboard":18,"Mouse":12,"Office Chair":55,"Standing Desk":90,
    "Bookshelf":35,"Filing Cabinet":30,"Sofa":80,"Lamp":20,"Whiteboard":28,
    "Jacket":45,"Jeans":30,"T-Shirt":15,"Sneakers":50,"Dress Shirt":35,
    "Blazer":60,"Boots":55,"Pen Set":8,"Notebook":10,"Stapler":12,
    "Printer Paper":6,"Binder":8,"Highlighters":5,"Tape":4,
    "Coffee Beans":12,"Energy Drink":3,"Protein Bar":2,"Green Tea":5,
    "Sparkling Water":2,"Granola Mix":6,
}
MARGIN_RATE = {
    "Electronics":0.22,"Furniture":0.30,"Clothing":0.45,
    "Office Supplies":0.35,"Food & Bev":0.40,
}

# Build rows
rows = []
start = datetime(2022, 1, 1)
end   = datetime(2024, 12, 31)
span  = (end - start).days

for i in range(N):
    cat      = np.random.choice(list(CATEGORIES.keys()))
    product  = np.random.choice(CATEGORIES[cat])
    region   = np.random.choice(REGIONS, p=[0.22,0.18,0.20,0.25,0.15])
    qty      = int(np.random.choice([1,2,3,4,5,6,8,10], p=[0.30,0.25,0.18,0.12,0.07,0.04,0.02,0.02]))
    base     = BASE_PRICE[product]
    price    = round(base * (1 + np.random.uniform(-0.15, 0.30)), 2)
    sales    = round(price * qty, 2)
    # occasionally introduce noise: missing values & duplicates
    if np.random.rand() < 0.015:
        sales = np.nan
    profit   = round(sales * MARGIN_RATE[cat] * np.random.uniform(0.7, 1.3), 2) if not np.isnan(sales) else np.nan
    order_dt = start + timedelta(days=int(np.random.triangular(0, span//2, span)))
    rows.append({
        "Order_ID":        f"ORD-{100000+i}",
        "Order_Date":      order_dt.strftime("%Y-%m-%d"),
        "Region":          region,
        "Product_Category":cat,
        "Product_Name":    product,
        "Quantity":        qty,
        "Sales_Amount":    sales,
        "Profit":          profit,
    })

# Inject ~50 duplicate rows
dup_indices = np.random.choice(N, 50, replace=False)
for idx in dup_indices:
    rows.append(rows[idx].copy())

df_raw = pd.DataFrame(rows)
print(f"Raw dataset shape: {df_raw.shape}")

# =============================================================================
# STEP 2 — DATA CLEANING
# =============================================================================
df = df_raw.copy()

# 2a. Remove duplicates
before_dup = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed : {before_dup - len(df)}")

# 2b. Handle missing values
print(f"Missing before     : {df.isnull().sum().sum()}")
df["Sales_Amount"].fillna(df["Sales_Amount"].median(), inplace=True)
df["Profit"].fillna(df["Sales_Amount"] * 0.25, inplace=True)
print(f"Missing after      : {df.isnull().sum().sum()}")

# 2c. Correct data types
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Quantity"]   = df["Quantity"].astype(int)

# =============================================================================
# STEP 3 — DATA TRANSFORMATION
# =============================================================================
df["Year"]          = df["Order_Date"].dt.year
df["Month"]         = df["Order_Date"].dt.month
df["Month_Name"]    = df["Order_Date"].dt.strftime("%b")
df["YearMonth"]     = df["Order_Date"].dt.to_period("M")
df["Profit_Margin"] = (df["Profit"] / df["Sales_Amount"] * 100).round(2)
df["Revenue_Tier"]  = pd.cut(
    df["Sales_Amount"],
    bins=[0, 50, 150, 500, np.inf],
    labels=["Low","Mid","High","Premium"]
)

print(f"\nCleaned dataset shape: {df.shape}")
print(df.dtypes)

# =============================================================================
# STEP 4 — KPI SUMMARY
# =============================================================================
total_revenue  = df["Sales_Amount"].sum()
total_profit   = df["Profit"].sum()
avg_margin     = (total_profit / total_revenue * 100)
top_region     = df.groupby("Region")["Sales_Amount"].sum().idxmax()
total_orders   = df["Order_ID"].nunique()
avg_order_val  = total_revenue / total_orders

print("\n" + "="*55)
print("         KEY PERFORMANCE INDICATORS (KPIs)")
print("="*55)
print(f"  Total Revenue       : ${total_revenue:>12,.2f}")
print(f"  Total Profit        : ${total_profit:>12,.2f}")
print(f"  Profit Margin       : {avg_margin:>11.2f}%")
print(f"  Top-Performing Region: {top_region}")
print(f"  Total Orders        : {total_orders:>12,}")
print(f"  Avg Order Value     : ${avg_order_val:>12,.2f}")
print("="*55)

# =============================================================================
# STEP 5 — VISUALIZATIONS  (6 charts, saved as one figure)
# =============================================================================
fig = plt.figure(figsize=(22, 24), facecolor="#0f1117")
fig.suptitle("📊  Sales Data Analysis Dashboard", fontsize=22, fontweight="bold",
             color="#e6edf3", y=0.98)

# ── 5.1  Total Sales by Region (horizontal bar) ──────────────────────────────
ax1 = fig.add_subplot(3, 2, 1)
reg_sales = df.groupby("Region")["Sales_Amount"].sum().sort_values()
colors_reg = [ACCENT if r == top_region else "#3a4a6b" for r in reg_sales.index]
bars = ax1.barh(reg_sales.index, reg_sales.values, color=colors_reg, edgecolor="none", height=0.6)
ax1.set_title("Total Sales by Region")
ax1.set_xlabel("Sales ($)")
ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
for bar, val in zip(bars, reg_sales.values):
    ax1.text(val + reg_sales.max()*0.01, bar.get_y()+bar.get_height()/2,
             f"${val/1e6:.2f}M", va="center", color="#c9d1d9", fontsize=9)

# ── 5.2  Monthly Sales Trend (line) ──────────────────────────────────────────
ax2 = fig.add_subplot(3, 2, 2)
monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Sales_Amount"].sum()
monthly.index = monthly.index.to_timestamp()
ax2.plot(monthly.index, monthly.values, color=ACCENT, linewidth=2.5, marker="o",
         markersize=4, markerfacecolor="#fff", markeredgecolor=ACCENT)
ax2.fill_between(monthly.index, monthly.values, alpha=0.15, color=ACCENT)
ax2.set_title("Monthly Sales Trend (2022–2024)")
ax2.set_ylabel("Sales ($)")
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
ax2.tick_params(axis="x", rotation=45)

# ── 5.3  Top 10 Products by Revenue ──────────────────────────────────────────
ax3 = fig.add_subplot(3, 2, 3)
top10 = df.groupby("Product_Name")["Sales_Amount"].sum().nlargest(10)
colors_top = [PALETTE[i % len(PALETTE)] for i in range(len(top10))]
bars3 = ax3.barh(top10.index[::-1], top10.values[::-1], color=colors_top[::-1], edgecolor="none", height=0.6)
ax3.set_title("Top 10 Products by Revenue")
ax3.set_xlabel("Revenue ($)")
ax3.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))

# ── 5.4  Profit vs Sales Scatter ─────────────────────────────────────────────
ax4 = fig.add_subplot(3, 2, 4)
cat_colors = {c: PALETTE[i] for i, c in enumerate(df["Product_Category"].unique())}
for cat, grp in df.sample(3000, random_state=1).groupby("Product_Category"):
    ax4.scatter(grp["Sales_Amount"], grp["Profit"], alpha=0.45, s=20,
                color=cat_colors[cat], label=cat, edgecolors="none")
ax4.set_title("Profit vs Sales (Sample 3K)")
ax4.set_xlabel("Sales Amount ($)")
ax4.set_ylabel("Profit ($)")
ax4.legend(fontsize=8, framealpha=0.3)

# ── 5.5  Category-wise Sales Distribution (donut) ────────────────────────────
ax5 = fig.add_subplot(3, 2, 5)
cat_sales = df.groupby("Product_Category")["Sales_Amount"].sum()
wedges, texts, autotexts = ax5.pie(
    cat_sales.values,
    labels=cat_sales.index,
    autopct="%1.1f%%",
    colors=PALETTE[:len(cat_sales)],
    pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor="#0f1117", linewidth=2),
    startangle=140,
)
for at in autotexts:
    at.set_color("#0f1117"); at.set_fontsize(9); at.set_fontweight("bold")
for t in texts:
    t.set_color("#c9d1d9"); t.set_fontsize(9)
ax5.set_title("Sales Distribution by Category")

# ── 5.6  Profit Margin by Category (box) ─────────────────────────────────────
ax6 = fig.add_subplot(3, 2, 6)
order_cats = df.groupby("Product_Category")["Profit_Margin"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Product_Category", y="Profit_Margin", order=order_cats,
            palette=PALETTE[:len(order_cats)], ax=ax6, linewidth=1.2, fliersize=3)
ax6.set_title("Profit Margin Distribution by Category")
ax6.set_xlabel("")
ax6.set_ylabel("Profit Margin (%)")
ax6.tick_params(axis="x", rotation=20)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("sales_dashboard.png", dpi=160, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print("\n✅  Dashboard saved → sales_dashboard.png")

# =============================================================================
# STEP 6 — EXPORT CLEANED DATASET
# =============================================================================
df.to_csv("cleaned_sales_data.csv", index=False)
print("✅  Cleaned dataset exported → cleaned_sales_data.csv")

# =============================================================================
# STEP 7 — INSIGHTS SUMMARY (printed to console)
# =============================================================================
best_cat   = df.groupby("Product_Category")["Sales_Amount"].sum().idxmax()
worst_cat  = df.groupby("Product_Category")["Sales_Amount"].sum().idxmin()
best_prod  = df.groupby("Product_Name")["Sales_Amount"].sum().idxmax()
best_month = monthly.idxmax().strftime("%B %Y")
high_margin_cat = df.groupby("Product_Category")["Profit_Margin"].median().idxmax()

print("\n" + "="*55)
print("               BUSINESS INSIGHTS")
print("="*55)
print(f"  1. '{top_region}' leads in revenue — prioritise")
print(f"     marketing & inventory here.")
print(f"  2. '{best_cat}' is the highest-grossing category.")
print(f"  3. '{best_prod}' is the single best-selling product.")
print(f"  4. Peak sales month: {best_month} — plan campaigns ahead.")
print(f"  5. '{high_margin_cat}' has the highest profit margin —")
print(f"     upselling here maximises bottom-line growth.")
print(f"  6. '{worst_cat}' needs pricing or promotion review.")
print("="*55)
