import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ===========================
# Color Palette
# ===========================
RUBY_RED = "#970C28"
DARK_CYAN = "#008B8B"
MINT_CREAM = "#F4FFFD"
PRUSSIAN_BLUE = "#011936"
CHARCOAL = "#465362"

CHART_PRIMARY = PRUSSIAN_BLUE
CHART_SECONDARY = DARK_CYAN

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="Spend Insights",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================
# Global Styles
# ===========================
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        * {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(135deg, {MINT_CREAM} 0%, #E8FBF8 100%);
        }}

        .block-container {{
            max-width: 1200px;
            padding: 3rem 2rem 5rem 2rem;
        }}

        .hero {{
            background: linear-gradient(135deg, {PRUSSIAN_BLUE}, #023859);
            padding: 4rem 3rem;
            border-radius: 24px;
            color: white;
            margin-bottom: 3rem;
        }}

        .hero h1 {{
            font-family: 'DM Serif Display', serif;
            font-size: 3.5rem;
            margin: 0;
        }}

        .hero p {{
            font-size: 1.25rem;
            opacity: 0.85;
            margin-top: 1rem;
        }}

        .section-title {{
            font-family: 'DM Serif Display', serif;
            font-size: 2rem;
            color: {PRUSSIAN_BLUE};
            margin-bottom: 1.5rem;
            margin-top: 2rem;
        }}

        .insight {{
            background: linear-gradient(135deg, {DARK_CYAN}, #6FDADA);
            border-left: 6px solid {RUBY_RED};
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }}

        .insight p {{
            font-size: 1.1rem;
            font-weight: 500;
            color: white;
            margin: 0;
        }}

        .stat {{
            background: linear-gradient(135deg, {PRUSSIAN_BLUE}, #023859);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            color: white;
        }}

        .stat-label {{
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.1em;
            opacity: 0.7;
        }}

        .stat-value {{
            font-family: 'DM Serif Display', serif;
            font-size: 3rem;
            margin-top: 0.5rem;
        }}

        .muted {{
            color: {CHARCOAL};
            font-size: 0.95rem;
        }}

        .chart-container {{
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(1,25,54,0.08);
            margin: 1.5rem 0;
        }}

        /* Remove Streamlit white box artifacts */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploadDropzone"],
        .element-container,
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="column"],
        [data-testid="stVegaLiteChart"],
        [data-testid="stMetric"],
        .stContainer {{
            background: transparent !important;
        }}

        /* Style metrics */
        [data-testid="stMetricValue"] {{
            font-size: 1.5rem;
            color: {PRUSSIAN_BLUE};
        }}

        [data-testid="stMetricLabel"] {{
            color: {CHARCOAL};
        }}

        /* Style dataframes */
        [data-testid="stDataFrame"] {{
            background: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# Hero
# ===========================
st.markdown(
    """
    <div class="hero">
        <h1>Spend Insights</h1>
        <p>Understand your spending patterns at a glance</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===========================
# Upload Section
# ===========================
with st.container():
    st.markdown('<div class="section-title">Upload your statement</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed"
    )

    st.markdown("<p class='muted'>CSV only. Data is processed locally.</p>", unsafe_allow_html=True)

# ===========================
# Helper Functions
# ===========================
def detect_amount_column(cols):
    """Detect the amount column in the CSV"""
    candidates = ["amount", "$ amount", "$ Amount", "charge", "debit"]
    for c in cols:
        if c.strip().lower() in [x.lower() for x in candidates]:
            return c
    return None

# ===========================
# Main Logic
# ===========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Detect and clean amount column
    amount_col = detect_amount_column(df.columns)
    if not amount_col:
        st.error("Could not find transaction amount column.")
        st.stop()

    df[amount_col] = (
        df[amount_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[amount_col])

    df["abs_amount"] = df[amount_col].abs()

    # Calculate summary statistics
    total_spend = df["abs_amount"].sum()
    tx_count = len(df)
    avg_tx = total_spend / tx_count
    median_spend = df["abs_amount"].median()

    # ===========================
    # Spending Summary
    # ===========================
    with st.container():
        st.markdown('<div class="section-title">Your Spending Summary</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"<div class='stat'><div class='stat-label'>Total Spent</div><div class='stat-value'>${total_spend:,.0f}</div></div>", 
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div class='stat'><div class='stat-label'>Transactions</div><div class='stat-value'>{tx_count}</div></div>", 
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"<div class='stat'><div class='stat-label'>Avg Purchase</div><div class='stat-value'>${avg_tx:,.0f}</div></div>", 
                unsafe_allow_html=True
            )

    # ===========================
    # Machine Learning Models
    # ===========================
    threshold = 100
    df["high_spend"] = (df["abs_amount"] > threshold).astype(int)


    X = df[["abs_amount"]]
    y = df["high_spend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    log_accuracy = accuracy_score(y_test, log_model.predict(X_test))

    # Decision Tree
    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(X_train, y_train)
    tree_accuracy = accuracy_score(y_test, tree_model.predict(X_test))

    # Linear Regression Trend
    df["purchase_order"] = np.arange(len(df))
    lr = LinearRegression()
    lr.fit(df[["purchase_order"]], df["abs_amount"])
    trend_coef = lr.coef_[0]
    r2 = r2_score(df["abs_amount"], lr.predict(df[["purchase_order"]]))

    # ===========================
    # Key Insights
    # ===========================
    with st.container():
        st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

        high_pct = df["high_spend"].mean() * 100

        # Insight 1: High Spend Analysis
        st.markdown(
            f"""
            <div class="insight">
                <p>💡 About {high_pct:.1f}% of your purchases fall into a higher spending group.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # High spend breakdown
        high_spend_df = df[df["high_spend"] == 1]
        normal_spend_df = df[df["high_spend"] == 0]
        
        high_spend_total = high_spend_df["abs_amount"].sum()
        high_spend_avg = high_spend_df["abs_amount"].mean()
        high_spend_count = len(high_spend_df)
        
        normal_spend_total = normal_spend_df["abs_amount"].sum()
        normal_spend_avg = normal_spend_df["abs_amount"].mean()
        normal_spend_count = len(normal_spend_df)

        # Summary cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {DARK_CYAN}, #6FDADA); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">High Spend Purchases</h4>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{high_spend_count} transactions</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;">${high_spend_total:,.0f}</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Avg: ${high_spend_avg:,.0f} per transaction</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {PRUSSIAN_BLUE}, #023859); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">Normal Spend Purchases</h4>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{normal_spend_count} transactions</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;">${normal_spend_total:,.0f}</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Avg: ${normal_spend_avg:,.0f} per transaction</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Top high spend transactions
        st.markdown('<h4 style="margin-top: 2rem; margin-bottom: 1rem; color: ' + PRUSSIAN_BLUE + ';">Top 10 Highest Purchases</h4>', unsafe_allow_html=True)
        
        top_transactions = high_spend_df.nlargest(10, "abs_amount")
        
        # Check what columns are available
        display_cols = ["abs_amount"]
        col_names = {"abs_amount": "Amount"}
        
        # Add optional columns if they exist
        if "Description" in df.columns or "description" in df.columns:
            desc_col = "Description" if "Description" in df.columns else "description"
            display_cols.insert(0, desc_col)
            col_names[desc_col] = "Description"
        
        if "Category" in df.columns or "category" in df.columns:
            cat_col = "Category" if "Category" in df.columns else "category"
            display_cols.append(cat_col)
            col_names[cat_col] = "Category"
        
        if "Date" in df.columns or "date" in df.columns or "Transaction Date" in df.columns:
            date_col = next((c for c in ["Date", "date", "Transaction Date"] if c in df.columns), None)
            if date_col:
                display_cols.insert(0, date_col)
                col_names[date_col] = "Date"
        
        # Format the table
        top_display = top_transactions[display_cols].copy()
        top_display = top_display.rename(columns=col_names)
        
        if "Amount" in top_display.columns:
            top_display["Amount"] = top_display["Amount"].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            top_display,
            use_container_width=True,
            hide_index=True
        )

        # Category breakdown if category exists
        if "Category" in df.columns or "category" in df.columns:
            cat_col = "Category" if "Category" in df.columns else "category"
            
            st.markdown('<h4 style="margin-top: 2rem; margin-bottom: 1rem; color: ' + PRUSSIAN_BLUE + ';">High Spend by Category</h4>', unsafe_allow_html=True)
            
            category_spend = high_spend_df.groupby(cat_col)["abs_amount"].agg([
                ('Total', 'sum'),
                ('Count', 'count'),
                ('Average', 'mean')
            ]).reset_index()
            
            category_spend = category_spend.sort_values('Total', ascending=False).head(10)
            category_spend.columns = ['Category', 'Total Spent', 'Transactions', 'Avg per Transaction']
            
            # Format currency
            category_spend['Total Spent'] = category_spend['Total Spent'].apply(lambda x: f"${x:,.0f}")
            category_spend['Avg per Transaction'] = category_spend['Avg per Transaction'].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(
                category_spend,
                use_container_width=True,
                hide_index=True
            )
            
            # Category chart
            cat_chart_data = high_spend_df.groupby(cat_col)["abs_amount"].sum().reset_index()
            cat_chart_data = cat_chart_data.sort_values("abs_amount", ascending=False).head(8)
            cat_chart_data.columns = ["Category", "Amount"]
            
            category_chart = (
                alt.Chart(cat_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("Amount:Q", title="Total Spent ($)"),
                    y=alt.Y("Category:N", sort="-x", title="Category"),
                    color=alt.value(DARK_CYAN)
                )
                .properties(height=300)
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.altair_chart(category_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Spending distribution comparison
        st.markdown('<h4 style="margin-top: 2rem; margin-bottom: 1rem; color: ' + PRUSSIAN_BLUE + ';">Spending Distribution Comparison</h4>', unsafe_allow_html=True)
        
        spend_counts = df["high_spend"].value_counts().rename(
            {0: "Normal Spend", 1: "High Spend"}
        ).reset_index()
        spend_counts.columns = ["Spend Type", "Count"]

        spend_chart = (
            alt.Chart(spend_counts)
            .mark_bar()
            .encode(
                x=alt.X("Spend Type:N", title="Spend Type"),
                y=alt.Y("Count:Q", title="Number of Transactions"),
                color=alt.Color(
                    "Spend Type:N",
                    scale=alt.Scale(
                        domain=["Normal Spend", "High Spend"],
                        range=[CHART_PRIMARY, CHART_SECONDARY]
                    ),
                    legend=None
                )
            )
            .properties(height=250)
        )

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.altair_chart(spend_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key takeaway
        pct_of_total = (high_spend_total / total_spend) * 100
        st.markdown(
            f"""
            <div style="background: #FFF3CD; border-left: 4px solid #FFC107; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <p style="margin: 0; color: #856404; font-weight: 500;">
                    💰 Your high-spend purchases represent {high_pct:.1f}% of transactions but account for {pct_of_total:.1f}% of your total spending.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #E0E0E0;'>", unsafe_allow_html=True)

        # Insight 2: Decision Tree Threshold
        # The decision tree is trained to classify purchases above the
        # fixed business threshold of $100 as "high spend".
        # We are NOT extracting a learned threshold from the tree.
        # We are visualizing the $100 rule directly.

        st.markdown(
            f"""
            <div class="insight">
                <p>📌 Purchases above ${threshold:,.0f} tend to fall into the higher spending group.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Visual explanation:
        # - Scatter plot of all purchases
        # - Vertical rule at $100 threshold
        # This shows how the classification boundary works.

        threshold_line = alt.Chart(
            pd.DataFrame({"Threshold": [threshold]})
        ).mark_rule(
            color=RUBY_RED,
            strokeWidth=3
        ).encode(
            x="Threshold:Q"
        )

        scatter = alt.Chart(df).mark_circle(
            size=60,
            opacity=0.5
        ).encode(
            x=alt.X("abs_amount:Q", title="Purchase Amount ($)"),
            y=alt.Y("high_spend:N", title="High Spend Group"),
            color=alt.Color(
                "high_spend:N",
                scale=alt.Scale(
                    domain=[0, 1],
                    range=[CHART_PRIMARY, CHART_SECONDARY]
                ),
                legend=None
            )
        )

        decision_tree_chart = (scatter + threshold_line).properties(height=300)

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.altair_chart(decision_tree_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Show model accuracy
        st.metric("Decision Tree Accuracy", f"{tree_accuracy:.2f}")

        st.markdown(
            "<p class='muted'>The decision tree predicts whether a purchase exceeds the $100 threshold based on transaction size.</p>",
            unsafe_allow_html=True
        )



        # Insight 3: Spending Trend
        st.markdown(
            f"""
            <div class="insight">
                <p>📈 Overall spending is {"increasing" if trend_coef > 0 else "stable or decreasing"} over time.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Prepare trend data
        trend_df = df[["purchase_order", "abs_amount"]].copy()
        trend_df["Trend"] = lr.predict(df[["purchase_order"]])

        # Trend chart
        trend_chart = (
            alt.Chart(trend_df)
            .transform_fold(
                ["abs_amount", "Trend"],
                as_=["Series", "Value"]
            )
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("purchase_order:Q", title="Transaction Order"),
                y=alt.Y("Value:Q", title="Amount ($)"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["abs_amount", "Trend"],
                        range=[CHART_PRIMARY, CHART_SECONDARY]
                    ),
                    legend=alt.Legend(title="Series")
                )
            )
            .properties(height=300)
        )

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.altair_chart(trend_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.metric("Trend R² Score", f"{r2:.2f}")
        
        # Add bottom spacing so content is fully visible
        st.markdown('<div style="padding-bottom: 4rem;"></div>', unsafe_allow_html=True)
