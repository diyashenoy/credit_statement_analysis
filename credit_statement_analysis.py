import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Bank of America Color Palette
BOA_COLORS = {
    'primary_red': '#E31837',
    'dark_blue': '#012169',
    'light_blue': '#0073CF',
    'gray': '#6E7780',
    'light_gray': '#F5F5F5',
    'white': '#FFFFFF',
    'success_green': '#00A758',
    'warning_orange': '#FF6B35'
}

# Configure page
st.set_page_config(
    page_title="Smart Spending Insights",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with BofA colors
st.markdown(f"""
    <style>
    .main {{
        background-color: {BOA_COLORS['white']};
    }}
    .stButton>button {{
        background-color: {BOA_COLORS['primary_red']};
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: {BOA_COLORS['dark_blue']};
    }}
    .upload-section {{
        background-color: {BOA_COLORS['light_gray']};
        padding: 30px;
        border-radius: 10px;
        border: 2px dashed {BOA_COLORS['gray']};
        text-align: center;
    }}
    .metric-card {{
        background-color: {BOA_COLORS['light_gray']};
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid {BOA_COLORS['primary_red']};
    }}
    .insight-card {{
        background-color: {BOA_COLORS['light_blue']}15;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid {BOA_COLORS['light_blue']};
        margin: 10px 0;
    }}
    .header-text {{
        color: {BOA_COLORS['dark_blue']};
        font-weight: 700;
    }}
    .subheader-text {{
        color: {BOA_COLORS['gray']};
    }}
    h1, h2, h3 {{
        color: {BOA_COLORS['dark_blue']};
    }}
    </style>
""", unsafe_allow_html=True)


def categorize_transaction(description):
    """Automatically categorize transactions based on merchant name"""
    description = description.upper()
    
    categories = {
        'Food & Dining': ['DOORDASH', 'UBER *EATS', 'DOMINO', 'CHIPOTLE', 'RESTAURANT', 
                          'CAFE', 'FOOD', 'TACO', 'PIZZA', 'BURGER', 'KITCHEN', 'RISTORA'],
        'Transportation': ['UBER *TRIP', 'TRANSIT', 'DELTA AIR', 'AIRLINE', 'PARKING'],
        'Shopping': ['AMAZON', 'SURF STYLE', 'MICHAELS'],
        'Groceries': ['AMAZON GROCE', 'SAFEWAY', 'QFC', 'MAYURI FOODS', 'GROCERY'],
        'Health & Wellness': ['CVS', 'WALGREENS', 'PHARMACY', 'WELLNESS', 'SPA', 'ZOOMCARE'],
        'Entertainment': ['NETFLIX', 'SPOTIFY', 'LIV MIAMI', 'CLUB'],
        'Fitness': ['EQUINOX', 'GYM'],
        'Insurance': ['LEMONADE INSURANCE'],
        'Utilities': ['SEATTLE CITY LIGHT', 'ELECTRIC', 'WATER', 'GAS'],
        'Subscriptions': ['APPLE.COM', 'NETFLIX', 'SPOTIFY', 'CHATGPT', 'OPENAI'],
        'Personal Care': ['NAILS', 'SALON', 'PARFUM', 'BEAUTY']
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in description:
                return category
    
    return 'Other'


def parse_credit_card_statement(df):
    """Parse and clean credit card statement data"""
    try:
        # Identify relevant columns
        required_cols = []
        
        # Find date column
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'transaction' in col.lower()]
        # Find description column
        desc_cols = [col for col in df.columns if 'merchant' in col.lower() or 'description' in col.lower()]
        # Find amount column
        amount_cols = [col for col in df.columns if 'amount' in col.lower() or '$' in col.lower()]
        
        if not date_cols or not desc_cols or not amount_cols:
            return None, "Could not identify required columns (Date, Description, Amount)"
        
        # Create standardized dataframe
        clean_df = pd.DataFrame()
        clean_df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        clean_df['Description'] = df[desc_cols[0]].astype(str)
        clean_df['Amount'] = pd.to_numeric(df[amount_cols[0]], errors='coerce')
        
        # Remove payment rows
        clean_df = clean_df[~clean_df['Description'].str.contains('Payment|PAYMENT', case=False, na=False)]
        
        # Remove rows with missing critical data
        clean_df = clean_df.dropna(subset=['Date', 'Amount'])
        
        # Only keep purchases (negative amounts or positive based on context)
        clean_df['Amount'] = clean_df['Amount'].abs()
        clean_df = clean_df[clean_df['Amount'] > 0]
        
        # Add category
        clean_df['Category'] = clean_df['Description'].apply(categorize_transaction)
        
        # Add additional features
        clean_df['Month'] = clean_df['Date'].dt.strftime('%Y-%m')
        clean_df['Day_of_Week'] = clean_df['Date'].dt.day_name()
        clean_df['Is_Weekend'] = clean_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        return clean_df, None
        
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def train_spending_pattern_models(df):
    """Train multiple ML models to predict spending categories"""
    
    # Prepare features
    feature_df = df.copy()
    
    # Create features
    le_dow = LabelEncoder()
    feature_df['Day_of_Week_Encoded'] = le_dow.fit_transform(feature_df['Day_of_Week'])
    
    # Features: Amount, Is_Weekend, Day_of_Week
    X = feature_df[['Amount', 'Is_Weekend', 'Day_of_Week_Encoded']]
    y = feature_df['Category']
    
    # Encode target
    le_category = LabelEncoder()
    y_encoded = le_category.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Model 1: Logistic Regression
    st.write("🔄 Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    models['Logistic Regression'] = lr_model
    results['Logistic Regression'] = {
        'accuracy': lr_accuracy,
        'predictions': lr_pred,
        'model': lr_model
    }
    
    # Model 2: Random Forest
    st.write("🔄 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'accuracy': rf_accuracy,
        'predictions': rf_pred,
        'feature_importance': rf_model.feature_importances_,
        'model': rf_model
    }
    
    # Model 3: Gradient Boosting
    st.write("🔄 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {
        'accuracy': gb_accuracy,
        'predictions': gb_pred,
        'feature_importance': gb_model.feature_importances_,
        'model': gb_model
    }
    
    return results, le_category, scaler, X_test, y_test


def main():
    # Header
    st.markdown(f"""
        <h1 style='color: {BOA_COLORS['dark_blue']}; text-align: center;'>
            💳 Smart Spending Insights
        </h1>
        <p style='text-align: center; color: {BOA_COLORS['gray']}; font-size: 18px;'>
            Transform your credit card statements into actionable insights
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 1: Upload Section
    st.markdown(f"""
        <div class='upload-section'>
            <h2 style='color: {BOA_COLORS['dark_blue']}'>📁 Upload Your Statement</h2>
            <p style='color: {BOA_COLORS['gray']}'>
                Upload your credit card statement (CSV or Excel format)<br>
                <small>Supported formats: .csv, .xlsx, .xls</small>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your credit card statement in CSV or Excel format"
    )
    
    if uploaded_file is not None:
        # Step 2: Validate and Parse
        with st.spinner('📊 Reading and validating your file...'):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                
                st.success("✅ File uploaded successfully!")
                
                # Show raw data preview
                with st.expander("📄 Preview Raw Data"):
                    st.dataframe(df_raw.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                return
        
        # Parse the data
        with st.spinner('🔍 Parsing transactions...'):
            df_clean, error = parse_credit_card_statement(df_raw)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 Please ensure your file has columns for Date, Description, and Amount")
                return
            
            st.success(f"✅ Successfully parsed {len(df_clean)} transactions!")
        
        # Step 3: Categorize and Display Overview
        st.markdown("---")
        st.markdown(f"<h2 class='header-text'>📊 Spending Overview</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_spent = df_clean['Amount'].sum()
        avg_transaction = df_clean['Amount'].mean()
        num_transactions = len(df_clean)
        top_category = df_clean.groupby('Category')['Amount'].sum().idxmax()
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='color: {BOA_COLORS['gray']}; margin: 0;'>Total Spent</h4>
                    <h2 style='color: {BOA_COLORS['primary_red']}; margin: 5px 0;'>${total_spent:,.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='color: {BOA_COLORS['gray']}; margin: 0;'>Transactions</h4>
                    <h2 style='color: {BOA_COLORS['dark_blue']}; margin: 5px 0;'>{num_transactions}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='color: {BOA_COLORS['gray']}; margin: 0;'>Avg Transaction</h4>
                    <h2 style='color: {BOA_COLORS['light_blue']}; margin: 5px 0;'>${avg_transaction:,.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='color: {BOA_COLORS['gray']}; margin: 0;'>Top Category</h4>
                    <h2 style='color: {BOA_COLORS['success_green']}; margin: 5px 0;'>{top_category}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Spending by Category
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<h3 class='header-text'>Spending by Category</h3>", unsafe_allow_html=True)
            category_spending = df_clean.groupby('Category')['Amount'].sum().sort_values(ascending=False)
            
            fig_pie = px.pie(
                values=category_spending.values,
                names=category_spending.index,
                title="",
                color_discrete_sequence=[BOA_COLORS['primary_red'], BOA_COLORS['dark_blue'], 
                                        BOA_COLORS['light_blue'], BOA_COLORS['success_green'],
                                        BOA_COLORS['warning_orange'], BOA_COLORS['gray']]
            )
            fig_pie.update_layout(
                font=dict(size=12),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown(f"<h3 class='header-text'>Top Spending Categories</h3>", unsafe_allow_html=True)
            
            fig_bar = px.bar(
                x=category_spending.head(8).values,
                y=category_spending.head(8).index,
                orientation='h',
                title="",
                labels={'x': 'Amount ($)', 'y': 'Category'},
                color=category_spending.head(8).values,
                color_continuous_scale=[[0, BOA_COLORS['light_blue']], [1, BOA_COLORS['primary_red']]]
            )
            fig_bar.update_layout(
                showlegend=False,
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Spending Over Time
        st.markdown(f"<h3 class='header-text'>Spending Trends</h3>", unsafe_allow_html=True)
        
        daily_spending = df_clean.groupby('Date')['Amount'].sum().reset_index()
        
        fig_line = px.line(
            daily_spending,
            x='Date',
            y='Amount',
            title="",
            labels={'Amount': 'Daily Spending ($)', 'Date': 'Date'}
        )
        fig_line.update_traces(line_color=BOA_COLORS['primary_red'], line_width=2)
        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Step 4: Machine Learning Analysis
        st.markdown("---")
        st.markdown(f"<h2 class='header-text'>🤖 AI-Powered Analysis</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {BOA_COLORS['gray']}'>Running multiple machine learning models to analyze your spending patterns...</p>", unsafe_allow_html=True)
        
        if st.button("🚀 Run AI Analysis", use_container_width=True):
            with st.spinner('Training AI models...'):
                results, le_category, scaler, X_test, y_test = train_spending_pattern_models(df_clean)
            
            st.success("✅ AI analysis complete!")
            
            # Model Performance Comparison
            st.markdown(f"<h3 class='header-text'>Model Performance Comparison</h3>", unsafe_allow_html=True)
            
            model_comparison = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[model]['accuracy'] for model in results.keys()]
            }).sort_values('Accuracy', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("<br>", unsafe_allow_html=True)
                for idx, row in model_comparison.iterrows():
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color: {BOA_COLORS['gray']}; margin: 0;'>{row['Model']}</h4>
                            <h2 style='color: {BOA_COLORS['primary_red']}; margin: 5px 0;'>{row['Accuracy']:.1%}</h2>
                            <p style='color: {BOA_COLORS['gray']}; font-size: 12px; margin: 0;'>Accuracy Score</p>
                        </div>
                        <br>
                    """, unsafe_allow_html=True)
            
            with col2:
                fig_comparison = px.bar(
                    model_comparison,
                    x='Accuracy',
                    y='Model',
                    orientation='h',
                    title="Model Accuracy Comparison",
                    labels={'Accuracy': 'Accuracy Score', 'Model': 'Model'},
                    color='Accuracy',
                    color_continuous_scale=[[0, BOA_COLORS['light_blue']], [1, BOA_COLORS['success_green']]]
                )
                fig_comparison.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_tickformat='.0%'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            st.markdown(f"<h3 class='header-text'>What Drives Your Spending?</h3>", unsafe_allow_html=True)
            
            feature_names = ['Transaction Amount', 'Weekend Spending', 'Day of Week']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'feature_importance' in results['Random Forest']:
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': results['Random Forest']['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Random Forest - Feature Importance",
                        color='Importance',
                        color_continuous_scale=[[0, BOA_COLORS['light_blue']], [1, BOA_COLORS['primary_red']]]
                    )
                    fig_importance.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                if 'feature_importance' in results['Gradient Boosting']:
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': results['Gradient Boosting']['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Gradient Boosting - Feature Importance",
                        color='Importance',
                        color_continuous_scale=[[0, BOA_COLORS['light_blue']], [1, BOA_COLORS['primary_red']]]
                    )
                    fig_importance.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            # Step 5: Insights
            st.markdown("---")
            st.markdown(f"<h2 class='header-text'>💡 Key Insights</h2>", unsafe_allow_html=True)
            
            # Generate insights
            insights = []
            
            # Top spending category
            top_cat_amount = category_spending.iloc[0]
            top_cat_name = category_spending.index[0]
            insights.append(f"Your highest spending is in **{top_cat_name}** at **${top_cat_amount:,.2f}**, representing {(top_cat_amount/total_spent)*100:.1f}% of total spending.")
            
            # Weekend vs Weekday
            weekend_spending = df_clean[df_clean['Is_Weekend'] == 1]['Amount'].sum()
            weekday_spending = df_clean[df_clean['Is_Weekend'] == 0]['Amount'].sum()
            weekend_days = df_clean[df_clean['Is_Weekend'] == 1]['Date'].nunique()
            weekday_days = df_clean[df_clean['Is_Weekend'] == 0]['Date'].nunique()
            
            if weekend_days > 0 and weekday_days > 0:
                avg_weekend = weekend_spending / weekend_days
                avg_weekday = weekday_spending / weekday_days
                if avg_weekend > avg_weekday:
                    insights.append(f"You tend to spend more on weekends (**${avg_weekend:.2f}**/day) compared to weekdays (**${avg_weekday:.2f}**/day).")
                else:
                    insights.append(f"Your weekday spending (**${avg_weekday:.2f}**/day) is higher than weekend spending (**${avg_weekend:.2f}**/day).")
            
            # High-value transactions
            high_value_threshold = df_clean['Amount'].quantile(0.9)
            high_value_count = len(df_clean[df_clean['Amount'] > high_value_threshold])
            insights.append(f"You have **{high_value_count}** high-value transactions (over ${high_value_threshold:.2f}) that account for a significant portion of your spending.")
            
            # Most frequent category
            most_frequent_cat = df_clean['Category'].mode()[0]
            freq_count = len(df_clean[df_clean['Category'] == most_frequent_cat])
            insights.append(f"**{most_frequent_cat}** is your most frequent spending category with **{freq_count}** transactions.")
            
            for idx, insight in enumerate(insights, 1):
                st.markdown(f"""
                    <div class='insight-card'>
                        <h4 style='color: {BOA_COLORS['dark_blue']}; margin: 0 0 10px 0;'>Insight #{idx}</h4>
                        <p style='color: {BOA_COLORS['gray']}; margin: 0;'>{insight}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Step 6: Suggestions
            st.markdown("---")
            st.markdown(f"<h2 class='header-text'>🎯 Suggested Actions</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {BOA_COLORS['gray']}'>Based on your spending patterns, here are some suggestions to consider:</p>", unsafe_allow_html=True)
            
            suggestions = []
            
            # Suggestion based on top category
            if top_cat_name in ['Food & Dining', 'Shopping']:
                suggestions.append(f"Consider setting a monthly budget for **{top_cat_name}** to track this major expense category.")
            
            # Subscription check
            subscription_cats = df_clean[df_clean['Category'] == 'Subscriptions']
            if len(subscription_cats) > 0:
                sub_total = subscription_cats['Amount'].sum()
                suggestions.append(f"Review your **${sub_total:.2f}** in subscriptions. Cancel any services you're not actively using.")
            
            # Weekend spending
            if 'avg_weekend' in locals() and avg_weekend > avg_weekday * 1.3:
                suggestions.append(f"Your weekend spending is notably higher. Planning weekend activities with a budget in mind could help reduce costs.")
            
            # Small frequent purchases
            small_purchases = df_clean[df_clean['Amount'] < 20]
            if len(small_purchases) > 20:
                small_total = small_purchases['Amount'].sum()
                suggestions.append(f"You have **{len(small_purchases)}** small purchases totaling **${small_total:.2f}**. These add up quickly – consider consolidating shopping trips.")
            
            for idx, suggestion in enumerate(suggestions, 1):
                st.markdown(f"""
                    <div style='background-color: {BOA_COLORS['success_green']}15; padding: 15px; border-radius: 8px; 
                                border-left: 4px solid {BOA_COLORS['success_green']}; margin: 10px 0;'>
                        <p style='color: {BOA_COLORS['dark_blue']}; margin: 0; font-weight: 500;'>
                            ✓ {suggestion}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed breakdown
            st.markdown("---")
            st.markdown(f"<h3 class='header-text'>📋 Detailed Transaction View</h3>", unsafe_allow_html=True)
            
            with st.expander("View All Transactions"):
                display_df = df_clean[['Date', 'Description', 'Category', 'Amount']].sort_values('Date', ascending=False)
                st.dataframe(
                    display_df.style.format({'Amount': '${:.2f}'}),
                    use_container_width=True,
                    height=400
                )
    
    else:
        # Show instructions when no file uploaded
        st.info("👆 Upload your credit card statement to get started")
        
        st.markdown(f"""
            <div style='background-color: {BOA_COLORS['light_gray']}; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h3 style='color: {BOA_COLORS['dark_blue']}'>How It Works</h3>
                <ol style='color: {BOA_COLORS['gray']}'>
                    <li><strong>Upload</strong> your credit card statement (CSV or Excel)</li>
                    <li><strong>Validate</strong> - We'll automatically parse your transactions</li>
                    <li><strong>Categorize</strong> - AI groups your spending into categories</li>
                    <li><strong>Analyze</strong> - Three ML models analyze your spending patterns</li>
                    <li><strong>Insights</strong> - Get actionable insights and suggestions</li>
                </ol>
                
                <h4 style='color: {BOA_COLORS['dark_blue']}; margin-top: 20px;'>What You'll Get</h4>
                <ul style='color: {BOA_COLORS['gray']}'>
                    <li>📊 Visual spending breakdowns by category</li>
                    <li>🤖 AI-powered spending pattern analysis</li>
                    <li>💡 Personalized insights about your habits</li>
                    <li>🎯 Actionable suggestions to optimize spending</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()