import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

state_coords = {
    "California": (36.7783, -119.4179),
    "Texas": (31.9686, -99.9018),
    "New York": (43.0000, -75.0000),
    "Washington": (47.7511, -120.7401),
    "Florida": (27.6648, -81.5158),
    "Illinois": (40.0000, -89.0000),
    "Pennsylvania": (41.2033, -77.1945),
    "Ohio": (40.4173, -82.9071),
    "Michigan": (44.3148, -85.6024),
    "North Carolina": (35.7596, -79.0193)
}

# ─────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────
df = pd.read_csv("dataset/Superstore dataset.csv", encoding="latin1")
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')

profit_model   = joblib.load("models/profit_classifier.pkl")
cluster_model  = joblib.load("models/customer_cluster.pkl")

anomaly_df = df.copy()

features = anomaly_df[
    ['Sales', 'Profit', 'Quantity', 'Discount']
]

iso = IsolationForest(
    contamination=0.03,
    random_state=42
)

anomaly_df['Anomaly'] = iso.fit_predict(features)
# ─────────────────────────────────────────
# GLOBAL KPIs
# ─────────────────────────────────────────
total_sales     = round(df['Sales'].sum(), 2)
total_profit    = round(df['Profit'].sum(), 2)
total_orders    = df['Order ID'].nunique()
total_customers = df['Customer ID'].nunique()

# ─────────────────────────────────────────
# PLOTLY DARK THEME HELPER
# ─────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor='#2a2d3a', showline=False),
    yaxis=dict(gridcolor='#2a2d3a', showline=False),
)

def apply_dark(fig):
    fig.update_layout(**DARK_LAYOUT)
    return fig

# ─────────────────────────────────────────
# BUILD CHARTS
# ─────────────────────────────────────────

# Chart 1 — Sales by Category
cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
fig1 = px.bar(cat_sales, x='Category', y='Sales',
              color_discrete_sequence=['#6ee7b7'])
apply_dark(fig1)
sales_chart = fig1.to_html(full_html=False)

# Chart 2 — Region Wise Sales (pie)
reg_sales = df.groupby('Region')['Sales'].sum().reset_index()
fig2 = px.pie(reg_sales, names='Region', values='Sales',
              color_discrete_sequence=['#818cf8','#fb923c','#6ee7b7','#f87171'])
apply_dark(fig2)
region_chart = fig2.to_html(full_html=False)

# Chart 3 — Monthly Sales Trend
monthly = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly['Order Date'] = monthly['Order Date'].astype(str)
fig3 = px.line(monthly, x='Order Date', y='Sales',
               color_discrete_sequence=['#818cf8'])
fig3.update_traces(line_width=2)
apply_dark(fig3)
monthly_chart = fig3.to_html(full_html=False)

# Chart 4 — Top 10 Sub-Categories by Profit
sub_profit = df.groupby('Sub-Category')['Profit'].sum().reset_index()
sub_profit = sub_profit.sort_values('Profit', ascending=True).tail(10)
fig4 = px.bar(sub_profit, x='Profit', y='Sub-Category', orientation='h',
              color='Profit', color_continuous_scale=['#f87171','#fbbf24','#6ee7b7'])
apply_dark(fig4)
profit_chart = fig4.to_html(full_html=False)

# ─────────────────────────────────────────
# BUILD DATA CONTEXT FOR CHATBOT
# ─────────────────────────────────────────
def build_data_context():
    top_customers = (
        df.groupby('Customer Name')['Sales']
          .sum().sort_values(ascending=False).head(5)
    )
    top_products = (
        df.groupby('Product Name')['Profit']
          .sum().sort_values(ascending=False).head(5)
    )
    low_products = (
        df.groupby('Product Name')['Profit']
          .sum().sort_values().head(5)
    )
    region_summary = df.groupby('Region').agg(
        Sales=('Sales','sum'), Profit=('Profit','sum')
    ).round(2)
    cat_summary = df.groupby('Category').agg(
        Sales=('Sales','sum'), Profit=('Profit','sum')
    ).round(2)

    ctx = f"""
You are an AI analyst assistant for a retail store. Here is the store's data summary:

OVERALL KPIs:
- Total Sales: ₹{total_sales:,.2f}
- Total Profit: ₹{total_profit:,.2f}
- Total Orders: {total_orders}
- Total Customers: {total_customers}
- Profit Margin: {round(total_profit/total_sales*100,1)}%

SALES BY CATEGORY:
{cat_summary.to_string()}

SALES BY REGION:
{region_summary.to_string()}

TOP 5 CUSTOMERS BY SALES:
{top_customers.to_string()}

TOP 5 MOST PROFITABLE PRODUCTS:
{top_products.to_string()}

BOTTOM 5 PRODUCTS (LOSS MAKING):
{low_products.to_string()}

Answer questions concisely and helpfully based on this data.
Use bullet points where appropriate. Keep responses under 200 words unless detail is needed.
"""
    return ctx

DATA_CONTEXT = build_data_context()

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def dashboard():
    return render_template(
        'dashboard.html',
        total_sales=total_sales,
        total_profit=total_profit,
        total_orders=total_orders,
        total_customers=total_customers,
        sales_chart=sales_chart,
        region_chart=region_chart,
        monthly_chart=monthly_chart,
        profit_chart=profit_chart,
    )


@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['sales']),
            float(request.form['quantity']),
            float(request.form['discount']),
            int(request.form['ship_mode']),
            int(request.form['segment']),
            int(request.form['region']),
            int(request.form['category']),
            int(request.form['subcategory']),
        ]
        result = profit_model.predict([features])[0]
    except Exception as e:
        result = f"Error: {e}"
    return render_template('prediction.html', result=result)


# ── CHATBOT ──────────────────────────────

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


@app.route('/chat', methods=['POST'])
def chat():

    user_message = request.json.get('message', '').lower()

    if "sales" in user_message:

        reply = f"""
        Total Sales: ₹{total_sales:,.2f}

        Total Orders: {total_orders}

        Total Customers: {total_customers}
        """

    elif "profit" in user_message:

        reply = f"""
        Total Profit: ₹{total_profit:,.2f}

        Profit Margin:
        {round(total_profit/total_sales*100,2)}%
        """

    elif "best category" in user_message:

        best_category = (
            df.groupby("Category")["Sales"]
            .sum()
            .idxmax()
        )

        reply = f"""
        Best Performing Category:
        {best_category}
        """

    elif "best region" in user_message:

        best_region = (
            df.groupby("Region")["Profit"]
            .sum()
            .idxmax()
        )

        reply = f"""
        Most Profitable Region:
        {best_region}
        """

    elif "loss" in user_message:

        worst_product = (
            df.groupby("Product Name")["Profit"]
            .sum()
            .idxmin()
        )

        reply = f"""
        Biggest Loss Product:

        {worst_product}

        Review discounts and pricing.
        """

    elif "recommendation" in user_message:

        reply = """
        Business Recommendations:

        • Increase Technology inventory

        • Reduce discount-heavy products

        • Focus on profitable regions

        • Improve low-performing categories
        """

    else:

        reply = """
        Ask me:

        • Total Sales

        • Total Profit

        • Best Category

        • Best Region

        • Loss Making Products

        • Business Recommendations
        """

    return jsonify({
        "reply": reply
    })


# ── CUSTOMERS ────────────────────────────

@app.route('/customers')
def customers_page():
    # Build customer-level features for clustering display
    cust_df = df.groupby('Customer ID').agg(
        CustomerName=('Customer Name', 'first'),
        TotalSales=('Sales', 'sum'),
        TotalProfit=('Profit', 'sum'),
        OrderCount=('Order ID', 'nunique'),
    ).reset_index()
    cust_df['TotalSales']  = cust_df['TotalSales'].round(2)
    cust_df['TotalProfit'] = cust_df['TotalProfit'].round(2)

    # Temporary segmentation based on sales quartiles

    cust_df['Cluster'] = pd.qcut(
        cust_df['TotalSales'],
        q=4,
        labels=[0, 1, 2, 3]
    )
    cust_df['Cluster'] = cust_df['Cluster'].astype(int)

    n_clusters = int(cust_df['Cluster'].nunique())

    SEGMENT_NAMES   = ['High Value', 'Loyal Buyers', 'At Risk', 'New Customers']
    SEGMENT_COLORS  = ['110,231,183', '129,140,248', '251,146,60', '248,113,113']

    cluster_summary = []
    for i in range(n_clusters):
        sub = cust_df[cust_df['Cluster'] == i]
        cluster_summary.append({
            'id': i,
            'name': SEGMENT_NAMES[i] if i < len(SEGMENT_NAMES) else f'Segment {i}',
            'count': len(sub),
            'avg_sales': round(sub['TotalSales'].mean(), 0),
            'color_rgb': SEGMENT_COLORS[i] if i < len(SEGMENT_COLORS) else '150,150,150',
        })

    # Pie chart
    pie_data = cust_df.groupby('Cluster').size().reset_index(name='Count')
    fig_pie = px.pie(pie_data, names='Cluster', values='Count',
                     color_discrete_sequence=['#6ee7b7','#818cf8','#fb923c','#f87171'])
    apply_dark(fig_pie)

    # Bar chart: avg sales/profit per cluster
    bar_data = cust_df.groupby('Cluster').agg(
        AvgSales=('TotalSales','mean'), AvgProfit=('TotalProfit','mean')
    ).reset_index().round(2)
    fig_bar = go.Figure()
    fig_bar.add_bar(x=bar_data['Cluster'].astype(str), y=bar_data['AvgSales'],
                    name='Avg Sales', marker_color='#6ee7b7')
    fig_bar.add_bar(x=bar_data['Cluster'].astype(str), y=bar_data['AvgProfit'],
                    name='Avg Profit', marker_color='#818cf8')
    apply_dark(fig_bar)

    # Scatter: Sales vs Profit
    fig_scatter = px.scatter(
        cust_df, x='TotalSales', y='TotalProfit', color='Cluster',
        hover_data=['CustomerName'],
        color_discrete_sequence=['#6ee7b7','#818cf8','#fb923c','#f87171']
    )
    apply_dark(fig_scatter)

    return render_template(
        'customers.html',
        total_customers=total_customers,
        n_clusters=n_clusters,
        cluster_summary=cluster_summary,
        cluster_pie=fig_pie.to_html(full_html=False),
        cluster_bar=fig_bar.to_html(full_html=False),
        scatter_chart=fig_scatter.to_html(full_html=False),
    )


# ── FORECAST ─────────────────────────────

@app.route('/forecast')
def forecast_page():
    # Aggregate historical monthly sales
    monthly = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index()

    # Predict next 30 days using forecast model
    # Assumes model was trained to predict next-step given last N values
    # Adjust this logic based on how your forecast_model was built
    predicted_total = round(
        float(monthly.tail(3).mean()),
        2
    )

    last_month_sales = round(float(monthly.iloc[-1]), 2)
    growth_pct = round((predicted_total - last_month_sales) / last_month_sales * 100, 1)

    # Build forecast dates
    import pandas as pd
    last_date = monthly.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=1, freq='MS')

    # Combined chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.values,
        mode='lines+markers', name='Historical',
        line=dict(color='#818cf8', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates),
        y=[predicted_total],
        mode='markers',
        name='Forecast',
        marker=dict(color='#6ee7b7', size=12, symbol='star')
    ))
    # Shaded forecast area
    fig.add_vrect(
        x0=str(forecast_dates[0]), x1=str(forecast_dates[0]),
        fillcolor='rgba(110,231,183,0.05)',
        layer='below', line_width=0
    )
    apply_dark(fig)
    fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)'))

    return render_template(
        'forecast.html',
        predicted_total=f"{predicted_total:,.0f}",
        growth_pct=growth_pct,
        forecast_chart=fig.to_html(full_html=False),
    )

@app.route('/insights')
def insights():

    top_category = (
        df.groupby('Category')['Sales']
        .sum()
        .idxmax()
    )

    top_region = (
        df.groupby('Region')['Profit']
        .sum()
        .idxmax()
    )

    worst_subcategory = (
        df.groupby('Sub-Category')['Profit']
        .sum()
        .idxmin()
    )

    best_customer = (
        df.groupby('Customer Name')['Sales']
        .sum()
        .idxmax()
    )

    return render_template(
        'insights.html',
        top_category=top_category,
        top_region=top_region,
        worst_subcategory=worst_subcategory,
        best_customer=best_customer
    )

@app.route('/products')
def products():

    top_products = (
        df.groupby('Product Name')
        ['Sales']
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    loss_products = (
        df.groupby('Product Name')
        ['Profit']
        .sum()
        .sort_values()
        .head(10)
    )

    product_profit = (
        df.groupby('Product Name')
        .agg({
            'Sales':'sum',
            'Profit':'sum'
        })
        .reset_index()
    )

    high_sales_low_profit = product_profit[
        (product_profit['Sales'] > product_profit['Sales'].mean())
        &
        (product_profit['Profit'] < product_profit['Profit'].mean())
    ]

    return render_template(
        'products.html',
        top_products=top_products.to_dict(),
        loss_products=loss_products.to_dict(),
        problem_products=
        high_sales_low_profit.head(10)
        .to_dict('records')
    )

@app.route('/regional')
def regional():

    state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)

    state_profit = df.groupby('State')['Profit'].sum().sort_values(ascending=False).head(10)

    city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(10)

    top_state = df.groupby('State')['Sales'].sum().idxmax()

    best_profit_state = df.groupby('State')['Profit'].sum().idxmax()

    worst_profit_state = df.groupby('State')['Profit'].sum().idxmin()

    sales_fig = px.bar(
        x=state_sales.values,
        y=state_sales.index,
        orientation='h',
        title='Top States by Sales'
    )

    profit_fig = px.bar(
        x=state_profit.values,
        y=state_profit.index,
        orientation='h',
        title='Top States by Profit'
    )

    apply_dark(sales_fig)
    apply_dark(profit_fig)

    geo_df = df.groupby('State').agg({
        'Sales':'sum',
        'Profit':'sum'
    }).reset_index()

    state_coords = {
        "California": (36.7783, -119.4179),
        "Texas": (31.9686, -99.9018),
        "New York": (43.0000, -75.0000),
        "Washington": (47.7511, -120.7401),
        "Florida": (27.6648, -81.5158),
        "Illinois": (40.0000, -89.0000),
        "Pennsylvania": (41.2033, -77.1945),
        "Ohio": (40.4173, -82.9071),
        "Michigan": (44.3148, -85.6024),
        "North Carolina": (35.7596, -79.0193)
    }

    geo_df = geo_df[
        geo_df["State"].isin(state_coords.keys())
    ]

    geo_df["lat"] = geo_df["State"].map(
        lambda x: state_coords[x][0]
    )

    geo_df["lon"] = geo_df["State"].map(
        lambda x: state_coords[x][1]
    )

    geo_fig = px.scatter_geo(
        geo_df,
        lat="lat",
        lon="lon",
        size="Sales",
        color="Profit",
        hover_name="State",
        hover_data=["Sales","Profit"],
        scope="usa",
        title="Interactive Geo Intelligence Map"
    )

    geo_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600
    )

    return render_template(
        'regional.html',
        sales_chart=sales_fig.to_html(full_html=False),
        profit_chart=profit_fig.to_html(full_html=False),
        geo_map=geo_fig.to_html(full_html=False),
        top_state=top_state,
        best_profit_state=best_profit_state,
        worst_profit_state=worst_profit_state,
        city_sales=city_sales.to_dict()
    )

@app.route('/executive')
def executive():

    total_sales = df['Sales'].sum()

    total_profit = df['Profit'].sum()

    profit_margin = round(
        (total_profit / total_sales) * 100,
        2
    )

    top_category = (
        df.groupby('Category')['Sales']
        .sum()
        .idxmax()
    )

    worst_category = (
        df.groupby('Category')['Profit']
        .sum()
        .idxmin()
    )

    top_region = (
        df.groupby('Region')['Sales']
        .sum()
        .idxmax()
    )

    worst_region = (
        df.groupby('Region')['Profit']
        .sum()
        .idxmin()
    )

    health_score = 85

    if profit_margin > 12:
        health_score += 5

    if total_profit > 250000:
        health_score += 5

    health_score = min(
        health_score,
        100
    )

    return render_template(
        "executive.html",
        health_score=health_score,
        profit_margin=profit_margin,
        top_category=top_category,
        worst_category=worst_category,
        top_region=top_region,
        worst_region=worst_region
    )

@app.route('/anomalies')
def anomalies():

    abnormal = anomaly_df[
        anomaly_df['Anomaly'] == -1
    ]

    total_anomalies = len(abnormal)

    top_anomalies = abnormal[
        [
            'Order ID',
            'Sales',
            'Profit',
            'Discount',
            'Category'
        ]
    ].head(25)

    # Sales distribution chart

    fig = px.scatter(
        anomaly_df,
        x='Sales',
        y='Profit',
        color='Anomaly',
        title='Anomaly Detection'
    )

    apply_dark(fig)

    return render_template(
        'anomalies.html',
        total_anomalies=total_anomalies,
        anomalies_table=
        top_anomalies.to_dict('records'),
        anomaly_chart=
        fig.to_html(full_html=False)
    )

if __name__ == '__main__':
    app.run(debug=True)
