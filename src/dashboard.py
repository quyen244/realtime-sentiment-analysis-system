"""
Streamlit Dashboard for ABSA Streaming System
Real-time visualization of sentiment analysis results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# Page configuration
st.set_page_config(
    page_title="ABSA Streaming Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sentiment mapping
SENTIMENT_MAP = {
    -1: 'None',
    0: 'Negative',
    1: 'Positive',
    2: 'Neutral'
}

SENTIMENT_COLORS = {
    'None': '#808080',      # Gray
    'Negative': '#FF4B4B',  # Red
    'Positive': '#00CC66',  # Green
    'Neutral': '#4B9FFF'    # Blue
}

ASPECTS = Config.ASPECT_LABELS

# Database connection
@st.cache_resource
def get_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='absa_db',
            user='postgres',
            password='1234'
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def fetch_data(query, params=None):
    """Execute query and return DataFrame"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

def get_sentiment_distribution(aspect, hours=1):
    """Get sentiment distribution for a specific aspect"""
    query = f"""
        SELECT 
            pred_{aspect.lower()} as sentiment,
            COUNT(*) as count
        FROM sentiment_analysis
        WHERE processed_at >= NOW() - INTERVAL '{hours} hours'
        GROUP BY pred_{aspect.lower()}
        ORDER BY sentiment
    """
    df = fetch_data(query)
    if not df.empty:
        df['sentiment_label'] = df['sentiment'].map(SENTIMENT_MAP)
    return df

def get_recent_reviews(limit=10):
    """Get most recent reviews with predictions"""
    query = f"""
        SELECT 
            review,
            pred_price, pred_shipping, pred_outlook, pred_quality,
            pred_size, pred_shop_service, pred_general, pred_others,
            processed_at
        FROM sentiment_analysis
        ORDER BY processed_at DESC
        LIMIT %s
    """
    return fetch_data(query, params=(limit,))

def get_time_series_data(hours=24):
    """Get time series data for sentiment trends"""
    query = f"""
        SELECT 
            DATE_TRUNC('minute', processed_at) as time,
            AVG(CASE WHEN pred_price >= 0 THEN pred_price ELSE NULL END) as avg_price,
            AVG(CASE WHEN pred_quality >= 0 THEN pred_quality ELSE NULL END) as avg_quality,
            AVG(CASE WHEN pred_general >= 0 THEN pred_general ELSE NULL END) as avg_general,
            COUNT(*) as review_count
        FROM sentiment_analysis
        WHERE processed_at >= NOW() - INTERVAL '%s hours'
        GROUP BY time
        ORDER BY time
    """
    return fetch_data(query, params=(hours,))

def get_overall_stats(hours=1):
    """Get overall statistics"""
    query = f"""
        SELECT 
            COUNT(*) as total_reviews,
            COUNT(CASE WHEN pred_general = 1 THEN 1 END) as positive_reviews,
            COUNT(CASE WHEN pred_general = 0 THEN 1 END) as negative_reviews,
            COUNT(CASE WHEN pred_general = 2 THEN 1 END) as neutral_reviews
        FROM sentiment_analysis
        WHERE processed_at >= NOW() - INTERVAL '{hours} hours'
    """
    return fetch_data(query)

def create_sentiment_pie_chart(df, title):
    """Create pie chart for sentiment distribution"""
    if df.empty:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=df['sentiment_label'],
        values=df['count'],
        marker=dict(colors=[SENTIMENT_COLORS[label] for label in df['sentiment_label']]),
        hole=0.4
    )])
    
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_time_series_chart(df):
    """Create time series chart for sentiment trends"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['avg_price'],
        name='Price', mode='lines+markers',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['avg_quality'],
        name='Quality', mode='lines+markers',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['avg_general'],
        name='General', mode='lines+markers',
        line=dict(color='#95E1D3', width=2)
    ))
    
    fig.update_layout(
        title='Sentiment Trends Over Time',
        xaxis_title='Time',
        yaxis_title='Average Sentiment',
        height=400,
        hovermode='x unified',
        yaxis=dict(range=[-1, 2])
    )
    
    return fig

def create_aspect_comparison(aspects_data):
    """Create bar chart comparing all aspects"""
    aspect_names = []
    positive_counts = []
    negative_counts = []
    neutral_counts = []
    
    for aspect, df in aspects_data.items():
        if not df.empty:
            aspect_names.append(aspect)
            pos = df[df['sentiment'] == 1]['count'].sum()
            neg = df[df['sentiment'] == 0]['count'].sum()
            neu = df[df['sentiment'] == 2]['count'].sum()
            
            positive_counts.append(pos)
            negative_counts.append(neg)
            neutral_counts.append(neu)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive',
        x=aspect_names,
        y=positive_counts,
        marker_color=SENTIMENT_COLORS['Positive']
    ))
    
    fig.add_trace(go.Bar(
        name='Negative',
        x=aspect_names,
        y=negative_counts,
        marker_color=SENTIMENT_COLORS['Negative']
    ))
    
    fig.add_trace(go.Bar(
        name='Neutral',
        x=aspect_names,
        y=neutral_counts,
        marker_color=SENTIMENT_COLORS['Neutral']
    ))
    
    fig.update_layout(
        title='Sentiment Distribution by Aspect',
        barmode='group',
        height=400,
        xaxis_title='Aspect',
        yaxis_title='Count'
    )
    
    return fig

# Main dashboard
def main():
    st.title("📊 ABSA Streaming Dashboard")
    st.markdown("Real-time Aspect-Based Sentiment Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, Config.DASHBOARD_REFRESH_INTERVAL)
        
        # Thêm tùy chọn "All Time" hoặc tăng giới hạn để dễ debug
        time_options = ["1 hour", "6 hours", "12 hours", "24 hours", "7 days", "30 days"]
        time_window = st.selectbox("Time window", time_options, index=3) 
        
        st.markdown("---")
        st.markdown("### System Info")
        st.info(f"Kafka Topic: {Config.KAFKA_TOPIC}")
        st.info(f"Database: {Config.POSTGRES_DB}")
    
    # Map time window sang số giờ
    hours_map = {
        "1 hour": 1, 
        "6 hours": 6, 
        "12 hours": 12, 
        "24 hours": 24,
        "7 days": 168,
        "30 days": 720
    }
    hours = hours_map.get(time_window, 24)
    
    # --- ÁP DỤNG THAM SỐ HOURS VÀO CÁC HÀM ---
    
    # Overall stats (Đã truyền hours)
    stats_df = get_overall_stats(hours)
    
    if not stats_df.empty and stats_df['total_reviews'].iloc[0] > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", stats_df['total_reviews'].iloc[0])
        with col2:
            st.metric("Positive", stats_df['positive_reviews'].iloc[0], 
                      delta_color="normal")
        with col3:
            st.metric("Negative", stats_df['negative_reviews'].iloc[0],
                      delta_color="inverse")
        with col4:
            st.metric("Neutral", stats_df['neutral_reviews'].iloc[0])
    else:
        st.warning(f"No data found in the last {time_window}. Try selecting a larger Time Window.")
    
    st.markdown("---")
    
    # Time series
    st.subheader("Sentiment Trends")
    ts_df = get_time_series_data(hours) # Hàm này đã đúng từ đầu

    # Debug: Hiển thị dataframe để kiểm tra nếu cần
    # st.write(f"Showing data for the last {hours} hours" , ts_df)
    
    if not ts_df.empty:
        st.plotly_chart(create_time_series_chart(ts_df), use_container_width=True)
    else:
        st.info("No time series data available yet")
    
    st.markdown("---")
    
    # Aspect comparison (Truyền hours vào vòng lặp)
    st.subheader("All Aspects Comparison")
    aspects_data = {aspect: get_sentiment_distribution(aspect, hours) for aspect in ASPECTS}
    
    if any(not df.empty for df in aspects_data.values()):
        st.plotly_chart(create_aspect_comparison(aspects_data), use_container_width=True)
    else:
        st.info("No aspect data available yet")
    
    st.markdown("---")
    
    # Individual aspect distributions (Truyền hours vào vòng lặp)
    st.subheader("Sentiment Distribution by Aspect")
    
    cols = st.columns(4)
    for idx, aspect in enumerate(ASPECTS):
        with cols[idx % 4]:
            df = get_sentiment_distribution(aspect, hours)

            if not df.empty:
                fig = create_sentiment_pie_chart(df, aspect)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {aspect} data")
    
    st.markdown("---")
    
    # Recent reviews table
    st.subheader("Recent Reviews")
    reviews_df = get_recent_reviews(limit=20)
    
    if not reviews_df.empty:
        # Create a display dataframe with sentiment labels
        display_df = reviews_df.copy()
        
        for aspect in ['price', 'shipping', 'outlook', 'quality', 'size', 'shop_service', 'general', 'others']:
            col_name = f'pred_{aspect}'
            if col_name in display_df.columns:
                display_df[col_name] = display_df[col_name].map(SENTIMENT_MAP)
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'review': 'Review',
            'pred_price': 'Price',
            'pred_shipping': 'Shipping',
            'pred_outlook': 'Outlook',
            'pred_quality': 'Quality',
            'pred_size': 'Size',
            'pred_shop_service': 'Service',
            'pred_general': 'General',
            'pred_others': 'Others',
            'processed_at': 'Time'
        })
        
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("No reviews available yet")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
