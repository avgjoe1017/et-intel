#!/usr/bin/env python3
"""
ET Social Intelligence - Streamlit Dashboard
Interactive web dashboard for exploring sentiment and entity data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from et_intel.core.ingestion import CommentIngester
from et_intel import config

# Page config
st.set_page_config(
    page_title="ET Social Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 ET Social Intelligence Dashboard")
st.markdown("**Transform social comments into strategic market intelligence**")

# Initialize ingester
@st.cache_resource
def get_ingester():
    return CommentIngester()

ingester = get_ingester()

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    df = ingester.load_all_processed()
    if len(df) == 0:
        return pd.DataFrame()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

if len(df) == 0:
    st.warning("⚠️ No processed data found. Please process some CSV files first using the CLI.")
    st.info("Run: `python -m et_intel.cli.cli --batch` to process CSVs")
    st.stop()

# Entity search filter
st.sidebar.subheader("Entity Search")
entity_search = st.sidebar.text_input(
    "Search entities in comments",
    placeholder="e.g., Taylor Swift, Travis Kelce..."
)

# Platform filter
platforms = df['platform'].unique().tolist() if 'platform' in df.columns else []
selected_platforms = st.sidebar.multiselect(
    "Platform",
    options=platforms,
    default=platforms
)

# Date range filter
if 'timestamp' in df.columns and len(df) > 0:
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(max_date - timedelta(days=30), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

# Apply platform filter
if selected_platforms:
    df = df[df['platform'].isin(selected_platforms)]

# Apply entity search filter
if entity_search:
    search_terms = [term.strip() for term in entity_search.split(',')]
    mask = pd.Series([False] * len(df))
    for term in search_terms:
        if term:
            mask |= df['comment_text'].astype(str).str.contains(term, case=False, na=False)
            # Also search in post_subject and post_caption if available
            if 'post_subject' in df.columns:
                mask |= df['post_subject'].astype(str).str.contains(term, case=False, na=False)
            if 'post_caption' in df.columns:
                mask |= df['post_caption'].astype(str).str.contains(term, case=False, na=False)
    df = df[mask]
    if len(df) > 0:
        st.sidebar.success(f"Found {len(df)} comments matching search")
    else:
        st.sidebar.warning("No comments found matching search")

# Week-over-week comparison
def calculate_week_over_week(df):
    """Calculate week-over-week metrics"""
    if 'timestamp' not in df.columns or len(df) == 0:
        return None
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['week'] = pd.to_datetime(df['date']) - pd.to_timedelta(pd.to_datetime(df['date']).dt.dayofweek, unit='D')
    
    current_week = df['week'].max()
    previous_week = current_week - timedelta(days=7)
    
    current_df = df[df['week'] == current_week]
    previous_df = df[df['week'] == previous_week]
    
    if len(current_df) == 0 or len(previous_df) == 0:
        return None
    
    metrics = {}
    if 'sentiment_score' in df.columns:
        current_sentiment = current_df['sentiment_score'].mean()
        previous_sentiment = previous_df['sentiment_score'].mean()
        metrics['sentiment'] = {
            'current': current_sentiment,
            'previous': previous_sentiment,
            'delta': current_sentiment - previous_sentiment,
            'pct_change': ((current_sentiment - previous_sentiment) / abs(previous_sentiment) * 100) if previous_sentiment != 0 else 0
        }
    
    metrics['comments'] = {
        'current': len(current_df),
        'previous': len(previous_df),
        'delta': len(current_df) - len(previous_df),
        'pct_change': ((len(current_df) - len(previous_df)) / len(previous_df) * 100) if len(previous_df) > 0 else 0
    }
    
    return metrics

wow_metrics = calculate_week_over_week(df)

# Main metrics with week-over-week
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_comments = len(df)
    if wow_metrics:
        delta_comments = wow_metrics['comments']['delta']
        st.metric(
            "Total Comments", 
            f"{current_comments:,}",
            delta=f"{delta_comments:+,} ({wow_metrics['comments']['pct_change']:+.1f}%)"
        )
    else:
        st.metric("Total Comments", f"{current_comments:,}")

with col2:
    if 'sentiment_score' in df.columns:
        avg_sentiment = df['sentiment_score'].mean()
        if wow_metrics and 'sentiment' in wow_metrics:
            delta_sentiment = wow_metrics['sentiment']['delta']
            st.metric(
                "Avg Sentiment", 
                f"{avg_sentiment:.2f}",
                delta=f"{delta_sentiment:+.2f} ({wow_metrics['sentiment']['pct_change']:+.1f}%)"
            )
        else:
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
    else:
        st.metric("Avg Sentiment", "N/A")

with col3:
    if 'platform' in df.columns:
        unique_platforms = df['platform'].nunique()
        st.metric("Platforms", unique_platforms)
    else:
        st.metric("Platforms", "N/A")

with col4:
    if 'post_id' in df.columns:
        unique_posts = df['post_id'].nunique()
        st.metric("Unique Posts", unique_posts)
    else:
        st.metric("Unique Posts", "N/A")

# Week-over-week summary
if wow_metrics:
    st.info(f"📊 **Week-over-Week**: Current week has {wow_metrics['comments']['current']:,} comments vs. {wow_metrics['comments']['previous']:,} last week ({wow_metrics['comments']['pct_change']:+.1f}% change)")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Sentiment Trends", "👥 Entities", "📊 Emotion Breakdown", "🔗 Relationships"])

# Tab 1: Sentiment Trends
with tab1:
    st.header("Sentiment Over Time")
    
    if 'timestamp' in df.columns and 'sentiment_score' in df.columns:
        # Daily sentiment trend with week-over-week comparison
        df['date'] = df['timestamp'].dt.date
        df['week'] = pd.to_datetime(df['date']) - pd.to_timedelta(pd.to_datetime(df['date']).dt.dayofweek, unit='D')
        
        daily_sentiment = df.groupby('date')['sentiment_score'].agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment', 'comment_count']
        
        # Calculate week-over-week comparison
        weekly_sentiment = df.groupby('week')['sentiment_score'].agg(['mean', 'count']).reset_index()
        weekly_sentiment.columns = ['week', 'avg_sentiment', 'comment_count']
        weekly_sentiment['week'] = weekly_sentiment['week'].dt.date
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['avg_sentiment'],
            mode='lines+markers',
            name='Daily Average Sentiment',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Add weekly average line
        if len(weekly_sentiment) > 1:
            fig.add_trace(go.Scatter(
                x=weekly_sentiment['week'],
                y=weekly_sentiment['avg_sentiment'],
                mode='lines+markers',
                name='Weekly Average',
                line=dict(color='#3498db', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Daily Average Sentiment (with Weekly Trend)",
            xaxis_title="Date",
            yaxis_title="Sentiment Score (-1 to +1)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Week-over-week comparison chart
        if len(weekly_sentiment) >= 2:
            st.subheader("Week-over-Week Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment comparison
                recent_weeks = weekly_sentiment.tail(4)  # Last 4 weeks
                fig_wow = go.Figure()
                fig_wow.add_trace(go.Bar(
                    x=recent_weeks['week'].astype(str),
                    y=recent_weeks['avg_sentiment'],
                    name='Avg Sentiment',
                    marker_color='#2ecc71'
                ))
                fig_wow.update_layout(
                    title="Weekly Average Sentiment (Last 4 Weeks)",
                    xaxis_title="Week",
                    yaxis_title="Sentiment Score",
                    height=300
                )
                st.plotly_chart(fig_wow, use_container_width=True)
            
            with col2:
                # Comment volume comparison
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=recent_weeks['week'].astype(str),
                    y=recent_weeks['comment_count'],
                    name='Comment Count',
                    marker_color='#3498db'
                ))
                fig_vol.update_layout(
                    title="Weekly Comment Volume (Last 4 Weeks)",
                    xaxis_title="Week",
                    yaxis_title="Number of Comments",
                    height=300
                )
                st.plotly_chart(fig_vol, use_container_width=True)
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df,
                x='sentiment_score',
                nbins=50,
                title="Sentiment Distribution",
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Comments'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            if 'primary_emotion' in df.columns:
                emotion_counts = df['primary_emotion'].value_counts()
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Emotion Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Sentiment data not available. Process data with sentiment analysis enabled.")

# Tab 2: Entities
with tab2:
    st.header("Top Entities")
    
    # Extract entities from comments
    if 'comment_text' in df.columns:
        # Show top mentioned entities (basic extraction)
        st.subheader("Entity Mentions")
        
        # Simple entity extraction from comments
        import re
        from collections import Counter
        
        all_text = ' '.join(df['comment_text'].astype(str))
        
        # Extract potential person names (First Last pattern)
        person_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        people_mentioned = re.findall(person_pattern, all_text)
        people_counter = Counter(people_mentioned)
        
        # Extract potential show/movie names (quoted or capitalized phrases)
        show_pattern = r'"([^"]+)"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        shows_mentioned = re.findall(show_pattern, all_text)
        shows_list = [s[0] if s[0] else s[1] for s in shows_mentioned if s[0] or s[1]]
        shows_counter = Counter(shows_list)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top People")
            if people_counter:
                top_people = people_counter.most_common(10)
                people_df = pd.DataFrame(top_people, columns=['Name', 'Mentions'])
                st.dataframe(people_df, use_container_width=True, hide_index=True)
            else:
                st.info("No person names detected")
        
        with col2:
            st.subheader("Top Shows/Movies")
            if shows_counter:
                top_shows = shows_counter.most_common(10)
                shows_df = pd.DataFrame(top_shows, columns=['Name', 'Mentions'])
                st.dataframe(shows_df, use_container_width=True, hide_index=True)
            else:
                st.info("No shows/movies detected")
        
        # Show filtered comments if entity search is active
        if entity_search:
            st.subheader(f"Comments Matching: '{entity_search}'")
            filtered_df = df[['comment_text', 'sentiment_score', 'timestamp', 'platform']].copy()
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            # Show sample comments
            st.subheader("Sample Comments")
            sample_size = st.slider("Number of comments to show", 5, 50, 10)
            sample_df = df[['comment_text', 'sentiment_score', 'timestamp', 'platform']].head(sample_size)
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Comment text not available in processed data.")

# Tab 3: Emotion Breakdown
with tab3:
    st.header("Emotion Analysis")
    
    if 'primary_emotion' in df.columns:
        # Emotion timeline
        df['date'] = df['timestamp'].dt.date if 'timestamp' in df.columns else None
        if df['date'].notna().any():
            emotion_timeline = df.groupby(['date', 'primary_emotion']).size().reset_index(name='count')
            fig_emotion = px.line(
                emotion_timeline,
                x='date',
                y='count',
                color='primary_emotion',
                title="Emotion Trends Over Time"
            )
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Emotion by platform
        if 'platform' in df.columns:
            emotion_platform = pd.crosstab(df['platform'], df['primary_emotion'])
            fig_heatmap = px.imshow(
                emotion_platform,
                labels=dict(x="Emotion", y="Platform", color="Count"),
                title="Emotion by Platform",
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Emotion data not available. Process data with emotion analysis enabled.")

# Tab 4: Relationships (NetworkX)
with tab4:
    st.header("Entity Relationships")
    
    st.info("Relationship graphs require entity extraction. Process data with the full pipeline to see relationship networks.")
    
    # Placeholder for NetworkX graph visualization
    # This would use the EntityExtractor to build relationship graphs
    st.markdown("""
    **Future Feature:**
    - Visual network graphs of entity relationships
    - Co-mention analysis
    - Storyline connections
    """)

# Generate Report button
st.sidebar.markdown("---")
st.sidebar.header("📄 Reports")

if st.sidebar.button("📥 Generate PDF Report", type="primary"):
    try:
        from et_intel.core.pipeline import IntelligencePipeline
        from et_intel.reporting.report_generator import ReportGenerator
        
        with st.spinner("Generating intelligence brief..."):
            # Initialize pipeline
            pipeline = IntelligencePipeline()
            
            # Generate report
            report_path = pipeline.generate_intelligence_brief()
            
            if report_path and Path(report_path).exists():
                # Read PDF and create download button
                with open(report_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                    st.sidebar.success("✅ Report generated!")
                    st.sidebar.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=Path(report_path).name,
                        mime="application/pdf"
                    )
            else:
                st.sidebar.error("Failed to generate report. Check logs for details.")
    except Exception as e:
        st.sidebar.error(f"Error generating report: {e}")
        st.sidebar.info("Make sure you have processed data and reportlab installed.")

# Footer
st.markdown("---")
st.markdown("**ET Social Intelligence System** | Built with Streamlit")
st.markdown(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

