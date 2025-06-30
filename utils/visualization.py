import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_threat_distribution(threats_df):
    """
    Create a visualization of threat distribution by attack category
    
    Parameters:
    -----------
    threats_df : pd.DataFrame
        DataFrame containing detected threats with 'attack_cat' column
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Group by attack category
    if 'attack_cat' in threats_df.columns:
        category_col = 'attack_cat'
    elif 'predicted_attack_cat' in threats_df.columns:
        category_col = 'predicted_attack_cat'
    else:
        # Fall back to other columns that might contain categories
        for possible_col in ['category', 'type', 'classification']:
            if possible_col in threats_df.columns:
                category_col = possible_col
                break
        else:
            # If no suitable column is found, return empty figure
            return go.Figure()
    
    category_counts = threats_df[category_col].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    # Create pie chart
    fig = px.pie(
        category_counts,
        values='Count',
        names='Category',
        title='Threat Distribution by Category',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.4
    )
    
    # Add source IP information if available
    if 'src_ip' in threats_df.columns:
        top_ips = threats_df['src_ip'].value_counts().head(5).reset_index()
        top_ips.columns = ['Source IP', 'Count']
        
        fig_ips = px.bar(
            top_ips,
            x='Source IP',
            y='Count',
            title='Top 5 Source IPs',
            color='Count',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        
        # Update layout
        fig.update_layout(
            title_text='Threat Analysis',
            annotations=[dict(text='Categories', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig_ips
    
    return fig

def create_user_behavior_charts(user_df):
    """
    Create visualizations for user behavior analysis
    
    Parameters:
    -----------
    user_df : pd.DataFrame
        DataFrame containing user behavior data with 'user_id' and 'prediction' columns
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Anomaly Distribution', 'Top Users by Anomaly Score')
    )
    
    # Pie chart of normal vs anomalous users
    if 'prediction' in user_df.columns:
        prediction_counts = user_df['prediction'].value_counts().reset_index()
        prediction_counts.columns = ['Prediction', 'Count']
        
        # Map prediction values to labels
        prediction_counts['Label'] = prediction_counts['Prediction'].map({
            -1: 'Anomalous',
            1: 'Normal'
        })
        
        pie = go.Pie(
            labels=prediction_counts['Label'],
            values=prediction_counts['Count'],
            hole=0.4,
            marker_colors=['#FF6B6B', '#4ECDC4']
        )
        fig.add_trace(pie, row=1, col=1)
        
        # Bar chart of top anomalous users
        if 'user_id' in user_df.columns and 'anomaly_score' in user_df.columns:
            anomalous_users = user_df[user_df['prediction'] == -1]
            if not anomalous_users.empty:
                top_users = anomalous_users.sort_values('anomaly_score', ascending=True).head(5)
                
                bar = go.Bar(
                    x=top_users['user_id'],
                    y=top_users['anomaly_score'],
                    marker_color='#FF6B6B'
                )
                fig.add_trace(bar, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text='User Behavior Analysis',
        showlegend=False,
        height=500
    )
    
    return fig

def create_anomaly_charts(logs_df):
    """
    Create visualizations for anomaly detection in log analysis
    
    Parameters:
    -----------
    logs_df : pd.DataFrame
        DataFrame containing log analysis results
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Anomaly Categories', 'Confidence Distribution')
    )
    
    # Filter for anomalous logs
    if 'category' in logs_df.columns and 'confidence' in logs_df.columns:
        anomalous_logs = logs_df[logs_df['category'] != 'Normal']
        
        if not anomalous_logs.empty:
            # Pie chart of anomaly categories
            category_counts = anomalous_logs['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            pie = go.Pie(
                labels=category_counts['Category'],
                values=category_counts['Count'],
                hole=0.4,
                marker_colors=px.colors.qualitative.Bold
            )
            fig.add_trace(pie, row=1, col=1)
            
            # Bar chart of confidence distribution
            confidence_bins = [0, 1, 2, 3]
            confidence_labels = ['Low', 'Medium', 'High', 'Critical']
            
            anomalous_logs['confidence_bin'] = pd.cut(
                anomalous_logs['confidence'],
                bins=[0, 1, 2, 3, 4],
                labels=confidence_labels,
                include_lowest=True
            )
            
            confidence_counts = anomalous_logs['confidence_bin'].value_counts().reset_index()
            confidence_counts.columns = ['Confidence', 'Count']
            confidence_counts = confidence_counts.sort_values('Confidence', key=lambda x: pd.Categorical(
                x, categories=confidence_labels, ordered=True
            ))
            
            bar = go.Bar(
                x=confidence_counts['Confidence'],
                y=confidence_counts['Count'],
                marker_color=px.colors.sequential.Plasma_r
            )
            fig.add_trace(bar, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text='Log Anomaly Analysis',
        showlegend=False,
        height=500
    )
    
    return fig

def create_attack_pie_chart(attack_data: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart of attack categories
    
    Parameters:
    -----------
    attack_data : pd.DataFrame
        DataFrame with attack classification results
        
    Returns:
    --------
    go.Figure
        Plotly figure object with pie chart
    """
    if 'attack_cat' not in attack_data.columns:
        return go.Figure().add_annotation(
            text="No 'attack_cat' column available",
            showarrow=False
    )
    # Count attack categories
    attack_counts = attack_data['attack_cat'].value_counts().reset_index()
    attack_counts.columns = ['Category', 'Count']
    
    # Create pie chart
    fig = px.pie(
        attack_counts, 
        values='Count', 
        names='Category',
        title='Attack Category Distribution',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.4,
    )
    
    # Update layout
    fig.update_layout(
        legend_title_text='Attack Categories',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Add total count annotation in the center
    total_attacks = attack_counts['Count'].sum()
    fig.add_annotation(
        text=f"Total:<br>{total_attacks}",
        x=0.5, y=0.5,
        font_size=20,
        showarrow=False
    )
    
    return fig

def create_time_series(data: pd.DataFrame) -> go.Figure:
    """
    Create a time series chart of attacks or events
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with timestamp and attack/event classification
        
    Returns:
    --------
    go.Figure
        Plotly figure object with time series chart
    """
    # Check if timestamp column exists
    if 'timestamp' not in data.columns:
        # Create a fake timestamp for visualization
        data = data.copy()
        data['timestamp'] = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=1),
            periods=len(data),
            freq='min'
        )
    
    # Group by timestamp and attack category
    if 'attack_cat' in data.columns:
        category_col = 'attack_cat'
    elif 'is_anomaly' in data.columns:
        # For anomaly detection results
        data = data.copy()
        data['anomaly_status'] = data['is_anomaly'].map({True: 'Anomaly', False: 'Normal'})
        category_col = 'anomaly_status'
    else:
        # Generic time series
        data = data.copy()
        data['event_type'] = 'Event'
        category_col = 'event_type'
    
    # Resample to 5-minute intervals for better visualization
    try:
        time_data = data.set_index('timestamp')
        time_data = time_data.groupby([pd.Grouper(freq='5min'), category_col]).size().reset_index()
        time_data.columns = ['timestamp', category_col, 'count']
    except:
        # Fallback if resampling fails
        time_data = data.groupby([category_col])['timestamp'].count().reset_index()
        time_data.columns = [category_col, 'count']
        time_data['timestamp'] = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=1),
            periods=len(time_data),
            freq='hour'
        )
    
    # Create time series chart
    fig = px.line(
        time_data,
        x='timestamp',
        y='count',
        color=category_col,
        title='Event Activity Over Time',
        labels={'count': 'Event Count', 'timestamp': 'Time'},
        line_shape='linear',
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Count',
        legend_title=category_col,
        hovermode='x unified'
    )
    
    return fig

def create_heatmap(data: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap of user activity patterns
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with user activity data
        
    Returns:
    --------
    go.Figure
        Plotly figure object with heatmap
    """
    # Check if necessary columns exist
    if 'user_id' not in data.columns or 'timestamp' not in data.columns:
        # Create sample heatmap data
        user_ids = [f"User {i}" for i in range(1, 11)]
        hours = list(range(24))
        activity = np.random.randint(0, 10, size=(len(user_ids), len(hours)))
        heatmap_data = pd.DataFrame(
            activity,
            index=user_ids,
            columns=hours
        )
    else:
        # Extract hour from timestamp
        data = data.copy()
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        
        # Group by user_id and hour
        heatmap_data = data.groupby(['user_id', 'hour']).size().unstack(fill_value=0)
        
        # Fill missing hours
        for hour in range(24):
            if hour not in heatmap_data.columns:
                heatmap_data[hour] = 0
        
        # Sort columns
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
        
        # Limit to top 15 users for better visualization
        if len(heatmap_data) > 15:
            user_totals = heatmap_data.sum(axis=1).sort_values(ascending=False)
            top_users = user_totals.head(15).index
            heatmap_data = heatmap_data.loc[top_users]
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="User", color="Activity Count"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title="User Activity Patterns by Hour",
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="User ID",
        coloraxis_colorbar=dict(
            title="Activity Count",
        )
    )
    
    return fig