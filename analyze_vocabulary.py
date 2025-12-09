import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from datetime import timedelta

# --- Configuration ---
OUTPUT_DIR = "vocabulary_report"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_HTML = os.path.join(OUTPUT_DIR, "vocabulary_analysis_report.html")

# Color Palette
COLORS = {
    'male': '#3498db',      # Blue
    'female': '#e91e63',    # Pink
    'neuter': '#f1c40f',    # Yellow
    'n/a': '#95a5a6',       # Grey
    'main': '#2c3e50',      # Dark Slate
    'accent': '#1abc9c',    # Turquoise
    'gap': '#ecf0f1',       # Light Grey for gaps
    'gap_text': '#95a5a6',  # Grey for text
    'streak': '#2ecc71',    # Green
    'shading': 'rgba(231, 76, 60, 0.1)' # Light red shading for gaps
}

# Red Color Scale (Light Red to Dark Red)
COLOR_SCALE_REDS = [[0, '#ffebee'], [1.0, '#8b0000']] # Light Red to Dark Red

# --- Helper Functions ---

def get_plot_div(fig):
    # Enable responsive to fix resize issues
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False, 'responsive': True})

def fmt_pct(num, total):
    if total == 0: return "0%"
    return f"{(num/total)*100:.1f}%"

def fmt_date(dt):
    return dt.strftime('%d.%m.%Y')

# HTML Builder

def generate_report(user_data, data_dir='data', utc_offset=0):
    print(f">>> LOADING DATA... (Offset: {utc_offset}h)")

    # Parse User Data
    user_items = []
    # Handle if user_data is dict (from JSON) or list (direct result)
    result_list = user_data.get("result", []) if isinstance(user_data, dict) else user_data

    
    def parse_date(ts):
        if ts is None: return pd.NaT
        try:
            # Check if unix timestamp (int/float)
            if isinstance(ts, (int, float)):
                 # Heuristic: if very large, it's ms, else s (OpenRussian uses s)
                 if ts > 30000000000: # > year 2920 if seconds, so clearly ms
                     return pd.to_datetime(ts, unit='ms')
                 return pd.to_datetime(ts, unit='s')
            return pd.to_datetime(ts)
        except:
            return pd.NaT

    for item in result_list:
        added_ts = item.get("added") # usage .get is safer
        word_info = item["word"]
        user_items.append({
            "added": parse_date(added_ts),
            "word_id": word_info["id"],
            "word_text": word_info["word"],
            "user_type_label": word_info.get("type", "unknown"),
            "level": word_info.get("level")
        })
    df_user = pd.DataFrame(user_items)
    df_user = df_user.sort_values("added")
    
    # Apply Timezone Offset
    if utc_offset != 0:
        df_user['added'] = df_user['added'] + pd.Timedelta(hours=utc_offset)

    # Load Reference CSVs
    print("Loading reference CSVs...")
    try:
        df_words = pd.read_csv(os.path.join(data_dir, "russian3 - words.csv"))
        # FILTER DISABLED WORDS
        df_words = df_words[df_words['disabled'] != 1]
        
        df_nouns = pd.read_csv(os.path.join(data_dir, "russian3 - nouns.csv"))
        df_verbs = pd.read_csv(os.path.join(data_dir, "russian3 - verbs.csv"))
        df_rels = pd.read_csv(os.path.join(data_dir, "russian3 - words_rels.csv"))
    except Exception as e:
        return f"<h3>Error loading data files: {str(e)}</h3><p>Ensure CSV files are in the '{data_dir}' directory.</p>"

    # Merge
    df_combined = pd.merge(df_user, df_words[['id', 'rank', 'type', 'bare', 'accented', 'usage_en', 'level']], left_on='word_id', right_on='id', how='left')
    # Fill level from user data if missing in CSV (though CSV is master)
    df_combined['level'] = df_combined['level_y'].fillna(df_combined['level_x'])
    df_combined['type'] = df_combined['type'].fillna(df_combined['user_type_label'])
    html_sections = []
    
    def add_section(title, content):
        html_sections.append(f"""
        <div class="card">
            <h2>{title}</h2>
            {content}
        </div>
        """)
    
    # =========================================================
    # 0. HEADER STATISTICS
    # =========================================================
    print(">>> Computing Overview Stats...")
    
    total_words = len(df_combined)
    # Filter out NaT values for date calculations
    df_combined_valid_dates = df_combined.dropna(subset=['added'])
    first_day = df_combined_valid_dates['added'].min()
    last_day = df_combined_valid_dates['added'].max()
    total_days = (last_day - first_day).days + 1 if not df_combined_valid_dates.empty else 0
    
    # Average Word Length (Exclude multi-words)
    single_words = df_combined[~df_combined['word_text'].str.contains(' ', na=False)]
    avg_len = single_words['word_text'].str.len().mean()
    
    # Words / Day
    words_per_day = len(df_combined_valid_dates) / total_days if total_days > 0 else 0
    
    # Active Days
    unique_days = df_combined_valid_dates['added'].dt.floor('d').nunique()
    active_pct = fmt_pct(unique_days, total_days)
    inactive_days = total_days - unique_days
    inactive_pct = fmt_pct(inactive_days, total_days)
    
    # Expanded Reach (only counting 'related' relations for consistency with Section 7)
    known_ids = set(df_combined['word_id'])
    related_rows = df_rels[(df_rels['word_id'].isin(known_ids)) & (df_rels['relation'] == 'related')]
    potential_ids = set(related_rows['rel_word_id'].unique())
    valid_ref_ids = set(df_words['id'])
    potential_ids = potential_ids.intersection(valid_ref_ids)
    new_potential = potential_ids - known_ids
    expanded_vocab = total_words + len(new_potential)
    
    
    header_html = f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-label">Total Words Saved</div>
            <div class="stat-value">{total_words:,}</div>
            <div class="stat-sub">{words_per_day:.1f} words / day</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Expanded Vocabulary Reach</div>
            <div class="stat-value">{expanded_vocab:,}</div>
            <div class="stat-sub">+{len(new_potential):,} related words</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Avg. Word Length</div>
            <div class="stat-value">{avg_len:.1f} <span style="font-size:1rem;">chars</span></div>
            <div class="stat-sub">(single words only)</div>
        </div>
    </div>
    """

    # Check for missing dates
    missing_dates = df_combined['added'].isna().sum()
    if missing_dates > 0:
        header_html += f"""
    <div style="background:#fff3cd; color:#856404; padding:15px; border-radius:8px; margin-top:20px; border:1px solid #ffeeba;">
        <strong>⚠️ Note:</strong> {missing_dates:,} words have no 'added' date information in your file. 
        These words are excluded from time-based charts (Timeline, Hours, Heatmap), but are included in total counts and vocabulary analysis.
    </div>
    """

    header_html += f"""
    <div class="stat-grid" style="margin-top:1rem;">
        <div class="stat-box">
            <div class="stat-label">Time Learning</div>
            <div class="stat-value">{total_days} <span style="font-size:1rem;">days</span></div>
            <div class="stat-sub">{fmt_date(first_day)} - {fmt_date(last_day)}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Active Learning Days</div>
            <div class="stat-value" style="color:{COLORS['streak']}">{unique_days}</div>
            <div class="stat-sub">{active_pct} of total time</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Inactive Days</div>
            <div class="stat-value" style="color:{COLORS['gap_text']}">{inactive_days}</div>
            <div class="stat-sub">{inactive_pct} of total time</div>
        </div>
    </div>
    """
    
    # =========================================================
    # 1. TIME-BASED ANALYTICS
    # =========================================================
    print(">>> 1. Time Analytics...")
    
    # 1.1 Timeline
    df_combined['date'] = df_combined['added'].dt.floor('d')
    daily_counts = df_combined.groupby('date').size().reset_index(name='count')
    # Reindex
    all_days_idx = pd.date_range(first_day.floor('d'), last_day.floor('d'))
    daily_counts = daily_counts.set_index('date').reindex(all_days_idx, fill_value=0).reset_index().rename(columns={'index': 'date'})
    daily_counts['cumulative'] = daily_counts['count'].cumsum()
    daily_counts['MA7'] = daily_counts['count'].rolling(window=7).mean()
    
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=daily_counts['date'].tolist(), y=daily_counts['cumulative'].tolist(), name='Total Words', fill='tozeroy', line=dict(color=COLORS['male'])))
    fig_tl.add_trace(go.Scatter(x=daily_counts['date'].tolist(), y=daily_counts['MA7'].tolist(), name='7-Day Avg', yaxis='y2', line=dict(color=COLORS['female'], width=2, dash='dot')))
    
    # Calculate streaks for stats only
    is_active = daily_counts['count'] > 0
    streak_groups = is_active.ne(is_active.shift()).cumsum()
    streaks_df = daily_counts.groupby(streak_groups).apply(
        lambda x: pd.Series({
            'start': x['date'].min(), 
            'end': x['date'].max(), 
            'days': len(x), 
            'is_streak': x['count'].iloc[0] > 0
        })
    ).reset_index(drop=True)
    
    # Add Milestone Markers
    MILESTONES = [500, 1000, 2000, 5000, 10000, 20000]
    milestone_stats = []
    
    # Use the first date from daily_counts for consistency
    start_date = daily_counts['date'].min()
    
    for milestone in MILESTONES:
        # Find the first date when cumulative count reached this milestone
        milestone_rows = daily_counts[daily_counts['cumulative'] >= milestone]
        if not milestone_rows.empty:
            milestone_date = milestone_rows.iloc[0]['date']
            milestone_count = milestone_rows.iloc[0]['cumulative']
            
            # Add vertical line using add_shape
            fig_tl.add_shape(
                type="line",
                x0=milestone_date, x1=milestone_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=COLORS['accent'], width=2, dash="dot")
            )
            
            # Add annotation at top
            fig_tl.add_annotation(
                x=milestone_date,
                y=1,
                yref="paper",
                text=f"{milestone:,}",
                showarrow=False,
                font=dict(size=10, color=COLORS['accent']),
                yshift=10
            )
            
            # Calculate stats using start_date for consistency
            days_from_start = (milestone_date - start_date).days
            milestone_stats.append({
                'milestone': milestone,
                'date': milestone_date,
                'days_total': days_from_start
            })
    
    # Calculate pace for each segment
    milestone_html = "<h4>Milestone Achievements</h4>"
    milestone_html += "<table class='freq-table'><thead><tr><th>Milestone</th><th>Date</th><th>Total Days</th><th>Period Days</th><th>Words/Day</th></tr></thead><tbody>"
    
    prev_milestone = 0
    prev_days = 0
    
    for stat in milestone_stats:
        m = stat['milestone']
        date_str = fmt_date(stat['date'])
        total_days = stat['days_total']
        
        # Period calculation
        period_days = total_days - prev_days
        period_words = m - prev_milestone
        pace = period_words / period_days if period_days > 0 else 0
        
        milestone_html += f"<tr><td><b>{m:,}</b></td><td>{date_str}</td><td>{total_days}</td><td>{period_days}</td><td>{pace:.1f}</td></tr>"
        
        prev_milestone = m
        prev_days = total_days
    
    # Add projection for next milestone
    if milestone_stats:
        last_milestone_stat = milestone_stats[-1]
        last_milestone = last_milestone_stat['milestone']
        last_milestone_date = last_milestone_stat['date']
        
        # Find next milestone that hasn't been reached
        next_milestone = None
        for m in MILESTONES:
            if m > total_words:
                next_milestone = m
                break
        
        if next_milestone:
            # Calculate current pace (since last milestone)
            days_since_last = (last_day - last_milestone_date).days
            words_since_last = total_words - last_milestone
            current_pace = words_since_last / days_since_last if days_since_last > 0 else 0
            
            if current_pace > 0:
                words_needed = next_milestone - total_words
                days_needed = words_needed / current_pace
                projected_date = last_day + timedelta(days=int(days_needed))
                projected_total_days = (projected_date - start_date).days
                projected_period_days = projected_total_days - prev_days
                
                # Format: Date (in ~X days)
                date_with_estimate = f"{fmt_date(projected_date)} <span style='color:#7f8c8d;'>(in ~{int(days_needed)} days)</span>"
                
                milestone_html += f"<tr style='background:#f0f9ff; border-top:2px solid {COLORS['accent']};'><td><b>{next_milestone:,}</b></td><td><i>{date_with_estimate}</i></td><td><i>~{projected_total_days}</i></td><td><i>~{projected_period_days}</i></td><td><i>{current_pace:.1f}</i></td></tr>"
                milestone_html += f"<tr style='background:#f0f9ff;'><td colspan='5' style='text-align:center; color:{COLORS['accent']}; font-size:0.85rem;'><i>↑ Projected based on current pace ({current_pace:.1f} words/day since last milestone)</i></td></tr>"
    
    milestone_html += "</tbody></table>"
    
    fig_tl.update_layout(
        title="Cumulative Learning Curve",
        yaxis=dict(title='Total Words', rangemode='tozero'),
        yaxis2=dict(title='Words/Day', overlaying='y', side='right', showgrid=False, rangemode='tozero'),
        template='plotly_white', height=400, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    
    # 1.2 GitHub Heatmap (Per Year)
    years = sorted(daily_counts['date'].dt.year.unique())
    gh_maps_html = ""
    
    for y in years:
        daily_year = daily_counts[daily_counts['date'].dt.year == y].copy()
        if daily_year.empty: continue
        
        daily_year['week'] = daily_year['date'].dt.isocalendar().week
        daily_year['weekday'] = daily_year['date'].dt.weekday # 0=Mon
        
        heatmap_data = daily_year.pivot_table(index='weekday', columns='week', values='count', fill_value=0)
        
        weekday_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
        heatmap_data.index = heatmap_data.index.map(weekday_map)
        heatmap_data = heatmap_data.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Fill range 1-52 (Drop 53 as requested)
        for w in range(1, 53):
            if w not in heatmap_data.columns:
                heatmap_data[w] = 0
        # Filter columns to only 1-52
        cols_to_keep = [c for c in heatmap_data.columns if 1 <= c <= 52]
        heatmap_data = heatmap_data[cols_to_keep]
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
    
        fig_gh = go.Figure(data=go.Heatmap(
            z=heatmap_data.values.tolist(),
            x=list(heatmap_data.columns),
            y=heatmap_data.index.tolist(),
            colorscale=[[0, '#ffffff'], [0.01, '#d6e9c6'], [1.0, '#216e39']], 
            showscale=False,
            xgap=2, ygap=2
        ))
        fig_gh.update_layout(
            title=f"Activity Map {y}",
            height=180,
            yaxis=dict(showgrid=False, zeroline=False, automargin=True),
            xaxis=dict(showgrid=False, zeroline=False, title="Week of Year", tickmode='linear', tick0=1, dtick=4),
            margin=dict(l=40, r=20, t=40, b=20)
        )
        gh_maps_html += get_plot_div(fig_gh)
    
    # Text Stats
    top_streaks = streaks_df[streaks_df['is_streak']].sort_values('days', ascending=False).head(3)
    top_gaps = streaks_df[~streaks_df['is_streak']].sort_values('days', ascending=False).head(3)
    def fmt_range(r): return f"{fmt_date(r['start'])} - {fmt_date(r['end'])}"
    streak_html = "<ul>" + "".join([f"<li><b>{row['days']} days</b> ({fmt_range(row)})</li>" for _, row in top_streaks.iterrows()]) + "</ul>"
    gap_html = "<ul>" + "".join([f"<li><b>{row['days']} days</b> ({fmt_range(row)})</li>" for _, row in top_gaps.iterrows()]) + "</ul>"
    
    
    # 1.3 Habits (Stacked)
    # Hour Stats
    df_combined['hour'] = df_combined['added'].dt.hour
    hour_counts = df_combined['hour'].value_counts().sort_index().reindex(range(24), fill_value=0)
    best_h = hour_counts.idxmax()
    worst_h = hour_counts[hour_counts > 0].idxmin() if (hour_counts > 0).any() else 0
    
    hour_ticks = list(range(0, 24, 2))
    hour_labels = [f"{h}:00" for h in hour_ticks]
    def fmt_hour_range(h): return f"{h}:00 - {(h+1)%24}:00"
    
    fig_hour = go.Figure(go.Bar(
        x=[int(x) for x in hour_counts.index], 
        y=hour_counts.values.tolist(),
        marker=dict(
            color=hour_counts.values.tolist(),
            colorscale=COLOR_SCALE_REDS,
            cmin=0,
            cmax=int(hour_counts.max())
        )
    ))
    fig_hour.update_layout(
        title="Words added by Hour",
        yaxis_title="Words",
        xaxis_title="Hour",
        height=350, 
        showlegend=False,
        xaxis=dict(tickmode='array', tickvals=hour_ticks, ticktext=hour_labels),
        margin=dict(l=20, r=20, t=40, b=40)
    )
    
    # Weekday Stats
    df_combined['weekday'] = df_combined['added'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df_combined['weekday'].value_counts().reindex(days_order, fill_value=0)
    best_d = day_counts.idxmax()
    worst_d = day_counts.idxmin()
    
    fig_day = go.Figure(go.Bar(
        x=day_counts.index.tolist(), 
        y=day_counts.values.tolist(),
        marker=dict(
            color=day_counts.values.tolist(),
            colorscale=COLOR_SCALE_REDS,
            cmin=0,
            cmax=int(day_counts.max())
        )
    ))
    fig_day.update_layout(
        title="Words added by Weekday",
        yaxis_title="Words",
        xaxis_title="Day",
        height=350, 
        showlegend=False, 
        margin=dict(l=20, r=20, t=40, b=40)
    )
    
    # Month vs Year Heatmap (Axes Swapped: X=Month, Y=Year)
    df_combined['year'] = df_combined['added'].dt.year
    df_combined['month_name'] = df_combined['added'].dt.month_name()
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    # Group by Year, Month for HEATMAP
    heatmap_my = df_combined.groupby(['year', 'month_name']).size().unstack(fill_value=0)
    heatmap_my = heatmap_my.reindex(columns=months_order)
    # Ensure all years are present as rows (Index)
    all_years = sorted(df_combined['year'].unique())
    heatmap_my = heatmap_my.reindex(all_years, fill_value=0)
    # Convert index (Year) to string
    heatmap_my.index = heatmap_my.index.astype(str)
    
    fig_my = go.Figure(go.Heatmap(
        z=heatmap_my.values.tolist(), 
        x=list(heatmap_my.columns), 
        y=heatmap_my.index.tolist(),
        colorscale=COLOR_SCALE_REDS,
        zmin=0,
        zmax=int(heatmap_my.max().max())
    ))
    fig_my.update_layout(
        title="Month vs Year Heatmap", 
        yaxis_title="Year",
        xaxis_title="Month",
        height=450
    )
    
    # Stats for Specific Month (Year + Month)
    # Use the grouped dataframe before unstacking, or just group again
    monthly_specific = df_combined.groupby(['year', 'month_name']).size().reset_index(name='count')
    monthly_specific = monthly_specific.sort_values('count', ascending=False)
    
    if not monthly_specific.empty:
        best_row = monthly_specific.iloc[0]
        worst_row = monthly_specific.iloc[-1]
        
        best_m_str = f"{best_row['month_name']} {best_row['year']} ({best_row['count']} words)"
        worst_m_str = f"{worst_row['month_name']} {worst_row['year']} ({worst_row['count']} words)"
    else:
        best_m_str = "N/A"
        worst_m_str = "N/A"
    
    # 1.4 Difficulty Timeline (Rank Evolution)
    # Resample to monthly average rank
    df_rank_evo = df_combined.set_index('added').resample('M')['rank'].mean().reset_index()
    # Filter out NaNs if any empty months
    df_rank_evo = df_rank_evo.dropna()
    
    fig_rank = go.Figure()
    fig_rank.add_trace(go.Scatter(
        x=df_rank_evo['added'].tolist(), 
        y=df_rank_evo['rank'].tolist(), 
        mode='lines+markers',
        name='Avg Rank',
        line=dict(color=COLORS['male'], width=3, shape='spline'), # 'spline' for smoothing
        marker=dict(size=8)
    ))
    fig_rank.update_layout(
        title="Average Word Difficulty per Month (Higher Rank = Rarer Word)",
        yaxis=dict(title='Avg Frequency Rank', rangemode='tozero'),
        xaxis=dict(title='Month'),
        template='plotly_white', height=350, margin=dict(l=40, r=40, t=40, b=40)
    )
    
    
    analysis_1_3_html = f"""
    <div>
        {get_plot_div(fig_hour)}
        <p><b>Strongest Hour:</b> {fmt_hour_range(best_h)} ({fmt_pct(hour_counts[best_h], total_words)})<br>
        <b>Weakest Hour:</b> {fmt_hour_range(worst_h)} ({fmt_pct(hour_counts[worst_h], total_words)})</p>
    </div>
    <div style="margin-top:20px;">
        {get_plot_div(fig_day)}
        <p><b>Strongest Day:</b> {best_d} ({fmt_pct(day_counts[best_d], total_words)})<br>
        <b>Weakest Day:</b> {worst_d} ({fmt_pct(day_counts[worst_d], total_words)})</p>
    </div>
    <div style="margin-top:20px;">
        {get_plot_div(fig_my)}
        <p><b>Strongest Month:</b> {best_m_str}<br>
        <b>Weakest Month:</b> {worst_m_str}</p>
    </div>
    """
    
    add_section("1. Time-Based Analytics", f"""
        <h3>1.1 Activity Map</h3>
        {gh_maps_html}
        <div class="grid">
            <div><h4>Top 3 Streaks</h4>{streak_html}</div>
            <div><h4>Top 3 Gaps</h4>{gap_html}</div>
        </div>
        <h3>1.2 Learning Timeline</h3>
        {get_plot_div(fig_tl)}
        <div style="margin-top:20px;">
            {milestone_html}
        </div>
        <h3>1.3 Habits</h3>
        {analysis_1_3_html}
        <h3>1.4 Difficulty Evolution</h3>
        <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:10px;">
            Tracking the average rank of words learned each month. An <b>upward trend</b> indicates you are tackling rarer and more difficult vocabulary over time.
        </p>
        {get_plot_div(fig_rank)}
    """)
    
    # =========================================================
    # 2. WORD TYPE ANALYSIS
    # =========================================================
    print(">>> 2. Word Types...")
    
    # 2.1 Horizontal Bar Chart
    type_counts = df_combined['type'].value_counts().reset_index()
    type_counts.columns = ['type', 'count']
    type_counts = type_counts.sort_values('count', ascending=False)
    
    # Create horizontal bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=type_counts['type'].tolist(),
        x=type_counts['count'].tolist(),
        orientation='h',
        text=[f"{c:,} ({fmt_pct(c, total_words)})" for c in type_counts['count']],
        textposition='auto',
        marker=dict(
            color=type_counts['count'].tolist(),
            colorscale=[[0, '#ffffff'], [1.0, '#8b0000']],
            line=dict(width=0)
        ),
        showlegend=False
    ))
    
    fig_bar.update_layout(
        title="Vocabulary Distribution",
        xaxis=dict(title='Word Count', showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(title=''),
        template='plotly_white',
        height=300,
        margin=dict(l=100, r=20, t=40, b=40)
    )
    
    # 2.2 Relative Cumulative Timeline
    # Create a daily record of added types
    df_timeline = df_combined[['date', 'type']].copy()
    # Pivot to count types per day
    daily_type_counts = pd.crosstab(df_timeline['date'], df_timeline['type']).reindex(all_days_idx, fill_value=0)
    # Accumulate
    cumulative_type_counts = daily_type_counts.cumsum()
    # Normalize to percentages
    cumulative_pct = cumulative_type_counts.div(cumulative_type_counts.sum(axis=1), axis=0).fillna(0) * 100
    cumulative_pct = cumulative_pct.reset_index().rename(columns={'index': 'date'})
    # Melt for Area Plot
    melted_pct = cumulative_pct.melt(id_vars='date', var_name='Type', value_name='Percentage')
    
    # Manually build stacked area chart to avoid px.area issues with DF serialization
    fig_area = go.Figure()
    types = melted_pct['Type'].unique()
    
    for t in types:
        subset = melted_pct[melted_pct['Type'] == t]
        fig_area.add_trace(go.Scatter(
            x=subset['date'].tolist(),
            y=subset['Percentage'].tolist(),
            name=t,
            stackgroup='one', # Stack them
            mode='none'       # No lines, just filled area
        ))

    fig_area.update_layout(
        title="Relative Vocabulary Composition over Time",
        yaxis=dict(range=[0, 100]), 
        height=450
    )
    
    
    add_section("2. Word Type Analysis", f"""
        {get_plot_div(fig_bar)}
        <div style="margin-top:30px;">
            {get_plot_div(fig_area)}
        </div>
    """)
    
    # =========================================================
    # 3. DEEP DIVE
    # =========================================================
    print(">>> 3. Deep Dive...")
    
    BANDS = [250, 500, 1000, 2000, 5000, 10000, 20000]
    
    def fmt_band_label(b):
        if b >= 1000: return f"{int(b/1000)}k"
        return str(b)
    
    def get_freq_table(subset_df, type_name=None, total_ref=None, title_override=None):
        # Get type-specific IDs from the proper tables
        if type_name == 'noun':
            type_ids = set(df_nouns['word_id'].unique())
            ref_subset = df_words[df_words['id'].isin(type_ids)]
        elif type_name == 'verb':
            type_ids = set(df_verbs['word_id'].unique())
            ref_subset = df_words[df_words['id'].isin(type_ids)]
        elif type_name == 'adjective':
            # Adjectives don't have a separate table, use type field
            ref_subset = df_words[df_words['type'] == 'adjective']
        elif type_name == 'adverb':
            # Adverbs don't have a separate table, use type field
            ref_subset = df_words[df_words['type'] == 'adverb']
        else:
            # All types
            ref_subset = df_words
            
        if total_ref is None: total_ref = len(ref_subset)
        
        label = title_override if title_override else (type_name.capitalize() if type_name else 'Total')
        html_out = f"<h4>{label}</h4>"
        html_out += """
        <table class="freq-table">
            <thead><tr><th>Band</th><th>Known <span style="font-weight:normal; color:#bdc3c7;">(of Total)</span></th><th>Coverage %</th></tr></thead>
            <tbody>
        """
        for b in BANDS:
            # Count distinct ranks in reference (handles duplicates)
            ref_ranks = ref_subset[ref_subset['rank'] <= b]['rank'].dropna().unique()
            ref_count = len(ref_ranks)
            
            # Count distinct ranks for known words (handles duplicates on user side too)
            my_ranks = subset_df[subset_df['rank'] <= b]['rank'].dropna().unique()
            my_count = len(my_ranks)
            
            pct = (my_count / ref_count * 100) if ref_count > 0 else 0
            
            # Gradient color: black (0%) to green (100%)
            # HSL: hue=120 (green), lightness varies from 15% to 45%
            lightness = 15 + (pct / 100) * 30  # 15% to 45%
            saturation = 50 + (pct / 100) * 30  # 50% to 80% for richer color
            bar_color = f'hsl(120, {saturation}%, {lightness}%)'
            
            row_str = f"Top {fmt_band_label(b)}"
            known_str = f"<b>{my_count}</b> <span style=\"color:#bdc3c7;\">({ref_count})</span>"
            
            # Visual bar with percentage
            bar_width = min(pct, 100)
            pct_bar = f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="flex:1; background:#ecf0f1; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:{bar_color}; width:{bar_width}%; height:100%; border-radius:4px;"></div>
                </div>
                <span style="font-weight:600; color:{bar_color}; min-width:45px;">{pct:.1f}%</span>
            </div>
            """
            
            html_out += f"<tr><td>{row_str}</td><td>{known_str}</td><td>{pct_bar}</td></tr>"
        
        # Total Row (Unfiltered)
        total_known = len(subset_df)
        total_avail = len(ref_subset)
        total_pct = (total_known / total_avail * 100) if total_avail > 0 else 0
        total_bar_width = min(total_pct, 100)
        
        # Gradient color for total row
        total_lightness = 15 + (total_pct / 100) * 30
        total_saturation = 50 + (total_pct / 100) * 30
        total_bar_color = f'hsl(120, {total_saturation}%, {total_lightness}%)'
        
        total_pct_bar = f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="flex:1; background:#ecf0f1; height:10px; border-radius:4px; overflow:hidden;">
                <div style="background:{total_bar_color}; width:{total_bar_width}%; height:100%; border-radius:4px;"></div>
            </div>
            <span style="font-weight:700; color:{total_bar_color}; min-width:45px;">{total_pct:.1f}%</span>
        </div>
        """
        
        html_out += f"<tr style='border-top:2px solid #eee;'><td><b>Total</b></td><td><b>{total_known}</b> <span style=\"color:#bdc3c7;\">({total_avail})</span></td><td>{total_pct_bar}</td></tr>"
    
        html_out += "</tbody></table>"
        return html_out
    
    # 3.1 Frequency Tables (Consolidated)
    # Use type-specific tables to properly identify words by type
    noun_ids = set(df_nouns['word_id'].unique())
    verb_ids = set(df_verbs['word_id'].unique())
    
    df_noun = df_combined[df_combined['word_id'].isin(noun_ids)]
    df_verb = df_combined[df_combined['word_id'].isin(verb_ids)]
    df_adj = df_combined[df_combined['type'] == 'adjective']
    df_adv = df_combined[df_combined['type'] == 'adverb']
    
    # All Word Types
    html_all_types = get_freq_table(df_combined, type_name=None, title_override="All Word Types")
    
    
    html_freq_overview = f"""
    <div style="margin-bottom:30px;">
        {html_all_types}
    </div>
    <div class="grid">
        <div>{get_freq_table(df_noun, 'noun')}</div>
        <div>{get_freq_table(df_verb, 'verb')}</div>
    </div>
    <div class="grid" style="margin-top:20px;">
        <div>{get_freq_table(df_adj, 'adjective')}</div>
        <div>{get_freq_table(df_adv, 'adverb')}</div>
    </div>
    """
    
    # Helper for Text Stats List
    def get_text_stat_list(series, title, mapping=None, color_map=None):
        counts = series.value_counts()
        total = len(series)
        
        html = f"<h4>{title}</h4><ul style='margin-top:10px; background:#f9f9f9; padding:15px; border-radius:8px;'>"
        
        idx_list = counts.index.tolist()
        # Sort index if needed or keep frequency order
        
        html += "<div style='display:grid; gap:10px;'>"
        for idx_val in idx_list:
            val_count = counts[idx_val]
            pct = (val_count / total * 100) if total > 0 else 0
            
            if mapping:
                if callable(mapping):
                    label = mapping(idx_val)
                elif isinstance(mapping, dict):
                    label = mapping.get(idx_val, str(idx_val))
                else:
                    label = str(idx_val)
            else:
                label = str(idx_val)
            
            # Gradient color bar
            lightness = 15 + (pct / 100) * 30
            saturation = 50 + (pct / 100) * 30
            bar_color = f'hsl(120, {saturation}%, {lightness}%)'
            
            # Color dot from color_map if provided
            dot_html = ""
            if color_map and idx_val in color_map:
                dot_html = f"<span style='display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{color_map[idx_val]}; margin-right:8px; border:1px solid #ddd;'></span>"
            
            html += f"""
            <div style='background:#f9f9f9; padding:10px 15px; border-radius:6px; border-left:3px solid {bar_color};'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                    <span style='font-weight:600; color:#2c3e50;'>{dot_html}{label}</span>
                    <span style='color:{bar_color}; font-weight:bold;'>{val_count} ({pct:.1f}%)</span>
                </div>
                <div style='background:#ecf0f1; height:6px; border-radius:3px; overflow:hidden;'>
                    <div style='background:{bar_color}; width:{pct}%; height:100%; border-radius:3px;'></div>
                </div>
            </div>
            """
        html += "</div>"
        return html
    
    
    # 3.2 Nouns Details (TEXT ONLY)
    df_noun_ext = pd.merge(df_noun, df_nouns, on='word_id', how='left')
    # Stats
    irr_pl = int(df_noun_ext['pl_only'].sum())
    irr_sg = int(df_noun_ext['sg_only'].sum())
    irr_ind = int(df_noun_ext['indeclinable'].sum())
    noun_stats = f"""
    <h4>Irregularities</h4>
    <div style='display:grid; gap:10px; margin-top:10px;'>
    """
    
    irregularities = [
        ('Plural Only', irr_pl),
        ('Singular Only', irr_sg),
        ('Indeclinable', irr_ind)
    ]
    
    for label, count in irregularities:
        pct = (count / len(df_noun) * 100) if len(df_noun) > 0 else 0
        lightness = 15 + (pct / 100) * 30
        saturation = 50 + (pct / 100) * 30
        bar_color = f'hsl(120, {saturation}%, {lightness}%)'
        
        noun_stats += f"""
        <div style='background:#f9f9f9; padding:10px 15px; border-radius:6px; border-left:3px solid {bar_color};'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                <span style='font-weight:600; color:#2c3e50;'>{label}</span>
                <span style='color:{bar_color}; font-weight:bold;'>{count} ({pct:.1f}%)</span>
            </div>
            <div style='background:#ecf0f1; height:6px; border-radius:3px; overflow:hidden;'>
                <div style='background:{bar_color}; width:{pct}%; height:100%; border-radius:3px;'></div>
            </div>
        </div>
        """
    
    noun_stats += "</div>"
    
    # Gender - exclude 'pl' and 'both'
    gender_colors = {'m': COLORS['male'], 'f': COLORS['female'], 'n': COLORS['neuter']}
    gender_map = {'m': 'Masculine', 'f': 'Feminine', 'n': 'Neuter'}
    gender_filtered = df_noun_ext[df_noun_ext['gender'].isin(['m', 'f', 'n'])]['gender']
    html_gender = get_text_stat_list(gender_filtered, "Gender Distribution", gender_map, gender_colors)
    
    # Animate - with emotes
    anim_map = {0.0: '🪨 Inanimate', 1.0: '🐕 Animate'}
    html_anim = get_text_stat_list(df_noun_ext['animate'].dropna(), "Animate vs Inanimate", anim_map)
    
    # Soft Sign
    soft_nouns = df_noun_ext[(df_noun_ext['bare'].str.endswith('ь', na=False)) & (df_noun_ext['gender'].isin(['m','f']))]
    html_soft = get_text_stat_list(soft_nouns['gender'], "Gender of Nouns ending in -ь", gender_map, gender_colors)
    
    html_nouns_detail = f"""
    <div class="grid">
        <div>
            {html_gender}
            <div style="margin-top:20px;">{html_soft}</div>
        </div>
        <div>
            {html_anim}
            <div style="margin-top:20px;">{noun_stats}</div>
        </div>
    </div>
    """
    
    # 3.3 Verbs Details (TEXT ONLY) & 3.4 Aspect Analysis
    df_verb_ext = pd.merge(df_verb, df_verbs, on='word_id', how='left')
    
    # Aspect Text - exclude 'both'
    aspect_filtered = df_verb_ext[df_verb_ext['aspect'].isin(['perfective', 'imperfective'])]['aspect']
    html_aspect = get_text_stat_list(aspect_filtered.dropna(), "Aspect", lambda x: x.capitalize())
    # Reflexive Text
    is_reflexive = df_verb_ext['bare'].str.endswith(('ся', 'сь'), na=False)
    html_reflex = get_text_stat_list(is_reflexive, "Reflexive", {True: 'Reflexive', False: 'Non-Reflexive'})
    # Ends in ь Text
    ends_soft = df_verb_ext['bare'].str.endswith('ь', na=False)
    html_verb_soft = get_text_stat_list(ends_soft, "Infinitives ending in -ь", {True: 'Ends in -ь', False: 'No -ь'})
    
    # 3.4 ASPECT PAIR ANALYSIS
    print(">>> Computing Verb Pairs...")
    # Partner column often contains "word;other_word" or just one word
    # We need to check if ANY partner word is in our known vocabulary
    known_verb_bare_lower = set(df_verb['bare'].str.lower())
    known_verb_ids = set(df_verb['word_id'])
    
    pair_stats = {
        'both': 0,      # Both Aspect and Partner known
        'orphan': 0,    # Only this verb known
        'no_partner': 0 # No partner listed in DB
    }
    
    for _, v_row in df_verb_ext.iterrows():
        partner_str = str(v_row['partner']).strip()
        if not partner_str or partner_str.lower() == 'nan':
            pair_stats['no_partner'] += 1
            continue
        
        # Split possible multiple partners
        partners = [p.strip().lower() for p in partner_str.split(';')]
        
        # Try to find if any partner is known. 
        # Note: Checking by 'bare' text is easiest, but ideally we'd check ID. 
        # The partner column is text, so we check text membership.
        is_pair_known = any(p in known_verb_bare_lower for p in partners)
        
        if is_pair_known:
            pair_stats['both'] += 1
        else:
            pair_stats['orphan'] += 1
    
    total_verbs_analyzed = pair_stats['both'] + pair_stats['orphan'] # Exclude 'no_partner' from mix? Or include?
    # Let's include only verbs THAT HAVE A PARTNER in the ratio
    if total_verbs_analyzed > 0:
        pair_ratio = pair_stats['both'] / total_verbs_analyzed
        orphan_ratio = pair_stats['orphan'] / total_verbs_analyzed
    else:
        pair_ratio = 0
        orphan_ratio = 0
    
    html_aspect_pairs = f"""
    <h4>Aspect Pair Completeness</h4>
    <div style='display:grid; gap:10px; margin-top:10px;'>
    """
    
    pair_items = [
        ('Complete Pairs (Both Known)', pair_stats['both'], pair_ratio * 100, COLORS['streak']),
        ('Orphans (One Known)', pair_stats['orphan'], orphan_ratio * 100, COLORS['female']),
        ('No Known Partner', pair_stats['no_partner'], (pair_stats['no_partner'] / len(df_verb) * 100) if len(df_verb) > 0 else 0, '#95a5a6')
    ]
    
    for label, count, pct, dot_color in pair_items:
        # Gradient based on percentage
        lightness = 15 + (pct / 100) * 30
        saturation = 50 + (pct / 100) * 30
        bar_color = f'hsl(120, {saturation}%, {lightness}%)'
        
        html_aspect_pairs += f"""
        <div style='background:#f9f9f9; padding:10px 15px; border-radius:6px; border-left:3px solid {bar_color};'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                <span style='font-weight:600; color:#2c3e50;'><span style='display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{dot_color}; margin-right:8px; border:1px solid #ddd;'></span>{label}</span>
                <span style='color:{bar_color}; font-weight:bold;'>{count} ({pct:.1f}%)</span>
            </div>
            <div style='background:#ecf0f1; height:6px; border-radius:3px; overflow:hidden;'>
                <div style='background:{bar_color}; width:{pct}%; height:100%; border-radius:3px;'></div>
            </div>
        </div>
        """
    
    html_aspect_pairs += "</div>"
    
    html_verbs_detail = f"""
    <div class="grid">
        <div>{html_aspect}</div>
        <div>{html_aspect_pairs}</div>
    </div>
    <div class="grid" style="margin-top:20px;">
        <div>{html_reflex}</div>
        <div>{html_verb_soft}</div>
    </div>
    """
    
    add_section("3. Deep Dive Analysis", f"""
        <h3>3.1 Frequency Overview</h3>
        {html_freq_overview}
    
        <h3 style="margin-top:40px; border-top:1px solid #ddd; padding-top:20px;">3.2 Nouns Analysis</h3>
        {html_nouns_detail}
    
        <h3 style="margin-top:40px; border-top:1px solid #ddd; padding-top:20px;">3.3 Verbs Analysis</h3>
        {html_verbs_detail}
    """)
    
    # =========================================================
    # 5. HALL OF FAME
    # =========================================================
    print(">>> 5. Hall of Fame (Longest Words)...")
    
    # Same as before
    hof_df = df_combined.copy()
    hof_df['clean_word'] = hof_df['bare'].fillna(hof_df['word_text'])
    hof_df = hof_df[~hof_df['clean_word'].str.contains(' ', na=False)]
    hof_df['len'] = hof_df['clean_word'].str.len()
    
    def get_longest_html(type_label):
        subset = hof_df[hof_df['type'] == type_label].copy()
        subset = subset.sort_values('len', ascending=False).head(5)
        
        html = f"<h4>Longest {type_label.capitalize()}s</h4>"
        html += "<table class='freq-table'><thead><tr><th>Word</th><th>Length</th></tr></thead><tbody>"
        for _, row in subset.iterrows():
            w = row['clean_word']
            l = row['len']
            link = f"https://en.openrussian.org/ru/{w}"
            w_html = f"<a href='{link}' target='_blank' style='color:{COLORS['male']}; text-decoration:none; font-weight:bold;'>{w}</a>"
            html += f"<tr><td>{w_html}</td><td>{int(l)}</td></tr>"
        html += "</tbody></table>"
        return html
    
    html_hof_grid = f"""
    <div class="grid">
        <div>{get_longest_html('noun')}</div>
        <div>{get_longest_html('verb')}</div>
    </div>
    <div class="grid" style="margin-top:20px;">
        <div>{get_longest_html('adjective')}</div>
        <div>{get_longest_html('adverb')}</div>
    </div>
    """
    
    add_section("5. Hall of Fame", list_html_hof := html_hof_grid)
    
    # =========================================================
    # 6. ESTIMATED COVERAGE & CEFR
    # =========================================================
    print(">>> 6. Coverage & CEFR...")
    
    def zipf_model_val(n):
        """
        Estimate text coverage based on vocabulary size using piecewise linear interpolation.
        Based on research-backed benchmarks for Russian language.
        
        Primary source: "A Frequency Dictionary of Russian" (Sharoff, Umanskaya, Wilson 2013)
        Based on 150-million-word Internet corpus.
        """
        # Research-based benchmark points: (vocabulary_size, coverage_percentage)
        benchmarks = [
            (0, 0.0),
            (1000, 0.72),      # Survival level
            (2000, 0.80),      # Conversational (Sharoff et al. 2013)
            (5000, 0.904),     # Advanced (Sharoff et al. 2013: 90.4%)
            (10000, 0.95),     # Near-Native (estimated)
            (20000, 0.97),     # Professional (estimated)
            (30000, 0.98),     # Highly proficient (estimated)
            (50000, 0.99),     # Near-complete (estimated, may require more)
        ]
        
        # Find the two benchmark points that n falls between
        for i in range(len(benchmarks) - 1):
            x1, y1 = benchmarks[i]
            x2, y2 = benchmarks[i + 1]
            
            if n <= x1:
                return y1
            elif x1 < n <= x2:
                # Linear interpolation
                return y1 + (y2 - y1) * (n - x1) / (x2 - x1)
        
        # If n is beyond the last benchmark, return the last value
        return benchmarks[-1][1]
    
    est_coverage = zipf_model_val(total_words)
    
    # Create visualization of coverage curve
    vocab_sizes = list(range(0, 55000, 500))
    coverage_vals = [zipf_model_val(v) * 100 for v in vocab_sizes]
    
    fig_coverage = go.Figure()
    
    # Add the coverage curve
    fig_coverage.add_trace(go.Scatter(
        x=vocab_sizes,
        y=coverage_vals,
        mode='lines',
        name='Coverage Curve',
        line=dict(color=COLORS['accent'], width=3),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    
    # Mark user's current position
    fig_coverage.add_trace(go.Scatter(
        x=[total_words],
        y=[est_coverage * 100],
        mode='markers+text',
        name='Your Position',
        marker=dict(size=15, color=COLORS['streak'], symbol='diamond'),
        text=[f'You are here<br>{total_words:,} words<br>{est_coverage*100:.1f}%'],
        textposition='top center',
        textfont=dict(size=11, color=COLORS['streak'], weight='bold')
    ))
    
    # Add benchmark markers
    benchmark_points = [
        (1000, 72, '1k: Survival'),
        (2000, 80, '2k: Conversational'),
        (5000, 90.4, '5k: Advanced'),
        (10000, 95, '10k: Near-Native'),
        (20000, 97, '20k: Professional'),
        (30000, 98, '30k: Highly Proficient'),
        (50000, 99, '50k: Near-Complete')
    ]
    
    fig_coverage.add_trace(go.Scatter(
        x=[p[0] for p in benchmark_points],
        y=[p[1] for p in benchmark_points],
        mode='markers',
        name='Benchmarks',
        marker=dict(size=8, color='#95a5a6', symbol='circle'),
        text=[p[2] for p in benchmark_points],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig_coverage.update_layout(
        title='Russian Vocabulary Coverage Curve',
        xaxis=dict(
            title='Vocabulary Size (words)',
            showgrid=True,
            gridcolor='#f0f0f0',
            tickformat=',',
            range=[0, 55000]
        ),
        yaxis=dict(
            title='Text Coverage (%)',
            showgrid=True,
            gridcolor='#f0f0f0',
            range=[0, 105]
        ),
        template='plotly_white',
        height=450,
        hovermode='closest'
    )
    
    zipf_explanation_full = """
    <p style="background: #eef2f3; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <strong>📊 Research-Based Benchmarks</strong> <span style="font-size: 0.85rem; color: #7f8c8d;">(Sharoff, Umanskaya, Wilson 2013)</span><br><br>
        • <strong>1,000 words:</strong> ~72% coverage <span style="color: #7f8c8d;">(Survival)</span><br>
        • <strong>2,000 words:</strong> ~80% coverage <span style="color: #7f8c8d;">(Conversational)</span><br>
        • <strong>5,000 words:</strong> ~90.4% coverage <span style="color: #7f8c8d;">(Advanced)</span><br>
        • <strong>10,000 words:</strong> ~95% coverage <span style="color: #7f8c8d;">(Near-Native)</span><br>
        • <strong>20,000 words:</strong> ~97% coverage <span style="color: #7f8c8d;">(Professional)</span><br>
        • <strong>30,000 words:</strong> ~98% coverage <span style="color: #7f8c8d;">(Highly Proficient)</span><br>
        • <strong>50,000+ words:</strong> ~99% coverage <span style="color: #7f8c8d;">(Near-Complete)</span><br>
    </p>
    <p style="font-size: 1.05rem; line-height: 1.6;">
        With your current vocabulary of <strong style="color: {color};">{vocab:,}</strong> words, you are estimated to understand approximately <strong style="color: {color};">{cov:.1f}%</strong> of a standard Russian text.
    </p>
    <p style="font-size: 0.9rem; color: #7f8c8d; margin-top: 15px; line-height: 1.5;">
        <strong>Note:</strong> Due to <em>Zipf's Law</em>, the final percentages (97% → 99%) require exponentially more words, as they consist of increasingly rare vocabulary. The jump from 95% to 99% coverage can require an additional 30,000-40,000+ words.
    </p>
    """.format(color=COLORS['accent'], vocab=total_words, cov=est_coverage*100)
    
    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    cefr_html = "<h4>CEFR Proficiency Breakdown</h4>"
    cefr_html += "<table class='freq-table'><thead><tr><th>Level</th><th>Mastery</th><th>Progress</th></tr></thead><tbody>"
    
    for lvl in cefr_levels:
        # Count unique word IDs (handles duplicate entries in database)
        ref_ids = df_words[df_words['level'] == lvl]['id'].unique()
        ref_total = len(ref_ids)
        
        my_ids = df_combined[df_combined['level'] == lvl]['word_id'].unique()
        my_total = len(my_ids)
        
        pct = (my_total / ref_total * 100) if ref_total > 0 else 0
        bar_width = min(pct, 100)
        
        # Gradient color: dark green to bright green based on percentage
        lightness = 15 + (pct / 100) * 30
        saturation = 50 + (pct / 100) * 30
        bar_color = f'hsl(120, {saturation}%, {lightness}%)'
        
        bar_html = f"<div style='background:#ecf0f1; width:100px; height:8px; border-radius:4px; display:inline-block;'><div style='background:{bar_color}; width:{bar_width}px; height:8px; border-radius:4px;'></div></div>"
        
        cefr_html += f"<tr><td><b>{lvl}</b></td><td>{my_total} / {ref_total}</td><td>{bar_html} {pct:.1f}%</td></tr>"
    
    # Add Total row
    total_ref = len(df_words[df_words['level'].isin(cefr_levels)]['id'].unique())
    total_my = len(df_combined[df_combined['level'].isin(cefr_levels)]['word_id'].unique())
    total_pct = (total_my / total_ref * 100) if total_ref > 0 else 0
    total_bar_width = min(total_pct, 100)
    
    # Gradient color for total
    total_lightness = 15 + (total_pct / 100) * 30
    total_saturation = 50 + (total_pct / 100) * 30
    total_bar_color = f'hsl(120, {total_saturation}%, {total_lightness}%)'
    
    total_bar_html = f"<div style='background:#ecf0f1; width:100px; height:8px; border-radius:4px; display:inline-block;'><div style='background:{total_bar_color}; width:{total_bar_width}px; height:8px; border-radius:4px;'></div></div>"
    
    cefr_html += f"<tr style='border-top:2px solid #eee;'><td><b>Total</b></td><td><b>{total_my} / {total_ref}</b></td><td>{total_bar_html} <b>{total_pct:.1f}%</b></td></tr>"
    
    cefr_html += "</tbody></table>"
    
    
    add_section("6. Estimated Vocabulary Coverage", f"""
    <div style="padding:20px;">
        {zipf_explanation_full}
        <div style="font-size:4rem; font-weight:bold; color:{COLORS['accent']}; text-align:center; margin: 30px 0;">
            {est_coverage*100:.1f}%
        </div>
        <div style="margin-top:30px;">
            {get_plot_div(fig_coverage)}
        </div>
        <div style="margin-top:40px;">
            {cefr_html}
        </div>
    </div>
    """)
    
    # =========================================================
    # 7. MISSING LINKS (Network)
    # =========================================================
    print(">>> 7. Missing Links...")
    
    # 1. Get all known word IDs
    known_ids_set = set(df_combined['word_id'])
    
    # 2. Filter relations
    #   - Source is known
    #   - Target is UNKNOWN
    #   - Relation type is 'related'
    missing_links = df_rels[
        (df_rels['word_id'].isin(known_ids_set)) & 
        (~df_rels['rel_word_id'].isin(known_ids_set)) &
        (df_rels['relation'] == 'related')
    ].copy()
    
    # 3. Connection Count
    # Create Mapping: Known ID -> Known Word Text
    known_id_map = df_combined.set_index('word_id')['bare'].fillna(df_combined['word_text']).to_dict()
    
    # Add a text column to missing_links for aggregation
    missing_links['known_source_text'] = missing_links['word_id'].map(known_id_map)
    
    # Group by Unknown ID
    # We collect 'known_source_text' into a list (so we can hyperlink individually later)
    link_agg = missing_links.groupby('rel_word_id').agg(
        connections=('word_id', 'count'),
        related_list=('known_source_text', lambda x: list(sorted(set(str(s) for s in x if pd.notna(s)))))
    ).reset_index()
    
    link_agg.columns = ['id', 'connections', 'related_list']
    
    # 4. Merge
    top_links_full = pd.merge(link_agg, df_words, on='id', how='inner')
    
    # Breakdown Logic (ON FULL SET)
    breakdown_html = "<h4>Connection Breakdown</h4>"
    breakdown_html += "<p style='font-size:0.9rem; color:#7f8c8d; margin-bottom:15px;'>Distribution of all unknown words by number of known relatives:</p>"
    conn_histogram = top_links_full['connections'].value_counts().sort_index(ascending=False)
    
    # Calculate max for scaling bars
    max_count = conn_histogram.max()
    
    breakdown_html += "<div style='margin-top:10px;'>"
    for conn_num, count_of_words in conn_histogram.items():
        # Calculate bar width proportionally
        bar_pct = (count_of_words / max_count) * 100
        
        # Gradient color based on connection count (more connections = darker/more saturated)
        if conn_num >= 8:
            bar_color = '#16a085'  # Teal
        elif conn_num >= 5:
            bar_color = '#27ae60'  # Green
        elif conn_num >= 3:
            bar_color = '#f39c12'  # Orange
        else:
            bar_color = '#95a5a6'  # Grey
        
        breakdown_html += f"""
        <div style='margin-bottom:12px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                <span style='font-weight:600; color:#2c3e50; font-size:0.95rem;'>{conn_num} connection{"s" if conn_num > 1 else ""}</span>
                <span style='color:#7f8c8d; font-size:0.9rem;'>{count_of_words:,} words</span>
            </div>
            <div style='background:#ecf0f1; height:8px; border-radius:4px; overflow:hidden;'>
                <div style='background:{bar_color}; width:{bar_pct}%; height:100%; border-radius:4px; transition:width 0.3s;'></div>
            </div>
        </div>
        """
    breakdown_html += "</div>"
    
    
    # 5. Sort Table (Connections Descending)
    top_links_table = top_links_full.sort_values(['connections', 'rank'], ascending=[False, True])
    
    # Take top 10 for table
    top_10_links = top_links_table.head(10)
    
    html_missing_links = "<table class='freq-table'><thead><tr><th>Word</th><th>Rank</th><th>Connections</th></tr></thead><tbody>"
    for _, r in top_10_links.iterrows():
        w = r['bare'] if pd.notna(r['bare']) else "Unknown"
        
        link_w = f"https://en.openrussian.org/ru/{w}"
        w_html = f"<a href='{link_w}' target='_blank' style='color:{COLORS['male']}; text-decoration:none; font-weight:bold;'>{w}</a>"
        
        # Format Connections: "N ({link1, link2, ...})"
        rel_list = r['related_list'] if isinstance(r['related_list'], list) else []
        
        # GENERATE INDIVIDUAL LINKS
        link_htmls = []
        for rel_word in rel_list:
            rel_url = f"https://en.openrussian.org/ru/{rel_word}"
            link_htmls.append(f"<a href='{rel_url}' target='_blank' style='color:#7f8c8d; text-decoration:none; border-bottom:1px dotted #999;'>{rel_word}</a>")
        
        rel_str = ", ".join(link_htmls)
        
        # Truncate visual list if too long? 
        # HTML length is huge, but displayed length is fine. 
        # CSS truncation might be better, or just limit number of items shown?
        # User requested hyperlinks for ALL. Let's just output them.
        # If list is enormous, it might break layout. Limit to e.g. 10 items?
        if len(link_htmls) > 10:
            rel_str = ", ".join(link_htmls[:10]) + ", ..."
            
        conn_text = f"{r['connections']} <span style='color:#7f8c8d; font-size:0.85rem;'>({rel_str})</span>"
        
        rank_val = int(r['rank']) if pd.notna(r['rank']) else "-"
        html_missing_links += f"<tr><td>{w_html}</td><td>{rank_val}</td><td>{conn_text}</td></tr>"
    html_missing_links += "</tbody></table>"
    
    add_section("7. Familiar Unknown Words", f"""
    {html_missing_links}
    <div style="margin-top:20px;">{breakdown_html}</div>
    """)
    
    # =========================================================
    # 8. TOP 100 MISSING WORDS (Rank Deduplicated)
    # =========================================================
    print(">>> 8. Top 100 Missing (Deduplicated)...")
    
    # Logic:
    # 1. User has a set of KNOWN RANKS (e.g. if user knows 'мочь' rank 105, they know rank 105)
    known_ranks = set(df_combined['rank'].dropna().astype(int))
    
    # 2. Get all Unknown Words
    unknown_mask = ~df_words['id'].isin(known_ids_set)
    missing_df_candidates = df_words[unknown_mask].copy()
    
    # 3. Filter out words whose RANK is already known (deduplication by rank logic)
    missing_df_candidates = missing_df_candidates[~missing_df_candidates['rank'].isin(known_ranks)]
    
    # 4. Furthermore, within the candidates, if rank 100 appears twice, we only want the first one.
    missing_df_candidates = missing_df_candidates.drop_duplicates(subset=['rank'])
    
    # Sort
    missing_df_candidates = missing_df_candidates.sort_values('rank', ascending=True).head(100)
    
    # Build horizontal list with hyperlinks
    html_top_missing = "<div style='line-height:2.2; font-size:1rem;'>"
    word_htmls = []
    
    for _, r in missing_df_candidates.iterrows():
        w = r['bare']
        if pd.isna(w): continue
        
        rank = int(r['rank'])
        link = f"https://en.openrussian.org/ru/{w}"
        # Format: word (#rank)
        word_html = f"<a href='{link}' target='_blank' style='color:{COLORS['male']}; text-decoration:none; font-weight:bold;'>{w}</a> <span style='color:#7f8c8d;'>(#{rank})</span>"
        word_htmls.append(word_html)
    
    # Join with tab spaces
    html_top_missing += " &nbsp;&nbsp;&nbsp; ".join(word_htmls)
    html_top_missing += "</div>"
    
    add_section("8. Top 100 unknown words by rank", f"""
    {html_top_missing}
    """)
    
    # =========================================================
    # WRITE HTML
    # =========================================================
    
    CSS_CONTENT = """
        body { font-family: 'Inter', sans-serif; background: #f4f6f8; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 5px; }
        h2 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 0; }
        h3 { color: #34495e; font-size: 1.1rem; margin-bottom: 10px; }
        h4 { color: #7f8c8d; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .header-sub { text-align: center; color: #7f8c8d; margin-bottom: 40px; }
        .card { background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); padding: 30px; margin-bottom: 30px; }
        
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-box { background: white; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #eee; }
        .stat-value { font-size: 2.2rem; font-weight: 700; color: #2c3e50; }
        .stat-label { font-size: 0.85rem; color: #95a5a6; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
        .stat-sub { font-size: 0.85rem; color: #bdc3c7; margin-top: 5px; }
    
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 30px; }
        
        ul { list-style: none; padding: 0; }
        li { padding: 6px 0; border-bottom: 1px solid #f9f9f9; font-size: 0.95rem; }
        li:last-child { border-bottom: none; }
        
        .freq-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .freq-table th { text-align: left; border-bottom: 2px solid #eee; padding: 8px; color: #7f8c8d; }
        .freq-table td { border-bottom: 1px solid #f9f9f9; padding: 8px; vertical-align: middle; }
        .freq-table tr:last-child td { border-bottom: none; }
        
        .js-plotly-plot { margin: 0 auto; width: 100% !important; }
    """
    
    final_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Russian Vocabulary Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        {CSS_CONTENT}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Russian Vocabulary Analysis</h1>
            <div class="header-sub">Generated on {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}</div>
    
            {header_html}
            {"".join(html_sections)}
            
        </div>
    </body>
    </html>
    """

    return final_html

if __name__ == "__main__":
    import sys
    
    # Determined input file
    input_file = "data/my_russian_words.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # Load User JSON
    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            user_data = json.load(f)
        
        report_html = generate_report(user_data, data_dir="data")
        
        with open(OUTPUT_HTML, "w") as f:
            f.write(report_html)
        print(f"DONE. Report generated at: {OUTPUT_HTML}")
    else:
        print(f"Error: Input file '{input_file}' not found.")
        print("Usage: python analyze_vocabulary.py [path_to_json_file]")
