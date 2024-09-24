### Library
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

### Constants
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
PERIOD_MAP = {
        "Daily": "STRFTIME('%Y-%m-%d', timestamp)",
        "Weekly": "DATE(timestamp, 'weekday 0')",
        "Monthly": "STRFTIME('%Y-%m', timestamp)",
        "Yearly": "STRFTIME('%Y', timestamp)"
    }

### Functions
# Data Loader
@st.cache_data
def load_stg(path: str):
    df = pd.read_csv(path)

    # Stripping blank spaces
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Dropping id column
    df.drop(columns=['id'], inplace=True)

    # Dropping duplicates
    df = df.drop_duplicates()

    # Extract all numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    columns_to_convert = [col for col in numeric_columns if col != 'frequency']

    # Convert values from MW to GW
    df[columns_to_convert] = df[columns_to_convert] / 1000

    return df, numeric_columns

# Data to In-Memory DB
@st.cache_resource
def create_db(df):
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    df.to_sql('stg_grid', conn, index=False, if_exists='replace')
    conn.execute('CREATE INDEX idx_timestamp ON stg_grid (timestamp);')
    return conn

def get_timestamp(conn):
    query = "SELECT MAX(timestamp) as max_timestamp,MIN(timestamp) as min_timestamp FROM stg_grid"
    max_timestamp = pd.read_sql(query, conn)['max_timestamp'][0]
    min_timestamp = pd.read_sql(query, conn)['min_timestamp'][0]
    min_timestamp = pd.to_datetime(min_timestamp)
    max_timestamp = pd.to_datetime(max_timestamp)
    return max_timestamp, min_timestamp

# Get overall summary of data
@st.cache_data
def get_summary_data(_conn, numeric_columns, min_date, max_date):
    sum_columns = ", ".join([f"AVG({col}) as {col}" for col in numeric_columns])

    query = f"""
        SELECT {sum_columns}
        FROM stg_grid
        WHERE timestamp >= DATETIME('{min_date}') AND timestamp <= DATETIME('{max_date}')
        ;
    """
    return pd.read_sql(query, conn)

# Pie Chart
@st.cache_data
def get_pie_data(df):
    source = df.drop(columns=['demand','frequency'], axis=1).iloc[0]
    source = source.replace([float('inf'), -float('inf')], pd.NA)
    source = source[source > 0].dropna()
    
    # Drop energy sources with extremely low or invalid values
    source = source.dropna()

    total_demand = df['demand'].iloc[0]
    percentages = round((source / total_demand) * 100,2)
    pie_data = percentages.reset_index().rename(columns={0: 'percentage', 'index': 'source'})
    pie_data['demand'] = total_demand
    pie_data['power'] = source.values

    return pie_data

def plot_pie(df):
    pie_data = get_pie_data(df)
    fig_pie = px.pie(pie_data, names='source', values='power', title='Energy Generation by Metered Source', hole=.4)
    fig_pie.update_traces(
        textinfo='percent',
        insidetextorientation='horizontal',
        textposition='inside',
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Gauge Figure
def get_gauge_data(df, variable):
    if variable == 'demand':
        ind_number = df[variable]
        max_bound = df[variable].max()
        ind_suffix = 'GW'
        ind_title = 'Demand'
    elif variable == 'frequency':
        ind_number = df[variable]
        max_bound = df[variable].max()
        ind_suffix = 'Hz'
        ind_title = 'Frequency'
    else:
        ind_number = df[variable]
        max_bound = df['demand'].mean() 
        ind_suffix = 'GW'
        ind_title = variable.capitalize()

    return ind_number, ind_suffix, ind_title, max_bound

def plot_gauge(df,variable,font_size):
    ind_number,ind_suffix,ind_title,max_bound = get_gauge_data(df,variable)
    fig = go.Figure(
        go.Indicator(
            value=ind_number.iloc[0],
            mode="gauge+number",
            domain={"x" : [0,1], "y": [0,0.85]},
            number={
                "suffix":ind_suffix,
                "font.size":font_size,
            },
            gauge={
                "axis":{"range":[0,max_bound],"tickwidth":1},
            },
            title={
                "text":ind_title,
                "font":{"size":font_size},
            },
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=80, b=0),
        height=170
    )
    st.plotly_chart(fig,use_container_width=True)

def plot_gauge_summary(df):
    source = [col for col in numeric_columns if col not in ['timestamp', 'frequency', 'demand']]
    mean_values = df[source].mean().sort_values(ascending=False)
    sorted_source = mean_values.index.tolist()

    mid_index = len(sorted_source) // 2
    row1 = sorted_source[:mid_index]
    row2 = sorted_source[mid_index:]

    row1_cols = st.columns(len(row1))
    for i, variable in enumerate(row1):
        with row1_cols[i]:
            plot_gauge(df, variable,15)

    row2_cols = st.columns(len(row2))
    for i, variable in enumerate(row2):
        with row2_cols[i]:
            plot_gauge(df, variable,15)

# Year-on-Year Chart
@st.cache_data
def get_yoy_data(_conn, variable):
    query = f"""
        SELECT STRFTIME('%Y', timestamp) as year,
               STRFTIME('%m', timestamp) as month,
               AVG({variable}) as avg_value
        FROM stg_grid
        GROUP BY year, month
        ORDER BY year, month;
    """
    df = pd.read_sql(query, conn)

    yoy_data = df.pivot(index='month', columns='year', values='avg_value')
    
    yoy_data.index = pd.to_datetime(yoy_data.index, format='%m').month_name()
    return yoy_data

def plot_yoy(df, variable):
    fig = go.Figure()
    
    for year in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[year], 
            mode='lines', 
            name=str(year)
        ))

    y_title = 'GW'
    if variable in ['demand','frequency']:
        plot_title = f"Year-on-Year Monthly Averages for {variable.capitalize()}"
        if variable == 'frequency':
            y_title = 'Hz'
    else:
        plot_title = f"Year-on-Year Monthly Averages for {variable.capitalize()} Demand"

    fig.update_layout(
        title=plot_title,
        xaxis_title=None,
        yaxis_title=y_title,
    )
    
    st.plotly_chart(fig,use_container_width=True)

# Peak, Trough and Moving Average Charts
@st.cache_data
def get_peak_trough_moving_avg_data(_conn, variable, period, window_size=7, 
                                    min_date=None, max_date=None,
                                    min_year=None, min_month=None, max_year=None, max_month=None,
                                    period_map = PERIOD_MAP):
    
    period_format = period_map[period]
    where_clause = ""

    if min_date and max_date:
        where_clause = f"WHERE timestamp >= DATETIME('{min_date}') AND timestamp <= DATETIME('{max_date}')"
    elif min_year and max_year:
        if min_month and max_month:
            min_date = f"{min_year}-{str(MONTHS.index(min_month) + 1).zfill(2)}-01"
            max_date = f"{max_year}-{str(MONTHS.index(max_month) + 1).zfill(2)}-01"
            where_clause = f"WHERE timestamp >= DATETIME('{min_date}') AND timestamp < DATETIME('{max_date}', '+1 month')"
        
        else:
            where_clause = f"""
                WHERE STRFTIME('%Y', timestamp) >= '{min_year}' 
                  AND STRFTIME('%Y', timestamp) <= '{max_year}'
            """

    query = f"""
        SELECT {period_format} as period,
               MAX({variable}) as peak,
               MIN({variable}) as trough,
               AVG({variable}) as avg_value,
               AVG(AVG({variable})) OVER (
                   ORDER BY {period_format}
                   ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW
               ) AS moving_avg
        FROM stg_grid
        {where_clause}
        GROUP BY {period_format}
        ORDER BY {period_format};
    """
    df = pd.read_sql(query, conn)
    return df

# Annotate extreme values
def add_ext_anno(df,
             type,
             fig,
             y_val,
             bgcolor,
             font_color,
             ay=-30):
    
    if type == 'max':
        ex_val = df[df[y_val] > (df[y_val].mean() + 2 * df[y_val].std())]
    elif type == 'min':
        ex_val = df[df[y_val] < (df[y_val].mean() - 2 * df[y_val].std())]

    for _, row in ex_val.iterrows():
        fig.add_annotation(
            x=row['period'],
            y=row[y_val],
            text=f"{row[y_val]:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=ay,
            bgcolor=bgcolor,
            font=dict(color=font_color)
        )

def plot_peak_trough_moving_avg(df, variable, period,show_anno):
    fig = go.Figure()
    # Peak
    fig.add_trace(go.Scatter(
        x=df['period'], 
        y=df['peak'], 
        mode='lines', 
        name='Peak', 
        line=dict(color='green')
    ))

    # Trough
    fig.add_trace(go.Scatter(
        x=df['period'], 
        y=df['trough'], 
        mode='lines', 
        name='Trough', 
        line=dict(color='red')
    ))

    # MA
    fig.add_trace(go.Scatter(
        x=df['period'], 
        y=df['moving_avg'], 
        mode='lines', 
        name='Moving Avg', 
        line=dict(color='skyblue', dash='dash')
    ))

    # Annotations
    if show_anno:
        add_ext_anno(df,'max',fig,'peak','lime','black')
        add_ext_anno(df,'min',fig,'peak','darkgreen','white')
        add_ext_anno(df,'max',fig,'trough','pink','black',ay=30)
        add_ext_anno(df,'min',fig,'trough','red','white',ay=30)

    # Figure Layout
    y_title = 'GW'

    if variable in ['demand','frequency']:
        plot_title = f"{period} {variable.capitalize()}"
        if variable == 'frequency':
            y_title = 'Hz'
    else:
        plot_title = f"{period} {variable.capitalize()} Demand"
    
    if period == 'Weekly':
        fig.update_layout(
            xaxis=dict(
                tickmode='auto',
                tickvals=df['period'],
                tickformat='%Y-%m (%W)'
            )
        )

    fig.update_layout(
        title = plot_title,
        xaxis_title=None,
        yaxis_title=y_title,
    )
    
    st.plotly_chart(fig,use_container_width=True)

### Code Execution
# Streamlit Configs
st.set_page_config(
    page_title="UK Gridwatch",
    page_icon='📈',
    layout='wide'
    )

# Title
st.title('UK Gridwatch Dashboard')

with st.sidebar:
    st.markdown(
        """
        **Upload CSV from [Gridwatch](https://www.gridwatch.templar.co.uk/download.php) for upload**
        """, 
        unsafe_allow_html=True
    )
    upload_file = st.file_uploader("", type=["csv"])

    # Data Loading
    if upload_file:
        df, numeric_columns = load_stg(upload_file)
        st.success("Custom dataset uploaded successfully!")
    else:
        df, numeric_columns = load_stg('gridwatch.csv')
        st.info("Using default dataset: gridwatch.csv")

conn = create_db(df)
max_timestamp,min_timestamp = get_timestamp(conn)

# Overall Summary
date_sel0 = st.columns(8)
with date_sel0[0]:
    min_date = st.date_input(
        'From',
        min_timestamp.date(),
        min_value=min_timestamp.date(),
        max_value=max_timestamp.date() - timedelta(days=1)
    )
with date_sel0[4]:
    max_date = st.date_input(
        'To',
        max_timestamp.date(),
        min_value=min_date + timedelta(days=1),
        max_value=max_timestamp.date()
    )

st.subheader(f"Gridwatch Summary ({min_date} to {max_date})")
df_summary = get_summary_data(conn, numeric_columns, min_date, max_date)

col_sum_pie, col_sum_gauge = st.columns((1,1))
with col_sum_pie:
    plot_pie(df_summary)
    
with col_sum_gauge:
    plot_gauge(df_summary, 'demand',25)
    plot_gauge(df_summary, 'frequency',25)

plot_gauge_summary(df_summary)

st.divider()
st.divider()

# Individual Source Summary
st.subheader(f"Gridwatch Peaks, Troughs and Moving Averages")
# YOY Chart
sol_source_select, col_source_yoy = st.columns((1,2))
with sol_source_select:
    var_source = st.selectbox("Select:", options=numeric_columns, index=numeric_columns.index('demand'))
    plot_gauge(df_summary, var_source,25)

with col_source_yoy:
    df_yoy = get_yoy_data(conn, var_source)
    plot_yoy(df_yoy, var_source)

# Yearly Chart
col_source_y = st.columns(8)
with col_source_y[0]:
    min_year = st.selectbox(
        'From Year',
        options=range(min_timestamp.year, max_timestamp.year),
        index=0
    )
with col_source_y[4]:
    max_year = st.selectbox(
        'To Year',
        options=range(min_year + 1, max_timestamp.year + 1),
        index=len(range(min_year + 1, max_timestamp.year + 1))-1
    )

anno_source_y = st.checkbox("Show Annotations", value=True, key='anno_y')
df_source_yearly = get_peak_trough_moving_avg_data(conn, var_source, "Yearly", window_size=3, min_year=min_year, max_year=max_year)
plot_peak_trough_moving_avg(df_source_yearly, var_source, "Yearly", show_anno=anno_source_y)

st.divider()

# Monthly and Weekly Charts
col_source_mw = st.columns(8)
with col_source_mw[0]:
    min_year = st.selectbox(
        'From Year',
        options=range(min_timestamp.year, max_timestamp.year + 1),
        index=max(len(range(min_timestamp.year, max_timestamp.year + 1)) - 2,0),
        key='min_year'
    )
with col_source_mw[4]:
    max_year = st.selectbox(
        'To Year',
        options=range(min_year, max_timestamp.year + 1),
        index=max(len(range(min_year + 1, max_timestamp.year + 1)),0),
        key='max_year'
    )

with col_source_mw[1]:
        if min_year == min_timestamp.year:
            from_month = MONTHS[min_timestamp.month - 1:]
            from_month_index = min_timestamp.month - 1 
        elif min_year == max_timestamp.year:
            from_month = MONTHS[:max_timestamp.month - 1]
            from_month_index = max(min_timestamp.month - 1,0)
        else:
            from_month = MONTHS
            from_month_index = max(max_timestamp.month - 1,0)

        min_month = st.selectbox(
            'Month',
            options=from_month,
            index=from_month_index,
            key='min_month'
        )
with col_source_mw[5]:
        if max_year == max_timestamp.year:
            if max_year == min_year:
                from_month_number = MONTHS.index(min_month) + 1
                to_month = MONTHS[from_month_number:max_timestamp.month]
            else:
                to_month = MONTHS[:max_timestamp.month]
        elif max_year == min_year:
            from_month_number = MONTHS.index(min_month) + 1
            to_month = MONTHS[from_month_number:]
        else:
            to_month = MONTHS

        to_month_index = min(from_month_index,len(to_month)-1)

        max_month = st.selectbox(
            'Month',
            options=to_month,
            index=to_month_index,
            key='max_month'
        )

anno_source_mw = st.checkbox("Show Annotations", value=True,key='anno_mw')
df_source_monthly = get_peak_trough_moving_avg_data(conn, var_source, "Monthly", window_size=6, min_year=min_year, max_year=max_year,min_month=min_month,max_month=max_month)
plot_peak_trough_moving_avg(df_source_monthly, var_source, "Monthly",show_anno=anno_source_mw)

df_source_weekly = get_peak_trough_moving_avg_data(conn, var_source, "Weekly", window_size=4, min_year=min_year, max_year=max_year,min_month=min_month,max_month=max_month)
plot_peak_trough_moving_avg(df_source_weekly, var_source, "Weekly", show_anno=anno_source_mw)

st.divider()

# Daily Charts
col_source_d = st.columns(8)
with col_source_d[0]:
    min_daily_date = st.date_input(
        'From',
        max_timestamp.date().replace(year=max_timestamp.date().year - 1),
        min_value=min_timestamp.date(),
        max_value=max_timestamp.date() - timedelta(days=2),
        key='min_daily'
    )
with col_source_d[4]:
    max_daily_date = st.date_input(
        'To',
        max_timestamp.date(),
        min_value=min_daily_date + timedelta(days=2),
        max_value=max_timestamp.date(),
        key='max_daily'
    )

anno_source_d = st.checkbox("Show Annotations", value=True, key='anno_d')
df_source_daily = get_peak_trough_moving_avg_data(conn, var_source, "Daily", window_size=7,min_date=min_daily_date,max_date=max_daily_date)
plot_peak_trough_moving_avg(df_source_daily, var_source, "Daily", show_anno=anno_source_d)
