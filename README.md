# UK Gridwatch Dashboard (Big Pay Challenge)
## Background
[Gridwatch](https://www.gridwatch.templar.co.uk/) provides a dataset of historical UK power consumption since 2011. The data can be downloaded as a single CSV file. Note that data integrity is not guaranteed.
The dataset consists of 23 columns:
• Four primary columns: ID column, a record timestamp, the recorded demand and the current frequency
• 19 columns indicating the amount contributed by each source.

### Challenge Problem:
Design a dashboard which reads in the file data, cleans it as necessary and presents the data. Charts should also show a moving average overlay. As well as presenting the overall dataset, you should provide summaries of the data. The summary data should be constructed from the data presented (i.e. not pre-computed) - you might think of reading the data into an in-memory database, such as sqllite, and computing the summaries there. Summaries should include year-on-year averages, daily/weekly/yearly peak and trough demand As an additional bonus, consider adding annotations indicating “interesting” features of the dataset.

### Notes
All features noted in the dashboards should be computed from the data provided to the dashboard, not pre-computed.

## Repo Structure
```
.
├── .streamlit/
│   ├── config.toml          # Streamlit configuration file for page layout and settings
├── dashboard.py             # Main dashboard script containing the logic and visualizations
├── gridwatch.csv            # Default dataset for dashboard
├── requirements.txt         # Python dependencies required for the project
└── README.md                # Project summary and setup instructions (this file)
```

## Running the App
There is a live version being hosted on streamlit which you may access via this [link](https://ukgridwatch.streamlit.app/). However, due to their limitations, you can only upload datasets up to 100MB. Hence if you'd like to run the entire dataset from Gridwatch, you will need to run a local version.
### Prerequisites
```
git clone https://github.com/yourusername/uk-gridwatch.git
cd uk-gridwatch
pip install -r requirements.txt
```

### Run
```
streamlit run dashboard.py
```

## Dashboard Overview
The dashboard visualizes UK power grid data from the gridwatch.csv dataset which can be downloaded from [https://www.gridwatch.templar.co.uk/](https://www.gridwatch.templar.co.uk/). It includes key features such as:

- Overall energy generation summaries.
- Gauge charts for monitoring energy demand and frequency.
- Year-on-Year and Peak-Trough-Moving Average charts.
- Customizable time ranges for dynamic data exploration.
- CSV file upload for analysis on other ranges besides default dataset.
- Dynamic annotations on extreme outliers.

## Functions
### `load_stg(path: str)`
Reads a CSV file and performs the following tasks:
- Cleans column names by stripping white spaces.
- Converts the `timestamp` column to datetime.
- Drops duplicate rows and the `id` column.
- Converts all numeric columns (except `frequency`) from MW to GW.
- Returns a DataFrame with cleaned data and a list of numeric columns.

**Parameters:**
- `path`: The path to the CSV file.

**Returns:**
- A `DataFrame` with cleaned data.
- A list of numeric columns in the dataset.

---

### `create_db(df)`
Creates an in-memory SQLite3 database from a given DataFrame. The function stores the data in the database and creates an index on the `timestamp` column for faster queries.

**Parameters:**
- `df`: The input `DataFrame` containing the dataset.

**Returns:**
- An SQLite3 connection object linked to the in-memory database.

---

### `get_timestamp(conn)`
Fetches the maximum and minimum timestamps from the dataset stored in the SQLite3 database.

**Parameters:**
- `conn`: SQLite3 connection object.

**Returns:**
- The maximum and minimum timestamps as `datetime` objects.

---

### `get_summary_data(_conn, numeric_columns, min_date, max_date)`
Generates a summary of the data by calculating the average of all numeric columns within a specified date range.

**Parameters:**
- `_conn`: SQLite3 connection object.
- `numeric_columns`: The list of numeric variables to calculate averages.
- `min_date`: The starting date for the summary.
- `max_date`: The ending date for the summary.

**Returns:**
- A `DataFrame` containing the averages of the numeric columns.

---

### `get_pie_data(df)`
Prepares data for plotting a pie chart of energy generation by source. It calculates the percentages of each energy source relative to the total demand. 

**Parameters:**
- `df`: A `DataFrame` from `get_summary_data(_conn, numeric_columns, min_date, max_date)`.

**Returns:**
- A `DataFrame` with energy sources, percentages, total demand, and power values for each source.

---

### `plot_pie(df)`
Generates a pie chart of energy generation by metered source using Plotly.

**Parameters:**
- `df`: A `DataFrame` from `get_pie_data(df)`.

**Returns:**
- A pie chart displayed in Streamlit.

---

### `get_gauge_data(df, variable)`
Fetches the necessary data to create a gauge chart.

**Parameters:**
- `df`: A `DataFrame` containing the data.
- `variable`: The variable to be plotted on the gauge.

**Returns:**
- The value of the indicator, suffix (e.g., GW, Hz), title, and the maximum bound for the gauge.

---

### `plot_gauge(df, variable, font_size)`
Plots a gauge chart using Plotly for a given variable (e.g., demand, frequency) with specified font size.

**Parameters:**
- `df`: A `DataFrame` with the data from `get_gauge_data(df, variable)`.
- `variable`: The variable to plot on the gauge chart.
- `font_size`: Font size for the gauge number.

**Returns:**
- A gauge chart displayed in Streamlit.

---

### `plot_gauge_summary(df)`
Creates a set of gauge charts to summarize the energy data. It generates gauges for all numeric columns except `timestamp`, `frequency`, and `demand`.

**Parameters:**
- `df`: A `DataFrame` containing the data.

**Returns:**
- Multiple gauge charts displayed in two rows evenly in Streamlit.

---

### `get_yoy_data(_conn, variable)`
Fetches the year-on-year monthly average data for a given variable.

**Parameters:**
- `_conn`: SQLite3 connection object.
- `variable`: The variable to fetch year-on-year data.

**Returns:**
- A `DataFrame` with monthly average values across different years.

---

### `plot_yoy(df, variable)`
Plots a year-on-year line chart for a specified variable. The chart shows monthly averages across different years.

**Parameters:**
- `df`: A pivoted `DataFrame` containing year-on-year data from `get_yoy_data(_conn, variable)`.
- `variable`: The variable to plot.

**Returns:**
- A year-on-year line chart displayed in Streamlit.

---

### `get_peak_trough_moving_avg_data(_conn, variable, period, window_size, min_date, max_date, min_year, min_month, max_year, max_month, period_map)`
Fetches data for peak, trough, and moving averages over a specified time interval (daily, weekly, monthly, or yearly) within a specified period.

**Parameters:**
- `_conn`: SQLite3 connection object.
- `variable`: The variable to calculate peak, trough, and moving averages.
- `period`: The interval period (e.g., daily, weekly).
- `window_size`: The size of the moving average window.
- `min_date`, `max_date`: Date range for filtering the data.
- `min_year`, `max_year`: Year range for filtering the data.
- `min_month`, `max_month`: Month range for filtering the data.
- `period_map`: A mapping to convert period names to SQL expressions.

**Returns:**
- A `DataFrame` with peak, trough, and moving average values for the specified period.

---

### `add_ext_anno(df, type, fig, y_val, bgcolor, font_color, ay=-30)`
Annotates extreme values on a Plotly chart. It identifies values that are more than two standard deviations away from the mean.

**Parameters:**
- `df`: A `DataFrame` containing the data.
- `type`: Either `'max'` or `'min'` to indicate which extremes to annotate.
- `fig`: The Plotly figure to add annotations to.
- `y_val`: The column name containing the values to annotate, Either `'peak'` or `'trough'`.
- `bgcolor`: Background color for the annotations.
- `font_color`: Font color for the annotation text.
- `ay`: Y-axis offset for the annotation arrow.

---

### `plot_peak_trough_moving_avg(df, variable, period, show_anno)`
Plots the peak, trough, and moving averages for a given variable over a specified time period with the option of showing and hiding annotations for extreme values.

**Parameters:**
- `df`: A `DataFrame` with peak, trough, and moving average data from `get_peak_trough_moving_avg_data(_conn, variable, period, window_size, min_date, max_date, min_year, min_month, max_year, max_month, period_map)`.
- `variable`: The variable to plot.
- `period`: The interval period (e.g., daily, weekly).
- `show_anno`: Whether to show annotations for extreme values (`Bool`).

**Returns:**
- A line chart with peak, trough, and moving averages displayed in Streamlit.
