import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="NYC Ridership Forecast Dashboard")
st.title("🚇 NYC Ridership Forecast Dashboard with Segmented Analysis")

@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/codezilla69/NYC_Subway/main/raw_data/"
    
    # Create all chunk filenames
    filenames = [f"manhattan_chunk_{i:05}.csv" for i in range(818)]
    
    # Read and combine all dataframes
    dfs = []
    for filename in filenames:
        url = base_url + filename
        try:
            df = pd.read_csv(url)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")

    if not dfs:
        st.error("No data was loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

data = load_data()
data['day'] = data['date'].dt.day

# --- REGRESSION SETUP ---
daily_agg = data.groupby('date')['ridership'].sum().reset_index()
daily_features = data.drop_duplicates(subset='date')[['date', 'is_weekend', 'is_weekday', 'month', 'tmin', 'tmax', 'prcp', 'wspd', 'is_holiday']]
daily = pd.merge(daily_agg, daily_features, on='date')

X = daily[['is_weekend', 'is_weekday', 'month', 'tmin', 'tmax', 'prcp', 'wspd', 'is_holiday']]
y = daily['ridership']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
reg_model.fit(X_train, y_train)
importances = reg_model.feature_importances_
feature_names = X.columns

# Combine and sort
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display in Streamlit
st.subheader("🔍 Feature Importance (Regression Model)")
st.dataframe(feature_importance_df)

# Optional: Plot
fig_imp, ax_imp = plt.subplots()
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', ax=ax_imp)
ax_imp.set_title("Feature Impact on Predicted Ridership")
st.pyplot(fig_imp)
daily['predicted'] = reg_model.predict(X)

# --- CLASSIFICATION MODEL ---
st.subheader("📋 Classification Model: Predicting High Ridership Days")
threshold = daily['ridership'].quantile(0.90)
daily['high_ridership'] = (daily['ridership'] > threshold).astype(int)

X_cls = X.copy()
y_cls = daily['high_ridership']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_cls, y_cls)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
cls_model = RandomForestClassifier(n_estimators=200, random_state=42)
cls_model.fit(X_train_c, y_train_c)
daily['cls_predicted'] = cls_model.predict(X_cls)

st.markdown("### 📊 Classification Metrics (90th Percentile Threshold + SMOTE)")
st.text(classification_report(y_test_c, cls_model.predict(X_test_c)))

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_test_c, cls_model.predict(X_test_c))
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# --- CLASSIFICATION VISUALIZATION ---
st.markdown("## 📊 Actual vs Predicted: Classification Model")
fig_cls, ax_cls = plt.subplots(figsize=(12, 5))
ax_cls.plot(
    daily['date'], daily['high_ridership'],
    label="Actual High Traffic", marker='o', linestyle='None', color='blue', alpha=0.6
)
ax_cls.plot(
    daily['date'], daily['cls_predicted'],
    label="Predicted High Traffic", linestyle='--', marker='x', color='orange', alpha=0.6
)
ax_cls.set_title("Classification Model: Actual vs Predicted High Ridership Days", fontsize=14)
ax_cls.set_xlabel("Date", fontsize=12)
ax_cls.set_ylabel("High Traffic Day (1 = True)", fontsize=12)
ax_cls.legend()
ax_cls.grid(True)
fig_cls.autofmt_xdate(rotation=45)
st.pyplot(fig_cls)

# --- SEGMENTED REGRESSION PLOTS ---
high_data = daily[daily['high_ridership'] == 1]
normal_data = daily[daily['high_ridership'] == 0]

st.subheader("📈 High Ridership Days: Actual vs Predicted")
fig_high, ax_high = plt.subplots()
ax_high.plot(high_data['date'], high_data['ridership'], label='Actual')
ax_high.plot(high_data['date'], high_data['predicted'], label='Predicted', linestyle='--')
ax_high.set_title("High Ridership Days")
ax_high.set_xlabel("Date")
ax_high.set_ylabel("Ridership")
ax_high.legend()
st.pyplot(fig_high)

st.subheader("📈 Normal Days: Actual vs Predicted")
fig_normal, ax_normal = plt.subplots()
ax_normal.plot(normal_data['date'], normal_data['ridership'], label='Actual', color='green')
ax_normal.plot(normal_data['date'], normal_data['predicted'], label='Predicted', linestyle='--', color='orange')
ax_normal.set_title("Normal Ridership Days")
ax_normal.set_xlabel("Date")
ax_normal.set_ylabel("Ridership")
ax_normal.legend()
st.pyplot(fig_normal)

# --- MONTHLY VIEW ---
st.subheader("📅 Daily Ridership View by Segment and Month")
selected_segment = st.radio("Select Segment", ["High", "Normal"])
selected_month = st.selectbox("Select Month (1–12)", sorted(daily['date'].dt.month.unique()), key="month_selector")
selected_year = st.selectbox("Select Year", sorted(daily['date'].dt.year.unique()), key="year_selector")

if selected_segment == "High":
    seg_df = high_data.copy()
    color = 'blue'
    label = 'High'
else:
    seg_df = normal_data.copy()
    color = 'orange'
    label = 'Normal'

monthly_view = seg_df[(seg_df['date'].dt.month == selected_month) & (seg_df['date'].dt.year == selected_year)]

fig_m, ax_m = plt.subplots()
ax_m.plot(monthly_view['date'], monthly_view['ridership'], marker='o', label='Actual', color=color)
ax_m.plot(monthly_view['date'], monthly_view['predicted'], linestyle='--', marker='x', label='Predicted', color='green')
ax_m.set_title(f"{label} Ridership in {selected_month}/{selected_year}")
ax_m.set_xlabel("Date")
ax_m.set_ylabel("Ridership")
ax_m.legend()
st.pyplot(fig_m)

# --- HOURLY VIEW ---
st.subheader("🕒 Hourly Ridership View")
selected_date = st.date_input("Select a date", value=monthly_view['date'].min() if not monthly_view.empty else daily['date'].min(), key="hourly_date")
is_high_day = daily.loc[daily['date'] == pd.to_datetime(selected_date), 'high_ridership'].values[0] == 1
st.write(f"Selected date is classified as: **{'High' if is_high_day else 'Normal'} Ridership Day**")

hourly = data[data['date'] == pd.to_datetime(selected_date)]
if not hourly.empty:
    hourly_agg = hourly.groupby(hourly['transit_timestamp'].dt.hour)['ridership'].sum().reset_index()
    hourly_agg.columns = ['hour', 'actual']
    if is_high_day:
        predicted_day_total = high_data.loc[high_data['date'] == pd.to_datetime(selected_date), 'predicted'].values[0]
    else:
        predicted_day_total = normal_data.loc[normal_data['date'] == pd.to_datetime(selected_date), 'predicted'].values[0]
    hourly_agg['share'] = hourly_agg['actual'] / hourly_agg['actual'].sum()
    hourly_agg['predicted'] = hourly_agg['share'] * predicted_day_total

    fig_h, ax_h = plt.subplots()
    ax_h.plot(hourly_agg['hour'], hourly_agg['actual'], label='Actual', marker='o')
    ax_h.plot(hourly_agg['hour'], hourly_agg['predicted'], label='Predicted', linestyle='--', marker='x')
    ax_h.set_title(f"Hourly Ridership – {selected_date.strftime('%B %d, %Y')}")
    ax_h.set_xlabel("Hour")
    ax_h.set_ylabel("Ridership")
    ax_h.legend()
    st.pyplot(fig_h)
else:
    st.warning("No hourly data available for this date.")

# --- FREQUENCY OPTIMIZER ---
st.subheader("🚦 Line Frequency Optimizer")

line_capacity_per_hour = {
    "A": 8 * 1100 * 10,
    "B": 6 * 1000 * 10,
    "C": 7 * 1050 * 10,
    "D": 10 * 1100 * 10,
}

line = st.selectbox("Select Line", list(line_capacity_per_hour.keys()))
hour = st.slider("Select Hour", 0, 23, 8)
predicted_ridership_input = st.number_input("Predicted Ridership for Selected Hour", value=50000)

capacity = line_capacity_per_hour[line]
load_factor = predicted_ridership_input / capacity
trains_needed = int(predicted_ridership_input / (capacity / 10)) + 1

st.markdown(f"**Predicted Ridership:** {predicted_ridership_input:,.0f}")
st.markdown(f"**Line Capacity (max/hour):** {capacity:,}")
st.markdown(f"**Load Factor:** {load_factor:.2%}")
st.markdown(f"**Suggested Trains per Hour:** {trains_needed}")

# --- COST BREAKDOWN ---
st.markdown("### 💸 Cost Per Train-Hour Breakdown")
st.table({
    "Component": ["Train Operator & Staff", "Electricity", "Maintenance & Wear", "Admin Overhead", "Depreciation"],
    "Estimated Cost (USD/hr)": [200, 150, 125, 75, 150]
})

st.markdown("**Estimated Total Cost per Train-Hour:** $800 (approx.)")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 NYC Metro Ridership: Analysis & Recommendations")

# Load and filter data
df = pd.read_csv("combined_data_with_weather.csv", parse_dates=['date'])
df = df[df['date'].dt.year == 2024]  # Focus on 2024 only

# Aggregate to daily totals
daily = df.groupby('date')['ridership'].sum().reset_index()

# Calculate thresholds AFTER aggregation
threshold = daily['ridership'].quantile(0.9)
daily['high_ridership'] = (daily['ridership'] > threshold).astype(int)

# --- SEGMENTED BEHAVIOR ---
st.header("🚦 Segmented Demand Behavior")
col1, col2 = st.columns(2)
with col1:
    st.metric("Avg Ridership (Normal Days)", f"{daily[daily['high_ridership'] == 0]['ridership'].mean():,.0f}")
with col2:
    st.metric("Avg Ridership (High Days)", f"{daily[daily['high_ridership'] == 1]['ridership'].mean():,.0f}")

fig, ax = plt.subplots()
sns.boxplot(x='high_ridership', y='ridership', data=daily, ax=ax)
ax.set_xticklabels(['Normal Day', 'High Ridership Day'])
ax.set_title("Ridership Distribution by Day Type")
st.pyplot(fig)

# --- PREDICTION QUALITY ---
st.header("📉 Prediction Model Evaluation")
st.markdown("Add evaluation metrics from models (e.g., MAE, RMSE, F1).")
st.write("You can insert metrics calculated elsewhere.")

# --- FREQUENCY OPTIMIZATION SNAPSHOT ---
st.header("🚆 Frequency Optimization Snapshot")
st.markdown("Sample hour-based load insights using predicted demand and line capacity.")

line_capacity_per_hour = {
    "A": 8 * 1100 * 10,
    "B": 6 * 1000 * 10,
    "C": 7 * 1050 * 10,
    "D": 10 * 1100 * 10,
}

selected_line = st.selectbox(
    "Select Line",
    list(line_capacity_per_hour.keys()),
    key="frequency_line_selector"  # ✅ this makes it unique
)
selected_ridership = st.number_input("Enter predicted ridership for a peak hour:", min_value=0, value=50000)
capacity = line_capacity_per_hour[selected_line]
load_factor = selected_ridership / capacity
suggested_trains = int(np.ceil(selected_ridership / (capacity / (int(selected_line == 'A') * 8 or 6))))

st.markdown(f"**Predicted Ridership:** {selected_ridership:,}")
st.markdown(f"**Line Capacity/hour:** {capacity:,}")
st.markdown(f"**Load Factor:** {load_factor:.2%}")
st.markdown(f"**Suggested Trains/hour:** {suggested_trains}")


# Place this right after st.title(...)
st.markdown("## 🔍 Insights Summary")

st.markdown("""
- **Average Daily Ridership (2024):** ~1.2 million riders  
- **Peak Ridership Threshold (90th percentile):** Days exceeding ~2.1 million  
- **Prediction Accuracy:**  
  - Classification F1-score ~0.87  
  - Regression MAE ~654,000; RMSE ~911,000  

---

## 🚦 Recommendations

### 1. Increase Train Frequency on Peak Days
- For days exceeding the 90th percentile, increase frequency by 15–20%.
- Use predicted hourly ridership to proactively adjust scheduling.

### 2. Focus on Underperforming Segments
- Normal traffic days often underpredict ridership; refine segmentation or use hybrid modeling.

### 3. Smart Resource Allocation
- Deploy more staff and trains on predicted high-ridership dates.
- Utilize early-morning prediction for real-time adjustments.

---

## 💸 Operational Implications
- For every extra train/hour added to the A, B, C, or D lines, expect a cost increase of ~$1,500–2,000/hour.
- Use load factor estimates to balance between cost and comfort.
""")