import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Sentiment-Aware Pricing Assistant",
    layout="wide"
)


# ---------- Utility helpers ---------- #
def find_column(candidates, keywords):
    """Try to find a column whose name contains any of the given keywords."""
    keywords = [k.lower() for k in keywords]
    for col in candidates:
        name = col.lower()
        if any(k in name for k in keywords):
            return col
    return candidates[0] if candidates else None


def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Try to parse any likely date columns
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "time"]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


def train_regression_model(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """Train a simple RandomForest regression model and return metrics and model."""
    data = df[feature_cols + [target_col]].dropna()
    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(random_state=random_state, n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    results = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred,
        }
    ).reset_index(drop=True)

    return model, (r2, mae, rmse), results, X_train.columns.tolist()


def build_claim_time_series(df, date_col, value_col=None, freq="M"):
    """Return a monthly time series of claim frequency or value."""
    ts = df.copy()
    ts = ts.dropna(subset=[date_col])
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.set_index(date_col).sort_index()

    if value_col:
        series = ts[value_col].resample(freq).sum()
    else:
        # Just count number of rows as claim frequency
        series = ts.resample(freq).size()
        series.name = "claim_count"

    return series


def forecast_series(series, periods=6):
    """Simple time-series forecast using Holt-Winters exponential smoothing."""
    if len(series) < 5:
        return None, None  # not enough data

    model = ExponentialSmoothing(
        series, trend="add", seasonal=None, damped_trend=True
    ).fit(optimized=True)

    forecast = model.forecast(periods)
    return series, forecast


def analyze_sentiment(df, text_col):
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].dropna().astype(str)

    scores = texts.apply(analyzer.polarity_scores)
    scores_df = pd.DataFrame(list(scores))

    compound_mean = scores_df["compound"].mean()
    pos_pct = (scores_df["compound"] > 0.05).mean() * 100
    neg_pct = (scores_df["compound"] < -0.05).mean() * 100
    neu_pct = 100 - pos_pct - neg_pct

    if compound_mean >= 0.05:
        overall = "Overall sentiment is **Positive**."
    elif compound_mean <= -0.05:
        overall = "Overall sentiment is **Negative**."
    else:
        overall = "Overall sentiment is **Neutral/Mixed**."

    return {
        "compound_mean": compound_mean,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct,
        "overall_label": overall,
    }


def compute_claim_ratio(df, premium_col, claim_col):
    valid = df[[premium_col, claim_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None, None

    overall_ratio = valid[claim_col].sum() / valid[premium_col].sum()
    valid = valid.copy()
    valid["claim_ratio"] = valid[claim_col] / valid[premium_col]
    return overall_ratio, valid["claim_ratio"]


# ---------- Main App ---------- #
def main():
    st.title("Sentiment-Aware Pricing Assistant")
    st.write(
        """
        Upload your insurance pricing and complaints dataset to:
        - Build a **premium prediction model** (regression)  
        - **Forecast claim frequency** using time-series  
        - Perform **text analytics** on customer sentiment  
        - Explore a **dashboard** including premium vs claim ratios
        """
    )

    # --- Data upload --- #
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded_file is None and "df" not in st.session_state:
        st.info("👆 Please upload a CSV file from the sidebar to get started.")
        return

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state["df"] = df
    else:
        df = st.session_state["df"]

    st.subheader("Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    date_candidates = [
        c
        for c in df.columns
        if "date" in c.lower() or "time" in c.lower() or str(df[c].dtype).startswith("datetime64")
    ]

    with st.expander("Step 2: Configure important columns", expanded=True):
        # Premium target
        default_premium = find_column(numeric_cols, ["premium", "price", "amount"])
        premium_col = st.selectbox(
            "Premium column (target for regression)",
            options=numeric_cols,
            index=numeric_cols.index(default_premium) if default_premium in numeric_cols else 0,
        )

        # Features
        default_features = [c for c in numeric_cols if c != premium_col]
        feature_cols = st.multiselect(
            "Feature columns for premium prediction",
            options=[c for c in numeric_cols if c != premium_col],
            default=default_features[:5],
        )

        # Date column
        if not date_candidates:
            date_candidates = df.columns.tolist()
        default_date = find_column(date_candidates, ["claim", "date", "time"])
        date_col = st.selectbox(
            "Date column for claim frequency",
            options=date_candidates,
            index=date_candidates.index(default_date) if default_date in date_candidates else 0,
        )

        # Claim amount / count
        default_claim = find_column(numeric_cols, ["claim", "loss", "paid"])
        claim_col = st.selectbox(
            "Claim amount / claim count column",
            options=numeric_cols,
            index=numeric_cols.index(default_claim) if default_claim in numeric_cols else 0,
        )

        # Complaint / feedback text
        if text_cols:
            default_text = find_column(text_cols, ["complaint", "feedback", "comment", "review"])
            text_col = st.selectbox(
                "Complaint / sentiment text column",
                options=text_cols,
                index=text_cols.index(default_text) if default_text in text_cols else 0,
            )
        else:
            text_col = None
            st.warning("No text columns detected for sentiment analysis.")

    # Persist configuration
    st.session_state["premium_col"] = premium_col
    st.session_state["feature_cols"] = feature_cols
    st.session_state["date_col"] = date_col
    st.session_state["claim_col"] = claim_col
    st.session_state["text_col"] = text_col

    tabs = st.tabs(
        [
            "1. Premium Prediction (Regression)",
            "2. Claim Frequency Forecast (Time Series)",
            "3. Sentiment Analysis",
            "4. Dashboard & Assistant",
        ]
    )

    # ---------- Tab 1: Regression ---------- #
    with tabs[0]:
        st.header("Premium Prediction Model")

        if not feature_cols:
            st.warning("Please select at least one feature column in the configuration above.")
        else:
            if st.button("Train regression model", key="train_model_btn"):
                with st.spinner("Training regression model..."):
                    model, metrics, results, used_features = train_regression_model(
                        df, premium_col, feature_cols
                    )
                    r2, mae, rmse = metrics
                    st.session_state["reg_model"] = model
                    st.session_state["reg_metrics"] = metrics
                    st.session_state["reg_results"] = results
                    st.session_state["reg_features"] = used_features

            if "reg_metrics" in st.session_state:
                r2, mae, rmse = st.session_state["reg_metrics"]
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:,.2f}")
                col3.metric("RMSE", f"{rmse:,.2f}")

                st.subheader("Sample predictions")
                st.dataframe(st.session_state["reg_results"].head(20))

                st.subheader("Quick what-if premium prediction")
                model = st.session_state["reg_model"]
                used_features = st.session_state["reg_features"]

                with st.form("prediction_form"):
                    input_values = {}
                    for colname in used_features:
                        col_min = float(df[colname].min())
                        col_max = float(df[colname].max())
                        default_val = float(df[colname].median())
                        input_values[colname] = st.slider(
                            colname,
                            min_value=col_min,
                            max_value=col_max,
                            value=default_val,
                        )
                    submitted = st.form_submit_button("Predict premium")

                if "prediction_form" in st.session_state or submitted:
                    if submitted:
                        x_new = np.array([[input_values[c] for c in used_features]])
                        pred = model.predict(x_new)[0]
                        st.success(f"Estimated premium: **{pred:,.2f}**")

    # ---------- Tab 2: Time Series ---------- #
    with tabs[1]:
        st.header("Claim Frequency Forecast")

        horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=6)

        if st.button("Build forecast", key="forecast_btn"):
            with st.spinner("Building time series and forecasting claim frequency..."):
                series = build_claim_time_series(df, date_col, value_col=claim_col)
                history, forecast = forecast_series(series, periods=horizon)

                if history is None:
                    st.error("Not enough data points to build a time-series model.")
                else:
                    ts_df = pd.concat(
                        [
                            history.rename("History"),
                            forecast.rename("Forecast"),
                        ],
                        axis=0,
                    )

                    st.subheader("Monthly claim frequency")
                    st.line_chart(ts_df)

                    st.write("Recent data (history + forecast):")
                    st.dataframe(ts_df.tail(horizon + 6))

    # ---------- Tab 3: Sentiment Analysis ---------- #
    with tabs[2]:
        st.header("Text Analytics: Consumer Sentiment on Pricing")

        if text_col is None:
            st.warning(
                "No text column configured. Please make sure your dataset has complaints / feedback text."
            )
        else:
            if st.button("Run sentiment analysis", key="sentiment_btn"):
                with st.spinner("Analyzing customer sentiment..."):
                    summary = analyze_sentiment(df, text_col)
                    st.session_state["sentiment_summary"] = summary

            if "sentiment_summary" in st.session_state:
                s = st.session_state["sentiment_summary"]

                st.markdown("#### Sentiment Insights (Key Points)")
                st.markdown(
                    f"""
                    - {s['overall_label']}  
                    - **Positive** comments: **{s['pos_pct']:.1f}%**  
                    - **Negative** comments: **{s['neg_pct']:.1f}%**  
                    - **Neutral / mixed** comments: **{s['neu_pct']:.1f}%**  
                    - Average sentiment score (compound): **{s['compound_mean']:.3f}**
                    """
                )

                # Simple distribution table
                st.write("Sentiment distribution summary:")
                dist_df = pd.DataFrame(
                    {
                        "Sentiment": ["Positive", "Negative", "Neutral/Mixed"],
                        "Percentage": [
                            s["pos_pct"],
                            s["neg_pct"],
                            s["neu_pct"],
                        ],
                    }
                )
                st.dataframe(dist_df)

    # ---------- Tab 4: Dashboard & Assistant ---------- #
    with tabs[3]:
        st.header("Premium vs Claim Dashboard & Pricing Assistant")

        # Claim ratio
        overall_ratio, ratio_series = compute_claim_ratio(df, premium_col, claim_col)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Average Premium", f"{df[premium_col].mean():,.2f}")
        col_b.metric("Average Claims", f"{df[claim_col].mean():,.2f}")
        if overall_ratio is not None:
            col_c.metric("Overall Claim / Premium Ratio", f"{overall_ratio:.3f}")
        else:
            col_c.metric("Overall Claim / Premium Ratio", "N/A")

        if ratio_series is not None:
            st.subheader("Distribution of Claim / Premium Ratio")
            st.bar_chart(ratio_series.reset_index(drop=True))

        # Recompute sentiment quickly for assistant (if possible)
        if text_col is not None:
            s = analyze_sentiment(df, text_col)
        else:
            s = None

        st.markdown("---")
        st.subheader("Sentiment-Aware Pricing Assistant")

        points = []

        # Claim ratio based guidelines
        if overall_ratio is not None:
            if overall_ratio < 0.5:
                points.append(
                    "Claim ratio is relatively **low**. There may be room to **reduce premiums** "
                    "slightly to improve competitiveness and customer satisfaction."
                )
            elif overall_ratio > 0.9:
                points.append(
                    "Claim ratio is **high**. Consider **reviewing underwriting rules** or **increasing premiums** "
                    "for high-risk segments."
                )
            else:
                points.append(
                    "Claim ratio is **balanced**. Focus on **fine-tuning premiums by segment** rather than "
                    "broad price changes."
                )

        # Sentiment-based guidelines
        if s is not None:
            if s["neg_pct"] > s["pos_pct"]:
                points.append(
                    "Customer sentiment is **more negative than positive**. Review complaints related to pricing "
                    "and consider targeted discounts or loyalty benefits."
                )
            elif s["pos_pct"] > s["neg_pct"] + 10:
                points.append(
                    "Customer sentiment is **strongly positive**. Pricing seems acceptable; focus on **maintaining service quality**."
                )
            else:
                points.append(
                    "Customer sentiment is **mixed**. Run focused campaigns (e.g., clearer communication of coverage and "
                    "benefits) to improve perceived value."
                )

        # Data quality / model guidance
        if "reg_metrics" in st.session_state:
            r2, mae, rmse = st.session_state["reg_metrics"]
            if r2 < 0.5:
                points.append(
                    "Premium prediction model performance (R²) is **moderate/low**. Consider adding more predictive "
                    "features or cleaning outliers."
                )
            else:
                points.append(
                    "Premium prediction model shows **good explanatory power**. You can use it as a starting point "
                    "for scenario-based pricing simulations."
                )

        if not points:
            st.info(
                "Once you configure columns, train the regression model, and run sentiment analysis, "
                "this section will summarise key pricing recommendations."
            )
        else:
            st.markdown("#### Recommended actions (Summary)")
            for p in points:
                st.markdown(f"- {p}")


if __name__ == "__main__":
    main()
