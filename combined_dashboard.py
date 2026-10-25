
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

PROJECTS = {
    "Cleaning Data": ROOT / "DataAnalytics-L1-CleaningData",
    "Customer Segmentation": ROOT / "DataAnalytics-L1-CustomerSegmentation",
    "EDA Retail Sales": ROOT / "DataAnalytics-L1-EDARetailSales",
    "Sentiment Analysis": ROOT / "DataAnalytics-L1-SentimentAnalysis",
}

st.set_page_config(
    page_title="OIBSIP Level 1 Analytics Hub",
    page_icon="📊",
    layout="wide",
)

st.title("📊 OIBSIP Level 1 Analytics Hub")
st.caption("Four projects • One interactive dashboard")

# --------------------------------------------------
# DATA DISCOVERY AND LOADING
# --------------------------------------------------

@st.cache_data
def discover_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if any(part.lower() in {"outputs", "screenshots", ".git", ".venv", "venv", "__pycache__"}
               for part in path.parts):
            continue

        if path.suffix.lower() in {".csv", ".xlsx"}:
            files.append(path)

    return sorted(files)


@st.cache_data
def load_data(file_path):
    path = Path(file_path)

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    # Try common encodings to support varied CSV datasets.
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path, encoding="latin-1", low_memory=False)


def choose_dataset(project_name, folder):
    files = discover_files(folder)

    if not files:
        st.warning(
            f"No CSV or XLSX dataset found in the **{project_name}** folder. "
            "Check that your dataset is saved there."
        )
        return None

    selected = st.selectbox(
        f"Choose dataset for {project_name}",
        files,
        format_func=lambda p: str(p.relative_to(ROOT)),
        key=f"dataset_{project_name}",
    )

    try:
        df = load_data(str(selected))
    except Exception as error:
        st.error(f"Could not load this dataset: {error}")
        return None

    if df.empty:
        st.warning("The selected dataset is empty.")
        return None

    return df


def show_overview(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    with st.expander("Preview dataset", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Column information"):
        info = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data type": df.dtypes.astype(str).values,
            "Missing values": df.isna().sum().values,
            "Unique values": df.nunique(dropna=True).values,
        })
        st.dataframe(info, use_container_width=True)


def numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def find_column(df, candidates):
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    for col in df.columns:
        name = str(col).strip().lower()
        if any(candidate in name for candidate in candidates):
            return col

    return None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a project",
    [
        "Home",
        "Cleaning Data",
        "Customer Segmentation",
        "EDA Retail Sales",
        "Sentiment Analysis",
    ],
)

st.sidebar.divider()
st.sidebar.caption("OIBSIP • Data Analytics • Level 1")

# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "Home":
    st.subheader("Project overview")

    cols = st.columns(4)
    descriptions = [
        ("🧹", "Cleaning Data", "Missing values, duplicates and data quality"),
        ("👥", "Customer Segmentation", "Customer patterns and clustering"),
        ("🛍️", "EDA Retail Sales", "Sales performance and trends"),
        ("💬", "Sentiment Analysis", "Sentiment labels and text exploration"),
    ]

    for col, (emoji, name, description) in zip(cols, descriptions):
        with col:
            st.markdown(f"### {emoji}")
            st.markdown(f"**{name}**")
            st.caption(description)

    st.info("Select a project from the sidebar to explore its dataset.")

    summary = []
    for name, folder in PROJECTS.items():
        files = discover_files(folder)
        summary.append({
            "Project": name,
            "Folder found": folder.exists(),
            "Datasets found": len(files),
        })

    st.subheader("Project dataset status")
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

# --------------------------------------------------
# CLEANING DATA
# --------------------------------------------------

elif page == "Cleaning Data":
    st.header("🧹 Data Cleaning")
    df = choose_dataset(page, PROJECTS[page])

    if df is not None:
        show_overview(df)

        st.subheader("Clean dataset")
        st.caption(
            "Choose operations below. The original file on disk will not be changed."
        )

        fill_missing = st.checkbox("Fill missing numeric values with column medians")
        remove_duplicates = st.checkbox("Remove duplicate rows")
        drop_empty = st.checkbox("Remove columns that are completely empty")

        cleaned = df.copy()

        if fill_missing:
            for col in numeric_columns(cleaned):
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

            # Fill remaining missing values in text/category columns.
            for col in cleaned.select_dtypes(
                include=["object", "category", "string"]
            ).columns:
                if cleaned[col].isna().any():
                    cleaned[col] = cleaned[col].fillna("Unknown")

        if remove_duplicates:
            cleaned = cleaned.drop_duplicates()

        if drop_empty:
            cleaned = cleaned.dropna(axis=1, how="all")

        st.write(f"Rows after cleaning: **{len(cleaned):,}**")
        st.dataframe(cleaned.head(100), use_container_width=True)

        st.download_button(
            "⬇️ Download cleaned CSV",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

elif page == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    df = choose_dataset(page, PROJECTS[page])

    if df is not None:
        show_overview(df)

        nums = numeric_columns(df)

        if len(nums) < 2:
            st.warning(
                "This dataset has fewer than two numeric columns. "
                "Choose a suitable customer dataset with numeric features."
            )
        else:
            selected_features = st.multiselect(
                "Choose numeric customer features",
                nums,
                default=nums[:min(3, len(nums))],
            )

            max_k = min(10, max(2, len(df) - 1))
            k = st.slider("Number of clusters", 2, max_k, min(3, max_k))

            if len(selected_features) < 2:
                st.info("Select at least two numeric features.")
            elif len(df) < k:
                st.warning("There are not enough rows for this number of clusters.")
            else:
                work = df[selected_features].replace([float("inf"), float("-inf")], pd.NA)
                work = work.apply(pd.to_numeric, errors="coerce")
                valid = work.dropna()

                if len(valid) < k:
                    st.warning("Not enough complete rows to perform clustering.")
                else:
                    scaled = StandardScaler().fit_transform(valid)
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(scaled)

                    result = df.loc[valid.index].copy()
                    result["Cluster"] = labels.astype(str)

                    a, b = st.columns(2)
                    a.metric("Customers clustered", f"{len(result):,}")
                    b.metric("Clusters", k)

                    x_axis = st.selectbox(
                        "X-axis", selected_features, index=0
                    )
                    y_axis = st.selectbox(
                        "Y-axis", selected_features, index=1
                    )

                    fig = px.scatter(
                        result,
                        x=x_axis,
                        y=y_axis,
                        color="Cluster",
                        hover_data=selected_features,
                        title="Customer Clusters",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Cluster sizes")
                    counts = (
                        result["Cluster"]
                        .value_counts()
                        .sort_index()
                        .rename_axis("Cluster")
                        .reset_index(name="Customers")
                    )
                    st.plotly_chart(
                        px.bar(counts, x="Cluster", y="Customers"),
                        use_container_width=True,
                    )

                    st.download_button(
                        "⬇️ Download customer segments",
                        result.to_csv(index=False).encode("utf-8"),
                        "customer_segments.csv",
                        "text/csv",
                    )

# --------------------------------------------------
# EDA RETAIL SALES
# --------------------------------------------------

elif page == "EDA Retail Sales":
    st.header("🛍️ Retail Sales Analysis")
    df = choose_dataset(page, PROJECTS[page])

    if df is not None:
        show_overview(df)
        nums = numeric_columns(df)

        if not nums:
            st.warning("No numeric columns were found for sales analysis.")
        else:
            sales_col = find_column(
                df, ["total sales", "sales", "revenue", "amount", "total"]
            )

            if sales_col not in nums:
                sales_col = st.selectbox(
                    "Select a sales/value column", nums, key="sales_col"
                )

            category_col = find_column(
                df, ["category", "product", "region", "store", "segment"]
            )

            date_col = find_column(
                df, ["date", "order date", "sale date", "transaction date"]
            )

            metric = pd.to_numeric(df[sales_col], errors="coerce")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total value", f"{metric.sum():,.2f}")
            c2.metric("Average value", f"{metric.mean():,.2f}")
            c3.metric("Records", f"{metric.notna().sum():,}")

            if category_col is not None:
                grouped = df.assign(_value=metric).groupby(
                    category_col, dropna=False
                )["_value"].sum().reset_index()

                grouped = grouped.sort_values("_value", ascending=False).head(20)

                st.subheader(f"Value by {category_col}")
                st.plotly_chart(
                    px.bar(
                        grouped,
                        x=category_col,
                        y="_value",
                        title=f"Value by {category_col}",
                    ),
                    use_container_width=True,
                )

            if date_col is not None:
                dated = df[[date_col]].copy()
                dated["_value"] = metric
                dated[date_col] = pd.to_datetime(
                    dated[date_col], errors="coerce"
                )
                dated = dated.dropna(subset=[date_col])

                if not dated.empty:
                    dated["_month"] = (
                        dated[date_col].dt.to_period("M").astype(str)
                    )
                    trend = dated.groupby("_month", as_index=False)["_value"].sum()

                    st.subheader("Sales trend by month")
                    st.plotly_chart(
                        px.line(
                            trend,
                            x="_month",
                            y="_value",
                            markers=True,
                        ),
                        use_container_width=True,
                    )

            st.subheader("Explore numeric relationships")
            x = st.selectbox("X-axis", nums, key="retail_x")
            y = st.selectbox(
                "Y-axis", nums, index=min(1, len(nums) - 1), key="retail_y"
            )
            st.plotly_chart(
                px.scatter(df, x=x, y=y, title=f"{y} vs {x}"),
                use_container_width=True,
            )

# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

elif page == "Sentiment Analysis":
    st.header("💬 Sentiment Analysis")
    df = choose_dataset(page, PROJECTS[page])

    if df is not None:
        show_overview(df)

        sentiment_col = find_column(
            df, ["sentiment", "sentiment label", "label", "polarity"]
        )
        text_col = find_column(
            df, ["text", "review", "comment", "tweet", "message", "content"]
        )

        if sentiment_col is not None:
            st.subheader("Sentiment distribution")
            counts = (
                df[sentiment_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.pie(
                        counts,
                        names="Sentiment",
                        values="Count",
                        title="Sentiment distribution",
                    ),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    px.bar(
                        counts,
                        x="Sentiment",
                        y="Count",
                        title="Records by sentiment",
                    ),
                    use_container_width=True,
                )

            st.caption(
                f"Using the existing label column: {sentiment_col}"
            )

        elif text_col is not None:
            st.info(
                "No existing sentiment-label column was detected. "
                "The optional preview below uses a simple keyword heuristic, "
                "not a trained sentiment model."
            )

            positive_words = {
                "good", "great", "excellent", "love", "happy", "amazing",
                "best", "wonderful", "positive", "like", "enjoy", "awesome"
            }
            negative_words = {
                "bad", "poor", "terrible", "hate", "sad", "awful",
                "worst", "negative", "disappointed", "angry", "boring"
            }

            def basic_sentiment(value):
                words = set(re.findall(r"[a-z]+", str(value).lower()))
                score = len(words & positive_words) - len(words & negative_words)

                if score > 0:
                    return "Positive"
                if score < 0:
                    return "Negative"
                return "Neutral"

            result = df.copy()
            result["Heuristic Sentiment"] = result[text_col].apply(basic_sentiment)

            counts = result["Heuristic Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Sentiment",
                    y="Count",
                    title="Keyword-based sentiment preview",
                ),
                use_container_width=True,
            )

            st.dataframe(
                result[[text_col, "Heuristic Sentiment"]].head(100),
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Download sentiment preview",
                result.to_csv(index=False).encode("utf-8"),
                "sentiment_preview.csv",
                "text/csv",
            )
        else:
            st.warning(
                "No obvious text or sentiment-label column was detected. "
                "Choose a dataset containing review text or sentiment labels."
            )

st.sidebar.divider()
st.sidebar.caption("Original datasets are read-only in this dashboard.")
