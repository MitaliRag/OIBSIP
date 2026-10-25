
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent

PROJECTS = {
    "Android App Market Analysis":
        ROOT / "Data Analytics-Level 2-Android App Market Analysis",
    "Fraud Detection":
        ROOT / "Data Analytics-Level 2-Fraud Detection",
    "House Price Prediction":
        ROOT / "Data Analytics-Level 2-House Price Prediction",
    "Wine Quality Prediction":
        ROOT / "Data Analytics-Level 2-Wine Quality Prediction",
}

st.set_page_config(
    page_title="OIBSIP Level 2 Analytics Hub",
    page_icon="📊",
    layout="wide",
)

st.title("📊 OIBSIP Level 2 Analytics Hub")
st.caption("Android Apps • Fraud Detection • House Prices • Wine Quality")

@st.cache_data
def find_csv_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []

    excluded = {".git", ".venv", "venv", "__pycache__"}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".xlsx"}
        and not any(part in excluded for part in p.parts)
    )

@st.cache_data
def read_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)

def number_columns(df):
    return df.select_dtypes(include="number").columns.tolist()

def find_column(df, words):
    columns = {str(c).strip().lower(): c for c in df.columns}
    for word in words:
        if word in columns:
            return columns[word]
    for c in df.columns:
        if any(word in str(c).lower() for word in words):
            return c
    return None

def dataset_picker(project, folder):
    files = find_csv_files(folder)

    if not files:
        st.error(f"No CSV or Excel dataset found in: {folder}")
        return None

    # Prefer useful processed files for each project.
    preferences = {
        "Android App Market Analysis": [
            "cleaned_googleplaystore.csv",
            "category_summary.csv",
            "googleplaystore.csv",
        ],
        "Fraud Detection": [
            "prediction_results.csv",
            "model_comparison.csv",
            "feature_importance.csv",
            "creditcard.csv",
        ],
        "House Price Prediction": [
            "cleaned_housing.csv",
            "prediction_results.csv",
            "model_coefficients.csv",
            "Housing.csv",
        ],
        "Wine Quality Prediction": [
            "prediction_results.csv",
            "feature_importance.csv",
            "winequality-red.csv",
        ],
    }

    names = {p.name.lower(): p for p in files}
    default = next(
        (names[n.lower()] for n in preferences.get(project, [])
         if n.lower() in names),
        files[0],
    )

    selected = st.selectbox(
        "Select dataset or results file",
        files,
        index=files.index(default),
        format_func=lambda p: str(p.relative_to(ROOT)),
        key=f"file_{project}",
    )

    try:
        df = read_dataset(str(selected))
    except Exception as exc:
        st.error(f"Unable to load file: {exc}")
        return None

    if df.empty:
        st.warning("This file contains no rows.")
        return None

    st.caption(f"Loaded: {selected.relative_to(ROOT)}")
    return df

def show_overview(df):
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns):,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    with st.expander("Preview data", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Column details"):
        details = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique values": df.nunique(dropna=True).values,
        })
        st.dataframe(details, use_container_width=True)

def show_explorer(df):
    nums = number_columns(df)

    st.subheader("Interactive data explorer")

    if nums:
        selected_num = st.selectbox(
            "Choose a numeric column",
            nums,
            key=f"hist_{len(df)}_{len(nums)}",
        )
        values = pd.to_numeric(df[selected_num], errors="coerce").dropna()

        if not values.empty:
            hist_df = pd.DataFrame({selected_num: values})
            st.plotly_chart(
                px.histogram(
                    hist_df,
                    x=selected_num,
                    title=f"Distribution of {selected_num}",
                    marginal="box",
                ),
                use_container_width=True,
            )

    categories = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 50
        and df[c].nunique(dropna=True) > 1
    ]

    if categories:
        category = st.selectbox(
            "Choose a category or label column",
            categories,
            key=f"category_{len(df)}_{len(categories)}",
        )
        counts = (
            df[category].fillna("Missing").astype(str)
            .value_counts().head(20)
            .rename_axis("Category")
            .reset_index(name="Count")
        )
        st.plotly_chart(
            px.bar(
                counts,
                x="Category",
                y="Count",
                title=f"Records by {category}",
            ),
            use_container_width=True,
        )

    if len(nums) >= 2:
        with st.expander("Numeric correlation matrix"):
            corr = df[nums].corr(numeric_only=True)
            st.plotly_chart(
                px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation matrix",
                ),
                use_container_width=True,
            )

    st.download_button(
        "Download selected dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="level2_selected_data.csv",
        mime="text/csv",
    )

def show_project_insights(project, df):
    st.subheader("Project-specific insights")
    nums = number_columns(df)

    if project == "Android App Market Analysis":
        category_col = find_column(df, ["category", "genre"])
        rating_col = find_column(df, ["rating"])
        installs_col = find_column(df, ["installs"])
        price_col = find_column(df, ["price"])

        a, b, c = st.columns(3)

        if rating_col:
            rating = pd.to_numeric(df[rating_col], errors="coerce")
            a.metric("Average rating", f"{rating.mean():.2f}" if rating.notna().any() else "N/A")
        else:
            a.metric("Average rating", "Column not found")

        if installs_col:
            installs = pd.to_numeric(
                df[installs_col].astype(str).str.replace(",", "", regex=False)
                .str.replace("+", "", regex=False),
                errors="coerce",
            )
            b.metric("Total installs (sum)", f"{installs.sum():,.0f}" if installs.notna().any() else "N/A")
        else:
            b.metric("Installs", "Column not found")

        if price_col:
            price = pd.to_numeric(
                df[price_col].astype(str).str.replace("$", "", regex=False),
                errors="coerce",
            )
            c.metric("Average price", f"{price.mean():.2f}" if price.notna().any() else "N/A")
        else:
            c.metric("Price", "Column not found")

        if category_col:
            st.caption(f"Category field detected: {category_col}")

    elif project == "Fraud Detection":
        target = find_column(df, ["class", "is_fraud", "fraud", "target", "label"])
        if target:
            counts = df[target].fillna("Missing").astype(str).value_counts()
            a, b = st.columns(2)
            a.metric("Label groups", len(counts))
            b.metric("Largest group", f"{counts.max():,}")
            st.plotly_chart(
                px.bar(
                    counts.rename_axis("Label").reset_index(name="Records"),
                    x="Label",
                    y="Records",
                    title=f"Label distribution: {target}",
                ),
                use_container_width=True,
            )
            st.caption("Labels are displayed as stored; confirm which label means fraud.")
        else:
            st.info(
                "No obvious fraud-label column was detected. "
                "Select prediction results or model comparison to inspect existing outputs."
            )

    elif project == "House Price Prediction":
        target = find_column(df, ["price", "saleprice", "house_price", "median_house_value"])
        if target:
            prices = pd.to_numeric(df[target], errors="coerce").dropna()
            if not prices.empty:
                a, b, c = st.columns(3)
                a.metric("Average price", f"{prices.mean():,.2f}")
                b.metric("Median price", f"{prices.median():,.2f}")
                c.metric("Maximum price", f"{prices.max():,.2f}")
                st.plotly_chart(
                    px.histogram(
                        pd.DataFrame({target: prices}),
                        x=target,
                        title="House price distribution",
                    ),
                    use_container_width=True,
                )
        else:
            st.info(
                "No obvious house-price column was detected in this file. "
                "Try the cleaned housing dataset or prediction results."
            )

    elif project == "Wine Quality Prediction":
        quality = find_column(df, ["quality", "predicted_quality", "prediction"])
        if quality:
            values = pd.to_numeric(df[quality], errors="coerce").dropna()
            if not values.empty:
                st.metric("Average quality/prediction", f"{values.mean():.2f}")
                st.plotly_chart(
                    px.histogram(
                        pd.DataFrame({quality: values}),
                        x=quality,
                        title=f"Distribution of {quality}",
                    ),
                    use_container_width=True,
                )
        else:
            st.info(
                "No obvious quality or prediction column was detected. "
                "Try the wine prediction results or feature-importance file."
            )

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home"] + list(PROJECTS.keys()),
)
st.sidebar.divider()
st.sidebar.caption("OIBSIP • Data Analytics • Level 2")

if page == "Home":
    st.subheader("Your four Level 2 projects")

    descriptions = {
        "Android App Market Analysis": "App ratings, installs, categories and pricing.",
        "Fraud Detection": "Transaction data, class distribution and model results.",
        "House Price Prediction": "Housing data, prices and prediction outputs.",
        "Wine Quality Prediction": "Wine features, quality and model results.",
    }

    cols = st.columns(4)
    for col, (name, folder) in zip(cols, PROJECTS.items()):
        with col:
            st.markdown(f"### {name}")
            st.caption(descriptions[name])
            st.write("Folder available" if folder.exists() else "Folder not found")

    st.info("Choose a project from the sidebar to explore its datasets and charts.")

    status = []
    for name, folder in PROJECTS.items():
        files = find_csv_files(folder)
        status.append({
            "Project": name,
            "Folder exists": folder.exists(),
            "CSV/XLSX files": len(files),
        })
    st.subheader("Dataset status")
    st.dataframe(pd.DataFrame(status), hide_index=True, use_container_width=True)

else:
    st.header(page)
    df = dataset_picker(page, PROJECTS[page])

    if df is not None:
        show_overview(df)
        show_project_insights(page, df)
        show_explorer(df)

st.sidebar.caption("Datasets are read-only; original files are not modified.")
