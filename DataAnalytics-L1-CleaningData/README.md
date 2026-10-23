
# Task 3: Data Cleaning and Preprocessing

## Project Overview
This project demonstrates data cleaning and preprocessing using Python and Pandas. It evaluates data quality, handles suitable missing values, removes duplicate rows, standardizes data formats, and identifies potential numeric outliers.

## Objectives
- Inspect dataset structure and data types.
- Measure missing values and duplicate records.
- Handle missing numeric and categorical values using suitable strategies.
- Standardize column names, text, dates, and numeric types.
- Detect potential outliers using the Interquartile Range (IQR) method.
- Export cleaned data and before-and-after quality reports.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- VS Code

## Methodology
1. Load the input CSV.
2. Inspect the original data quality.
3. Standardize column names and trim text whitespace.
4. Remove duplicate rows.
5. Convert recognized date and numeric columns to appropriate types.
6. Fill missing numeric values with medians and categorical values with modes when possible.
7. Report remaining missing dates and other unresolved values.
8. Detect potential outliers using IQR bounds without automatically deleting them.
9. Save the cleaned dataset, reports, and visualizations.

## Output Files
All generated files are saved in the `outputs` directory:
- `cleaned_data.csv`
- `before_after_summary.csv`
- `missing_values_report.csv`
- `outlier_report.csv`
- `missing_values_comparison.png`
- `numeric_outlier_boxplot.png`

## How to Run
Install dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn
```

Run the program:

```bash
python main.py
```

## Conclusion
The project produces a cleaned dataset and reports that help assess data quality before further analysis. Review the reports to confirm which cleaning actions were applied and whether any missing values or outliers require additional treatment.
