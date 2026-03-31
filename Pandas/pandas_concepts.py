# ============================================================
# PANDAS MAIN CONCEPTS
# ============================================================
# Pandas is a Python library for data manipulation and analysis.
# It provides two core data structures: Series and DataFrame.

import pandas as pd
import numpy as np

# ============================================================
# 1. SERIES
# ============================================================
# A Series is a one-dimensional labeled array (like a column in a spreadsheet).

s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
print("--- Series ---")
print(s)
print("Value at 'b':", s["b"])
print("dtype:", s.dtype)


# ============================================================
# 2. DATAFRAME
# ============================================================
# A DataFrame is a two-dimensional table with labeled rows and columns.

data = {
    "name":   ["Alice", "Bob", "Charlie", "Diana"],
    "age":    [25, 30, 35, 28],
    "salary": [50000, 60000, 70000, 55000],
    "dept":   ["HR", "Eng", "Eng", "HR"],
}
df = pd.DataFrame(data)
print("\n--- DataFrame ---")
print(df)


# ============================================================
# 3. READING & WRITING DATA
# ============================================================
# Pandas reads from and writes to many formats.

# df = pd.read_csv("file.csv")          # Read CSV
# df = pd.read_excel("file.xlsx")       # Read Excel
# df = pd.read_json("file.json")        # Read JSON
# df = pd.read_sql(query, connection)   # Read from SQL DB

# df.to_csv("output.csv", index=False)  # Write CSV
# df.to_excel("output.xlsx")            # Write Excel


# ============================================================
# 4. INSPECTING DATA
# ============================================================

print("\n--- Inspection ---")
print(df.head(2))          # First N rows
print(df.tail(1))          # Last N rows
print(df.shape)            # (rows, columns)
print(df.dtypes)           # Column data types
print(df.info())           # Summary of DataFrame
print(df.describe())       # Statistical summary of numeric columns


# ============================================================
# 5. SELECTING DATA
# ============================================================

print("\n--- Selection ---")
print(df["name"])                        # Select a column (Series)
print(df[["name", "age"]])               # Select multiple columns (DataFrame)
print(df.iloc[0])                        # Row by integer position
print(df.loc[1])                         # Row by label/index
print(df.loc[0:1, "name":"age"])         # Slice rows and columns by label
print(df.iloc[0:2, 0:2])                 # Slice rows and columns by position


# ============================================================
# 6. FILTERING (BOOLEAN INDEXING)
# ============================================================

print("\n--- Filtering ---")
print(df[df["age"] > 28])                          # Rows where age > 28
print(df[(df["dept"] == "Eng") & (df["age"] < 35)]) # Multiple conditions


# ============================================================
# 7. ADDING & MODIFYING COLUMNS
# ============================================================

df["bonus"] = df["salary"] * 0.10         # New column
df["age_group"] = df["age"].apply(lambda x: "Senior" if x >= 30 else "Junior")
print("\n--- Modified DataFrame ---")
print(df)


# ============================================================
# 8. HANDLING MISSING DATA
# ============================================================

df_missing = pd.DataFrame({
    "a": [1, None, 3],
    "b": [None, 5, 6],
})
print("\n--- Missing Data ---")
print(df_missing.isna())                   # Boolean mask of NaN values
print(df_missing.dropna())                 # Drop rows with any NaN
print(df_missing.fillna(0))               # Replace NaN with 0
print(df_missing["a"].fillna(df_missing["a"].mean()))  # Fill with column mean


# ============================================================
# 9. SORTING
# ============================================================

print("\n--- Sorting ---")
print(df.sort_values("salary", ascending=False))         # Sort by column
print(df.sort_values(["dept", "age"], ascending=[True, False]))  # Multi-column sort


# ============================================================
# 10. GROUPBY & AGGREGATION
# ============================================================
# groupby splits data into groups and applies aggregate functions.

print("\n--- GroupBy ---")
grouped = df.groupby("dept")["salary"].mean()
print(grouped)                              # Average salary per department

print(df.groupby("dept").agg(
    avg_salary=("salary", "mean"),
    count=("name", "count"),
    max_age=("age", "max"),
))


# ============================================================
# 11. MERGING & JOINING
# ============================================================

dept_info = pd.DataFrame({
    "dept":     ["HR", "Eng"],
    "location": ["New York", "San Francisco"],
})

merged = pd.merge(df, dept_info, on="dept", how="left")  # SQL-style join
print("\n--- Merged ---")
print(merged[["name", "dept", "location"]])

# pd.concat([df1, df2])  stacks DataFrames vertically (like UNION ALL)
# pd.merge(df1, df2, on="key", how="inner/left/right/outer")


# ============================================================
# 12. PIVOT TABLES
# ============================================================

print("\n--- Pivot Table ---")
pivot = df.pivot_table(values="salary", index="dept", aggfunc="mean")
print(pivot)


# ============================================================
# 13. APPLYING FUNCTIONS
# ============================================================

print("\n--- Apply ---")
print(df["salary"].apply(lambda x: f"${x:,}"))  # Apply function to each value
print(df[["age", "salary"]].apply(np.sqrt))      # Apply to entire DataFrame


# ============================================================
# 14. STRING OPERATIONS
# ============================================================

print("\n--- String Operations ---")
print(df["name"].str.lower())
print(df["name"].str.contains("a", case=False))
print(df["dept"].str.replace("Eng", "Engineering"))


# ============================================================
# 15. DATETIME OPERATIONS
# ============================================================

dates = pd.date_range("2024-01-01", periods=4, freq="ME")  # Monthly end dates
df["hire_date"] = dates
print("\n--- Datetime ---")
print(df["hire_date"].dt.year)
print(df["hire_date"].dt.month_name())
print(df[df["hire_date"] > "2024-02-01"])


# ============================================================
# 16. INDEX OPERATIONS
# ============================================================


print("\n--- Index ---")
df_indexed = df.set_index("name")    # Set a column as the index
print(df_indexed.loc["Alice"])       # Access by index value
df_reset = df_indexed.reset_index()  # Restore default integer index


# ============================================================
# 17. REINDEXING
# ============================================================
# reindex() conforms a DataFrame/Series to a new set of labels.
# Missing labels get NaN by default (or a fill value you specify).

print("\n--- Reindexing ---")

s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s.reindex(["a", "b", "c", "d", "e"]))
# a    10.0
# b    20.0
# c    30.0
# d     NaN   <- new label, no data
# e     NaN

# Fill missing values introduced by reindex
print(s.reindex(["a", "b", "c", "d", "e"], fill_value=0))

# Forward-fill: carry the last known value forward
print(s.reindex(["a", "b", "c", "d", "e"], method="ffill"))

# ---- DataFrame reindex ----
df2 = pd.DataFrame(
    {"price": [100, 200, 300], "qty": [5, 10, 15]},
    index=["item1", "item2", "item3"],
)

# Add new rows
print(df2.reindex(["item1", "item2", "item3", "item4"]))

# Reorder / add columns
print(df2.reindex(columns=["qty", "price", "discount"]))
# discount column is new -> filled with NaN

# ---- Aligning two objects ----
# reindex_like() reshapes one object to match the index/columns of another
s1 = pd.Series([1, 2, 3], index=["x", "y", "z"])
s2 = pd.Series([10, 20], index=["x", "y"])
print(s2.reindex_like(s1))          # z gets NaN

# ---- Common use-case: time series with a complete date range ----
dates = pd.date_range("2024-01-01", periods=5, freq="D")
sparse = pd.Series([100, 300], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
print(sparse.reindex(dates, method="ffill"))  # fill gaps by carrying value forward


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# CREATING
#   pd.Series([...])                     1D labeled array
#   pd.DataFrame({col: [...]})           2D table
#   pd.read_csv / read_excel / read_json Load from file
#
# INSPECTING
#   df.head(n) / df.tail(n)             First/last n rows
#   df.shape / df.dtypes / df.info()    Structure info
#   df.describe()                        Stats summary
#
# SELECTING
#   df["col"]                            Column as Series
#   df[["col1","col2"]]                  Columns as DataFrame
#   df.iloc[row, col]                    By integer position
#   df.loc[row, col]                     By label
#
# FILTERING
#   df[df["col"] > value]                Boolean filter
#   df.query("col > value")              SQL-like filter
#
# CLEANING
#   df.dropna() / df.fillna(val)         Handle NaN
#   df.drop_duplicates()                 Remove duplicates
#   df["col"].astype(type)               Change dtype
#
# TRANSFORMING
#   df["new"] = ...                      Add column
#   df.rename(columns={"old":"new"})     Rename columns
#   df.apply(func)                       Apply function
#   df["col"].str.*                      String methods
#   df["col"].dt.*                       Datetime methods
#
# AGGREGATING
#   df.groupby("col").agg(...)           Group and aggregate
#   df.pivot_table(...)                  Pivot table
#
# COMBINING
#   pd.concat([df1, df2])                Stack vertically
#   pd.merge(df1, df2, on="key")         SQL-style join
#
# REINDEXING
#   df.reindex([...])                    Conform to new row labels
#   df.reindex(columns=[...])            Conform to new column labels
#   df.reindex(..., fill_value=0)        Fill missing with a value
#   df.reindex(..., method="ffill")      Forward-fill gaps
#   df.reindex(..., method="bfill")      Backward-fill gaps
#   df.reindex_like(other)               Match index/columns of another object
