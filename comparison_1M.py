import pandas as pd
import polars as pl
import time
import numpy as np

# Creating synthetic data
rows = 1_000_000
df_pandas = pd.DataFrame({
    "id": np.arange(rows),
    "group": np.random.choice(['A', 'B', 'C', 'D'], rows),
    "value": np.random.rand(rows),
})
df_pandas.to_csv("test_data_1M.csv", index=False)

# Benchmarking Function
def benchmark_libraries():
    results = {}

    # === Pandas ===
    start = time.time()
    df_pd = pd.read_csv("test_data_1M.csv")
    results["Pandas_Read"] = time.time() - start

    start = time.time()
    agg_pd = df_pd.groupby("group")["value"].mean()
    results["Pandas_Agg"] = time.time() - start

    start = time.time()
    filt_pd = df_pd[df_pd["value"] > 0.5]
    results["Pandas_Filter"] = time.time() - start

    start = time.time()
    joined_pd = df_pd.merge(filt_pd, on="id", how="inner")
    results["Pandas_Join"] = time.time() - start

    # === Polars ===
    start = time.time()
    df_pl = pl.read_csv("test_data_1M.csv")
    results["Polars_Read"] = time.time() - start

    start = time.time()
    agg_pl = df_pl.group_by("group").agg(pl.col("value").mean())
    results["Polars_Agg"] = time.time() - start

    start = time.time()
    filt_pl = df_pl.filter(pl.col("value") > 0.5)
    results["Polars_Filter"] = time.time() - start

    start = time.time()
    joined_pl = df_pl.join(filt_pl, on="id", how="inner")
    results["Polars_Join"] = time.time() - start

    return results

# Running benchmark
result = benchmark_libraries()

# Display results with comparison
print(f"{'Operation':<20} {'Pandas (s)':<15} {'Polars (s)':<15} {'Comparison':<25}")
print("-" * 75)

operations = ["Read", "Agg", "Filter", "Join"]
for op in operations:
    pandas_time = result[f"Pandas_{op}"]
    polars_time = result[f"Polars_{op}"]

    if polars_time < pandas_time:
        speed = pandas_time / polars_time
        comparison = f"🔼 {speed:.1f}× faster"
    elif abs(polars_time - pandas_time) / pandas_time < 0.05:
        comparison = "⚖️ Similar speed"
    else:
        slowdown = polars_time / pandas_time
        comparison = f"🔽 {slowdown:.1f}× slower"

    op_label = {
        "Read": "Read Data",
        "Agg": "Aggregation",
        "Filter": "Filtering",
        "Join": "Joining"
    }[op]

    print(f"{op_label:<20} {pandas_time:<15.6f} {polars_time:<15.6f} {comparison:<25}")
