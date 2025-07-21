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

# Saving it to CSV for reloading
df_pandas.to_csv("test_data.csv", index=False)


# Benchmarking Functionn
def benchmark_libraries():
    results = {}

    # === Pandas ===
    start = time.time()
    df_pd = pd.read_csv("test_data.csv")
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
    df_pl = pl.read_csv("test_data.csv")
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

# Format output
print(f"{'Operation':<20} {'Pandas (s)':<15} {'Polars (s)':<15}")
print("-" * 50)
print(f"{'Read Data':<20} {result['Pandas_Read']:<15.6f} {result['Polars_Read']:<15.6f}")
print(f"{'Aggregation':<20} {result['Pandas_Agg']:<15.6f} {result['Polars_Agg']:<15.6f}")
print(f"{'Filtering':<20} {result['Pandas_Filter']:<15.6f} {result['Polars_Filter']:<15.6f}")
print(f"{'Joining':<20} {result['Pandas_Join']:<15.6f} {result['Polars_Join']:<15.6f}")
