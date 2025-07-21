import pandas as pd
import polars as pl
import numpy as np
import time
import os

#Setp 1: Generate Dataset of 10 milliom rows
def generate_large_csv():
    rows = 10_000_000  # 10 million rows
    print("📦 Generating large dataset...")
    df = pd.DataFrame({
        "id": np.arange(rows),
        "group": np.random.choice(['A', 'B', 'C', 'D'], size=rows),
        "value": np.random.rand(rows)
    })
    df.to_csv("test_data_10M.csv", index=False)
    print("✅ test_data_10M.csv created.")

# Step 2: Benchmarking

def benchmark_libraries():
    results = {}

    # === Pandas ===
    print("🐼 Benchmarking Pandas...")
    start = time.time()
    df_pd = pd.read_csv("test_data_10M.csv")
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
    print("🧊 Benchmarking Polars...")
    start = time.time()
    df_pl = pl.read_csv("test_data_10M.csv")
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


# -----------------------
# Step 3: Run & Display
# -----------------------
if __name__ == "__main__":
    if not os.path.exists("test_data_10M.csv"):
        generate_large_csv()
    else:
        print("📂 test_data_10M.csv already exists. Skipping generation.")

    result = benchmark_libraries()

    print("\n📊 Benchmark Results (10M rows)")
    print(f"{'Operation':<20} {'Pandas (s)':<15} {'Polars (s)':<15} {'Comparison':<25}")
    print("-" * 80)

    operations = ["Read Data", "Aggregation", "Filtering", "Joining"]
    keys = ["Read", "Agg", "Filter", "Join"]
    polar_wins = 0

    for op, key in zip(operations, keys):
        p_time = result[f"Pandas_{key}"]
        pl_time = result[f"Polars_{key}"]

        if pl_time < p_time:
            ratio = p_time / pl_time
            comparison = f"🔼 {ratio:.1f}× faster"
            polar_wins += 1
        elif abs(pl_time - p_time) / p_time < 0.05:
            comparison = "⚖️ Similar speed"
        else:
            ratio = pl_time / p_time
            comparison = f"🔽 {ratio:.1f}× slower"

        print(f"{op:<20} {p_time:<15.6f} {pl_time:<15.6f} {comparison:<25}")

    print("-" * 80)
    print(f"🏁 Polars was faster in {polar_wins} out of {len(operations)} operations.")

