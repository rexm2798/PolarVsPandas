# 🐼 vs 🧊 Pandas vs Polars: DataFrame Performance Benchmark

Welcome to the **Tech Face-Off** benchmarking project, where we put two of the most popular DataFrame libraries — **Pandas** and **Polars** — head-to-head under real-world conditions.

---

## 📌 Project Overview

This project compares the performance of **Pandas** and **Polars** across:
- CSV **Read**
- **Aggregation** (groupby + mean)
- **Filtering** (value > 0.5)
- **Joining** (self-join on ID)

Benchmarks were run on:
- **1 million rows**
- **10 million rows**
- **50 million rows**

All datasets were synthetically generated using NumPy and saved as CSVs.

---

## 🧪 Dataset Description

Each dataset contains:
- `id`: sequential integer IDs
- `group`: random categorical values from `'A'`, `'B'`, `'C'`, `'D'`
- `value`: random float values between 0 and 1
