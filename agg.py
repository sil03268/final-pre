import pandas as pd
df = pd.read_csv("all.csv", dtype={"code": str}, parse_dates=["date"], low_memory=False)
df["has_upjong"] = df.upjong.notna()
df.groupby("date").agg(n=("code", "nunique"), n_upjong=("has_upjong", "sum")).to_csv("agg_daily.csv")
df.groupby("code").date.agg(["min", "max", "count"]).to_csv("agg_stock.csv")
