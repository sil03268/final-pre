import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("all.csv", dtype={"code": str}, parse_dates=["date"])
df["market"] = df["market"].fillna("미분류(ETF 등)")

# ---- 증거 수치 ----
days = df.date.drop_duplicates().sort_values()
bdays = pd.bdate_range(days.min(), days.max())
missing = bdays.difference(days)
print("행", len(df), "| 종목", df.code.nunique(), "| 거래일", len(days))
print("기간", days.min().date(), "~", days.max().date())
print("평일 중 폴더 없는 날(공휴일 포함)", len(missing))
print("미분류(upjong 없음) 종목", df[df.upjong.isna()].code.nunique())
print("--- 결측률(%) ---")
print((df.isna().mean() * 100).round(2).to_string())

cnt = df.groupby("date").code.nunique()
life = df.groupby("code").date.agg(["min", "max", "count"])

fig, ax = plt.subplots(2, 2, figsize=(14, 9))

# A
ax[0, 0].plot(cnt.index, cnt.values)
for m in missing:
    ax[0, 0].axvline(m, color="red", alpha=0.08)
ax[0, 0].set_title("A. Stocks read per day (red = no folder)")

# B
ax[0, 1].hist(life["count"], bins=40)
ax[0, 1].set_title("B. Trading days per stock")
ax[0, 1].set_xlabel("days observed")
ax[0, 1].set_ylabel("stocks")

# C
mk = df.groupby(["date", "market"]).code.nunique().unstack().fillna(0)
ax[1, 0].stackplot(mk.index, mk.T.values, labels=mk.columns)
ax[1, 0].legend(loc="upper left", fontsize=7)
ax[1, 0].set_title("C. Stocks by market over time")

# D
na = (df.isna().mean() * 100).sort_values()
ax[1, 1].barh(na.index, na.values)
ax[1, 1].set_title("D. Missing rate by column (%)")

plt.tight_layout()
plt.savefig("fig_scope.png", dpi=150)
print("done")

