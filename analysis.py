# analysis.py - 공통 질문 1~4: 그림 4장 + memo.md 생성
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

names = {f.name for f in fm.fontManager.ttflist}
kr = next((n for n in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"] if n in names), None)
if kr:
    plt.rcParams["font.family"] = kr
plt.rcParams["axes.unicode_minus"] = False
memo = []
def log(s):
    print(s); memo.append(s)

df = pd.read_csv("all.csv", dtype={"code": str}, parse_dates=["date"], low_memory=False)
df["code"] = df["code"].str.split("_").str[-1]
n_all = len(df)
df = df[df.upjong.notna() & df.mktcap_eok.notna()].copy()   # 업종·시총 있는 종목(주식)만. ETF 등 제외
log(f"[대상] 전체 {n_all:,}행 중 업종·시총이 있는 {len(df):,}행, 종목 {df.code.nunique()}개")
df = df.sort_values(["code", "date"]).reset_index(drop=True)
lab = df.groupby("upjong_no").upjong.agg(lambda s: s.mode()[0])
L = lambda no: lab[no] if kr else str(no)

# ---- 일별 수익률(액면분할·결측 처리) ----
g = df.groupby("code")
pp, ps, pd_ = g.price.shift(), g.shares.shift(), g.date.shift()
df["ret"] = df.price / pp - 1
split = (df.shares / ps - 1).abs() > 0.2          # 주식수 20% 이상 변동 = 분할/병합/증자
mc, pmc = df.price * df.shares, pp * ps
df.loc[split, "ret"] = (mc / pmc - 1)[split]      # 그날은 시총 기준 수익률 사용
gap = (df.date - pd_).dt.days > 10                # 10일 넘게 끊긴 구간은 수익률 제외
df.loc[gap, "ret"] = np.nan
out = df.ret.abs() > 0.31                         # 상하한가 30% 초과 = 데이터 오류로 봄
df.loc[out, "ret"] = np.nan
log(f"[수익률 처리] 분할/증자 보정 {int(split.sum()):,}행, 공백 제외 {int(gap.sum()):,}행, ±31% 초과 제외 {int(out.sum()):,}행")
df["lr"] = np.log1p(df.ret)

dates = sorted(df.date.unique()); d0, d1 = dates[0], dates[-1]

# ===== Q1 시총 순위 =====
a, b = df[df.date == d0], df[df.date == d1]
s0, s1 = a.nlargest(30, "mktcap_eok"), b.nlargest(30, "mktcap_eok")
kept = set(s0.code) & set(s1.code)
new = s1[~s1.code.isin(s0.code)]
log(f"\n[Q1] 기준일 {str(d0)[:10]} → {str(d1)[:10]}")
log(f"처음 상위30 중 지금도 30위 안: {len(kept)}종목")
log("새로 들어온 종목(업종): " + ", ".join(f"{r.name_ if False else r.code}({r.upjong})" for r in new.itertuples()))
sec0 = a.groupby("upjong_no").mktcap_eok.sum().rank(ascending=False)
sec1 = b.groupby("upjong_no").mktcap_eok.sum()
sec1r = sec1.rank(ascending=False)
move = (sec0 - sec1r).dropna()
top = sec1.nlargest(15).index.intersection(move.index)
mv = move[top].sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([L(i) for i in mv.index], mv.values, color=["tab:red" if v > 0 else "tab:blue" for v in mv.values])
ax.axvline(0, color="k", lw=.8)
ax.set_title(f"Q1. Sector rank change by total mkt cap (top-15 sectors now)\nTop30 stocks kept: {len(kept)}/30  (red = moved up)")
ax.set_xlabel("rank change (start rank - end rank)")
plt.tight_layout(); plt.savefig("q1_rank.png", dpi=150); plt.close()
log("업종 순위 이동(상위15 업종): " + ", ".join(f"{L(i)} {int(v):+d}" for i, v in move[top].sort_values(ascending=False).items()))

# ===== Q2 업종 수익률 격차 =====
cnt = df.groupby("upjong_no").code.nunique()
sec_ret = df.pivot_table(index="date", columns="upjong_no", values="ret", aggfunc="mean")
keep5 = cnt[cnt >= 5].index
cum = (1 + sec_ret[keep5].fillna(0)).cumprod() - 1
final = cum.iloc[-1].sort_values()
disp = cum.std(axis=1)
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
hi, lo = final.index[-3:], final.index[:3]
for c in cum.columns:
    ax[0].plot(cum.index, cum[c] * 100, color="lightgray", lw=.6)
for c in hi: ax[0].plot(cum.index, cum[c] * 100, lw=2, label=L(c))
for c in lo: ax[0].plot(cum.index, cum[c] * 100, lw=2, ls="--", label=L(c))
ax[0].legend(fontsize=7); ax[0].set_ylabel("cumulative return (%)")
ax[0].set_title("Q2. Sector cumulative return (equal-weighted) and spread across sectors")
ax[1].plot(disp.index, disp * 100, color="k"); ax[1].set_ylabel("std across sectors (%p)")
plt.tight_layout(); plt.savefig("q2_spread.png", dpi=150); plt.close()
contrib = ((final - final.mean()) ** 2 / ((final - final.mean()) ** 2).sum()).sort_values(ascending=False)
log(f"\n[Q2] 업종 수(5종목 이상) {len(keep5)}개. 격차(표준편차) 시작 {disp.iloc[20]*100:.1f}%p → 끝 {disp.iloc[-1]*100:.1f}%p")
log("누적수익률 상위: " + ", ".join(f"{L(c)} {final[c]*100:.0f}%" for c in final.index[-3:][::-1]))
log("누적수익률 하위: " + ", ".join(f"{L(c)} {final[c]*100:.0f}%" for c in final.index[:3]))
log(f"격차 기여도 상위 5개 업종 합계: {contrib.iloc[:5].sum()*100:.0f}% (" + ", ".join(L(c) for c in contrib.index[:5]) + ")")

# ===== Q3 업종 간 상관 =====
keep10 = cnt[cnt >= 10].index
R = sec_ret[keep10]
mkt = df.groupby("date").ret.mean()
def avgcorr(x):
    c = x.corr().values
    return np.nanmean(c[np.triu_indices_from(c, 1)])
xs, ys = [], []
for i in range(60, len(R), 10):
    xs.append(R.index[i]); ys.append(avgcorr(R.iloc[i - 60:i]))
res = {}
for q in (0.90, 0.95):
    shock = mkt.abs() >= mkt.abs().quantile(q)
    res[q] = (avgcorr(R[shock.reindex(R.index).fillna(False)]), avgcorr(R[~shock.reindex(R.index).fillna(False)]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
ax[0].plot(xs, ys); ax[0].set_title("Q3. Avg pairwise sector correlation (60d rolling)")
ax[0].set_ylabel("mean correlation")
w = .35; k = list(res)
ax[1].bar(np.arange(2) - w / 2, [res[q][0] for q in k], w, label="shock days")
ax[1].bar(np.arange(2) + w / 2, [res[q][1] for q in k], w, label="normal days")
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(["top10%", "top5%"]); ax[1].set_xlabel("shock = |market ret| top x%")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("q3_corr.png", dpi=150); plt.close()
log(f"\n[Q3] 업종 수(10종목 이상) {len(keep10)}개. 60일 평균 상관 최소 {min(ys):.2f} ~ 최대 {max(ys):.2f}")
for q in k:
    log(f"충격일 기준 상위 {int((1-q)*100)}%: 충격일 상관 {res[q][0]:.2f} vs 보통일 {res[q][1]:.2f}")

# ===== Q4 거래량 급증 =====
df["cum"] = df.groupby("code").lr.transform(lambda s: s.fillna(0).cumsum())
mcum = np.log1p(mkt).fillna(0).cumsum()
ex = df.cum - df.date.map(mcum)
for kk in (5, 10, 20):
    df[f"f{kk}"] = ex.groupby(df.code).shift(-kk) - ex
avgv = df.groupby("code").volume.transform(lambda s: s.shift().rolling(20, min_periods=15).mean())
ratio = df.volume / avgv
liquid = avgv * df.price >= 1e8
base = [df.loc[liquid, f"f{kk}"].mean() * 100 for kk in (5, 10, 20)]
fig, ax = plt.subplots(figsize=(8, 5))
log("\n[Q4] 급증 = 당일 거래량 ≥ 직전 20일 평균의 T배, 직전 평균 거래대금 1억원 이상 종목만. 수익률 = 시장(전체 평균) 대비 초과 로그수익률")
log(f"기준(급증 아닌 전체 평균) 5/10/20일: " + " / ".join(f"{x:.2f}%" for x in base))
for j, T in enumerate((3, 5, 10)):
    m = liquid & (ratio >= T)
    vals = [df.loc[m, f"f{kk}"].mean() * 100 for kk in (5, 10, 20)]
    ax.bar(np.arange(3) + j * .25 - .25, vals, .25, label=f"spike ≥{T}x (n={int(m.sum()):,})")
    log(f"T={T}배: 사건 {int(m.sum()):,}건, 5/10/20일 초과수익률 " + " / ".join(f"{x:.2f}%" for x in vals))
ax.plot(np.arange(3), base, "k--o", label="all days (baseline)")
ax.set_xticks(range(3)); ax.set_xticklabels(["5d", "10d", "20d"]); ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("mean excess return after spike (%)"); ax.set_title("Q4. Returns after volume spikes (by threshold)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("q4_volume.png", dpi=150); plt.close()

# ===== 정의 메모 =====
defs = """# 정의 메모 (질문 1~4)
- 자료: all.csv (2024-03-27 ~ 2026-09-02, 거래일 591). 분석 대상은 업종·시총이 있는 종목(ETF·ETN 등 제외).
- 일별 수익률: 종가 변화율. 주식수가 20% 이상 바뀐 날은 시가총액 변화율로 대체(액면분할·병합 보정). 10일 넘게 끊긴 구간과 ±31% 초과는 제외(NaN).
- 업종: 그날 자료의 업종을 사용(재분류 종목은 날짜별로 다른 업종에 속함). 업종 수익률 = 소속 종목 단순평균.
- Q1: 시작일·마지막일의 시총 상위 30위 비교. 업종 순위 = 업종 시총 합계 순위.
- Q2: 5종목 이상 업종만. 격차 = 업종 누적수익률의 표준편차.
- Q3: 10종목 이상 업종만. 충격일 = 시장 평균수익률 절댓값 상위 10%(민감도: 5%).
- Q4: 급증 = 거래량 ≥ 직전 20일 평균 × T(3·5·10). 거래대금 1억 미만 종목 제외. 수익률 = 시장 대비 초과(로그).

# 결과
"""
open("memo.md", "w", encoding="utf-8").write(defs + "\n".join(memo))
print("\ndone: q1_rank.png q2_spread.png q3_corr.png q4_volume.png memo.md")

