# fin_analysis.py - 재무제표 질문 5개(Q5~Q9). 입력: all.csv, fin_changes.csv  (둘 다 서버 홈, GitHub에 올리지 않음)
# 실행: python3 fin_analysis.py            -> fin_memo.md, fin_q5~q9 그림(png)
# 그림의 제목·축은 영어(서버에 한글 글꼴이 없을 수 있음). 표는 fin_memo.md 에 한글로 나옴.
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL, FIN = "all.csv", "fin_changes.csv"
START = pd.Timestamp("2024-03-27")
START_FIN = START + pd.Timedelta(days=19)   # 시작 시점에 알려진 재무값 기준일(2024-04-15)
MIN_SEC = 5                                  # 업종당 최소 종목 수
FINANCIAL = ["은행", "증권", "보험", "카드", "금융", "자산운용", "창업투자"]   # Q8에서 제외하는 업종 이름 조각
LABEL = {"rev": "매출액", "op": "영업이익", "roe": "ROE", "debt": "부채비율", "bps": "BPS"}  # fin_extract.py --test 로 확인
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def rk(s): return s.rank()
def sp(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    return (d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()) if len(d) > 3 else np.nan), len(d)

# ---------------- 데이터 ----------------
df = pd.read_csv(ALL, dtype={"code": str}, usecols=["date", "code", "upjong", "price", "mktcap_eok", "shares"])
df["code"] = df.code.str.split("_").str[-1]
df["date"] = pd.to_datetime(df.date)
df = df.dropna(subset=["upjong", "price"]); df = df[df.price > 0]
df = df.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"])
sector = df.groupby("code").upjong.last()          # 업종이 바뀐 종목은 가장 최근 업종으로 고정
g = df.groupby("code")
ret = g.price.pct_change()
gap = (df.date - g.date.shift()).dt.days
shr = g.shares.pct_change().abs()
excl = (gap > 10) | (shr >= 0.2) | (ret.abs() > 0.31)
P(f"[수익률 제외] 날짜공백>10일 {int((gap>10).sum())}, 주식수 20%+변동 {int((shr>=0.2).sum())}, |r|>31% {int((ret.abs()>0.31).sum())}")
ret[excl] = np.nan; df["ret"] = ret
PR = df.pivot(index="date", columns="code", values="price")
R = df.pivot(index="date", columns="code", values="ret")
MC = df.pivot(index="date", columns="code", values="mktcap_eok")
dates = R.index; END = dates[-1]
mkt = R.mean(axis=1)                               # 시장 = 전 종목 동일가중 평균(공통질문 4와 같은 정의)
EX = R.sub(mkt, axis=0)
obs = R.notna().sum()
cum = np.expm1(np.log1p(R).sum()); cum[obs < 100] = np.nan   # 종목 누적수익률(제외된 날은 0으로 취급)
secs = sector.reindex(R.columns)
S_eq = R.T.groupby(secs).mean().T
S_cum_eq = np.expm1(np.log1p(S_eq).sum())
W = MC.shift(1).where(R.notna())
S_mc = (R * W).T.groupby(secs).sum().T / W.T.groupby(secs).sum().T
S_cum_mc = np.expm1(np.log1p(S_mc).sum())
S_cum_med = cum.groupby(secs).median()
P(f"기간 {dates[0].date()} ~ {END.date()}, 거래일 {len(dates)}, 종목 {R.shape[1]}")

fc = pd.read_csv(FIN, dtype={"code": str, "old": str, "new": str})
fc["date"] = pd.to_datetime(fc.date)
num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
fc["oldn"], fc["newn"] = num(fc.old), num(fc.new)
fc["init"] = fc.init.astype(bool)

def mask(key):
    lab = LABEL[key]
    return (fc.item == lab) if key in ("rev", "op") else fc.item.str.startswith(lab)
def snap(key, period, asof):
    d = fc[mask(key) & (fc.period == period) & fc.newn.notna() & (fc.date <= asof)]
    return d.sort_values("date").groupby("code").newn.last()
def rho(a, b, tag):
    r, n = sp(a, b); P(f"  {tag}: 스피어만 상관 {r:+.2f} (n={n})"); return r

# ================= Q5 영업이익률 vs 수익률 =================
P("\n## Q5. 영업이익률이 높은 업종이 주가 수익률도 높았는가")
rev0, op0 = snap("rev", "Y2023.12", START_FIN), snap("op", "Y2023.12", START_FIN)
rev1, op1 = snap("rev", "Y2025.12", END), snap("op", "Y2025.12", END)
t = pd.DataFrame({"rev0": rev0, "op0": op0, "rev1": rev1, "op1": op1, "cum": cum, "sec": sector}).dropna()
t = t[(t.rev0 > 0) & (t.rev1 > 0)]
t["m0"], t["m1"] = t.op0 / t.rev0, t.op1 / t.rev1
t["dm"] = t.m1 - t.m0
P(f"분석 종목 {len(t)} (시작·끝 모두 연간 확정 매출액·영업이익 있는 종목; 매출 0 이하 제외)")
sa = t.groupby("sec").apply(lambda x: pd.Series({
    "n": len(x), "m0_a": x.op0.sum() / x.rev0.sum(), "m1_a": x.op1.sum() / x.rev1.sum(),
    "m0_b": x.m0.median(), "m1_b": x.m1.median()}))
sa = sa[sa.n >= MIN_SEC]
sa["dm_a"], sa["dm_b"] = sa.m1_a - sa.m0_a, sa.m1_b - sa.m0_b
sa["ret_eq"], sa["ret_mc"], sa["ret_med"] = S_cum_eq, S_cum_mc, S_cum_med
sa = sa.dropna(); P(f"업종 {len(sa)}개 (종목 {MIN_SEC}개 이상)")
P("[정의 (a): 업종 이익률 = 업종 영업이익 합 / 매출액 합 (규모 가중)]  [(b): 종목 이익률의 중앙값]")
for rv, nm in [("ret_eq", "동일가중 누적수익률"), ("ret_mc", "시총가중 누적수익률"), ("ret_med", "종목 누적수익률 중앙값")]:
    P(f"- 수익률 = {nm}")
    rho(sa.m0_a, sa[rv], "시작 이익률(a) vs 수익률")
    rho(sa.m0_b, sa[rv], "시작 이익률(b) vs 수익률")
    rho(sa.dm_a, sa[rv], "이익률 변화(a) vs 수익률")
    rho(sa.dm_b, sa[rv], "이익률 변화(b) vs 수익률")
# 업종 안에서
wi = t.groupby("sec").apply(lambda x: sp(x.m0, x.cum)[0] if len(x) >= 8 else np.nan).dropna()
P(f"[업종 안] 종목 시작 이익률 vs 종목 수익률 상관: 업종 {len(wi)}개, 평균 {wi.mean():+.2f}, 양수인 업종 {(wi>0).mean():.0%}")
# 수준 vs 변화
z = sa[["m0_a", "dm_a", "ret_eq"]].apply(rk); z = (z - z.mean()) / z.std()
b = np.linalg.lstsq(np.c_[np.ones(len(z)), z.m0_a, z.dm_a], z.ret_eq.values, rcond=None)[0]
P(f"[수준 vs 변화] 순위 회귀 계수: 원래 이익률 {b[1]:+.2f}, 이익률 변화 {b[2]:+.2f}  (절댓값이 큰 쪽이 수익률과 더 관련)")
hi = sa.m0_a >= sa.m0_a.median(); up = sa.dm_a >= sa.dm_a.median()
for h_, u_ in [(1, 0), (0, 1), (1, 1), (0, 0)]:
    s_ = sa[(hi == bool(h_)) & (up == bool(u_))]
    P(f"  원래 이익률 {'높음' if h_ else '낮음'} / 이익률 {'상승' if u_ else '하락'}: 업종 {len(s_)}개, 평균 수익률 {s_.ret_eq.mean():+.1%}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(sa.m0_a * 100, sa.ret_eq * 100); ax[0].set_xlabel("sector operating margin at start (%) [sum OP / sum sales]")
ax[0].set_ylabel("sector cumulative return (%) [equal weight]"); ax[0].set_title("Q5: margin level vs return")
ax[1].scatter(sa.dm_a * 100, sa.ret_eq * 100); ax[1].set_xlabel("change in margin FY2023->FY2025 (%p)")
ax[1].set_ylabel("sector cumulative return (%)"); ax[1].set_title("Q5: margin change vs return")
plt.tight_layout(); plt.savefig("fin_q5_margin.png", dpi=130); plt.close()

# ================= 이벤트(실적 발표일) 만들기 =================
op = fc[mask("op")]
isQ = op.period.str.startswith("Q"); isE = op.period.str.endswith("(E)")
ev = op[isQ & ~isE & op.old.isna() & ~op.init & op.newn.notna()][["code", "date", "period", "newn"]].rename(columns={"newn": "act"})
pend = pd.to_datetime(ev.period.str[1:] + ".01", format="%Y.%m.%d") + pd.offsets.MonthEnd(0)
lag = (ev.date - pend).dt.days
P(f"\n[발표일 이벤트] 분기 영업이익이 '추정→확정'으로 처음 나타난 (종목,날짜): {len(ev)}건; 분기말~표 변경 0~100일 밖 {int(((lag<0)|(lag>100)).sum())}건 제외")
ev = ev[(lag >= 0) & (lag <= 100)].copy()
E = op[isQ & isE & op.newn.notna()].sort_values("date")
hist = {k: (x.date.values, x.newn.values) for k, x in E.groupby(["code", "period"])}
def est_at(code, per, d, h):
    z_ = hist.get((code, per + "(E)"))
    if z_ is None: return np.nan
    i = np.searchsorted(z_[0], np.datetime64(d - pd.Timedelta(days=h)), side="right") - 1
    return z_[1][i] if i >= 0 else np.nan
for h in (0, 30, 60, 90):
    ev[f"e{h}"] = [est_at(c, p, d, h) for c, p, d in zip(ev.code, ev.period, ev.date)]
rv = fc[mask("rev") & fc.period.str.startswith("Q") & ~fc.period.str.endswith("(E)") & fc.old.isna() & ~fc.init & fc.newn.notna()]
ev = ev.merge(rv[["code", "period", "date", "newn"]].rename(columns={"newn": "rev"}), on=["code", "period", "date"], how="left")
ev = ev[ev.e0.notna() & (ev.e0 != 0)].copy()
ev["sec"] = ev.code.map(sector)
ev["s_a"] = (ev.act - ev.e0) / ev.e0.abs()                         # 정의(a): 추정치 대비 %
ev["s_b"] = np.where(ev.rev > 0, (ev.act - ev.e0) / ev.rev, np.nan)  # 정의(b): 매출액 대비 (이익률 %p 오차에 해당)
P(f"추정치가 있는 이벤트 {len(ev)}건 (확정 직전 마지막 추정치를 e0 로 사용)")

# ================= Q6 반응 속도 =================
P("\n## Q6. 실적 발표 뒤 주가는 며칠 만에 반응을 끝내는가")
ci = {c: j for j, c in enumerate(R.columns)}; X = EX.values
cars, keep = [], []
for r_ in ev.itertuples():
    i = dates.searchsorted(r_.date); j = ci.get(r_.code)
    if j is None or i - 10 < 0 or i + 10 >= len(dates): continue
    w = X[i - 10:i + 11, j]
    if np.isnan(w).sum() > 5: continue
    cars.append(np.nancumsum(w)); keep.append(r_.Index)
cars = np.array(cars); e6 = ev.loc[keep].copy()
ks = np.arange(-10, 11); b1 = 9   # k=-1 위치
P(f"사용 이벤트 {len(e6)}건. CAR(k) = 표 변경일(k=0)을 기준으로 k=-10일부터 k일까지의 시장 대비 초과수익률(산술) 누적.")
def grp_a(x): return np.where(x.s_a > 0.05, "beat", np.where(x.s_a < -0.05, "miss", "inline"))
e6["g"] = grp_a(e6)
def spread_days(top, bot, tag):
    sp_ = (cars[top].mean(0) - cars[top].mean(0)[b1]) - (cars[bot].mean(0) - cars[bot].mean(0)[b1])
    fin_ = sp_[-1]
    d90 = next((int(k) for k, v in zip(ks[10:], sp_[10:]) if fin_ > 0 and v >= 0.9 * fin_), None)
    P(f"  {tag}: 상위-하위 CAR 차이(k=-1 기준) k=0 {sp_[10]:+.2%}, k=1 {sp_[11]:+.2%}, k=3 {sp_[13]:+.2%}, k=10 {fin_:+.2%}, 차이의 90%에 도달한 날 k={d90}")
    return sp_
P("[정의 (a): 서프라이즈 = (확정-추정)/|추정|, +5% 초과 beat, -5% 미만 miss]")
for gname in ["beat", "inline", "miss"]:
    m_ = (e6.g == gname).values
    P(f"  {gname}: n={m_.sum()}, CAR(-1)={cars[m_].mean(0)[b1]:+.2%}, CAR(0)={cars[m_].mean(0)[10]:+.2%}, CAR(10)={cars[m_].mean(0)[-1]:+.2%}")
sa_ = spread_days((e6.g == "beat").values, (e6.g == "miss").values, "(a) beat vs miss")
sb_ = spread_days((e6.s_a > 0).values, (e6.s_a < 0).values, "(b1) 부호만: 추정 상회 vs 하회")
q = e6.s_b.dropna(); hi_, lo_ = q.quantile(.8), q.quantile(.2)
sc_ = spread_days((e6.s_b >= hi_).values, (e6.s_b <= lo_).values, "(b2) 매출액 대비 오차 상위20% vs 하위20%")
fig, ax = plt.subplots(figsize=(8, 4.5))
for gname in ["beat", "inline", "miss"]:
    ax.plot(ks, cars[(e6.g == gname).values].mean(0) * 100, label=gname)
ax.axvline(0, color="gray", ls=":"); ax.set_xlabel("trading days from table-change date"); ax.set_ylabel("cumulative excess return (%)")
ax.set_title("Q6: reaction around earnings update"); ax.legend(); plt.tight_layout(); plt.savefig("fin_q6_car.png", dpi=130); plt.close()

# ================= Q7 추정 오차 =================
P("\n## Q7. 추정치는 어느 업종에서 가장 많이 빗나갔는가")
ev["abs_a"], ev["abs_b"] = ev.s_a.abs(), ev.s_b.abs()
P(f"전체: 추정치보다 실제가 낮았던 비율 {(ev.act<ev.e0).mean():.1%}, 오차(a) 중앙값 {ev.s_a.median():+.1%}, 절대오차(a) 중앙값 {ev.abs_a.median():.1%}")
sq = ev.groupby("sec").agg(n=("s_a", "size"), abs_a=("abs_a", "median"), bias_a=("s_a", "median"),
                           below=("act", lambda x: np.nan), abs_b=("abs_b", "median"), bias_b=("s_b", "median"))
sq["below"] = ev.assign(b=(ev.act < ev.e0)).groupby("sec").b.mean()
sq = sq[sq.n >= 30]; P(f"업종 {len(sq)}개 (이벤트 30건 이상)")
P("[정의 (a) 추정치 대비 % 오차 중앙값 상위 5 업종]")
for s_, r_ in sq.sort_values("abs_a", ascending=False).head(5).iterrows(): P(f"  {s_}: 절대오차 {r_.abs_a:.0%}, 방향(중앙값) {r_.bias_a:+.0%}, 실제<추정 비율 {r_.below:.0%}, n={int(r_.n)}")
P("[정의 (b) 매출액 대비 오차(이익률 %p에 해당) 중앙값 상위 5 업종]")
for s_, r_ in sq.sort_values("abs_b", ascending=False).head(5).iterrows(): P(f"  {s_}: 절대오차 {r_.abs_b:.1%}p, 방향 {r_.bias_b:+.1%}p, n={int(r_.n)}")
r_, n_ = sp(sq.abs_a, sq.abs_b); P(f"(a)와 (b) 업종 순위의 스피어만 상관: {r_:+.2f}  (낮을수록 정의에 따라 결론이 바뀜)")
top_a, top_b = set(sq.abs_a.nlargest(10).index), set(sq.abs_b.nlargest(10).index)
P(f"상위 10 업종 중 (a)(b) 공통 {len(top_a & top_b)}개")
bias_sec = sq[(sq.below >= 0.6) | (sq.below <= 0.4)]
P(f"방향 쏠림: 업종 {len(sq)}개 중 '실제<추정' 비율이 60% 이상 또는 40% 이하인 업종 {len(bias_sec)}개")
P("[추정치 수정 경로] h일 전 추정치로 본 오차 (실제-추정)/|추정| 의 중앙값")
full = ev.dropna(subset=["e30", "e60", "e90"])
for h in (90, 60, 30, 0):
    P(f"  발표 {h}일 전 추정: 중앙값 {((full.act-full[f'e{h}'])/full[f'e{h}'].abs()).replace([np.inf,-np.inf],np.nan).median():+.1%}, 절대 중앙값 {((full.act-full[f'e{h}']).abs()/full[f'e{h}'].abs()).replace([np.inf,-np.inf],np.nan).median():.1%}  (n={len(full)})")
P(f"  90일 전→발표 직전 추정치 수정폭 중앙값 {((full.e0-full.e90)/full.e90.abs()).replace([np.inf,-np.inf],np.nan).median():+.1%}")
top = sq.sort_values("abs_a", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(range(len(top)), top.abs_a * 100); ax.set_yticks(range(len(top))); ax.set_yticklabels([f"sector{i}" for i in range(len(top))])
ax.invert_yaxis(); ax.set_xlabel("median |actual - estimate| / |estimate| (%)"); ax.set_title("Q7: top 15 sectors by estimate error (names in fin_memo.md)")
plt.tight_layout(); plt.savefig("fin_q7_error.png", dpi=130); plt.close()
P("그림 Q7 막대 순서(위→아래): " + ", ".join(top.index))

# ================= Q8 부채비율 vs 변동성 =================
P("\n## Q8. 부채비율이 높은 업종이 더 크게 흔들렸는가")
d0 = snap("debt", "Y2023.12", START_FIN).rename("debt")
sd = pd.DataFrame({"debt": d0, "sec": sector}).dropna()
sd = sd[sd.debt >= 0]
vol_st = (R.std() * np.sqrt(252)); sd["vol"] = vol_st.reindex(sd.index)
fin_mask = sd.sec.apply(lambda s: any(k in s for k in FINANCIAL))
P(f"종목 {len(sd)}, 금융 업종(이름에 {FINANCIAL} 포함) 종목 {int(fin_mask.sum())}")
cr = mkt <= mkt.quantile(0.10)
P(f"급락일 = 시장 평균 일수익률 하위 10% ({int(cr.sum())}일, 기준 {mkt.quantile(.1):+.2%})")
def q8(sd_, tag):
    a = sd_.groupby("sec").agg(n=("debt", "size"), debt=("debt", "median"), vol_b=("vol", "mean")); a = a[a.n >= MIN_SEC]
    a["vol_a"] = S_eq[a.index].std() * np.sqrt(252)
    a["crash"] = S_eq.loc[cr, a.index].mean(); a["normal"] = S_eq.loc[~cr, a.index].mean()
    P(f"[{tag}] 업종 {len(a)}개")
    rho(a.debt, a.vol_a, "부채비율(중앙값) vs 업종수익률 변동성(a)")
    rho(a.debt, a.vol_b, "부채비율(중앙값) vs 종목 변동성 평균(b)")
    rho(a.debt, a.crash, "부채비율 vs 급락일 평균 수익률 (음수면 높을수록 더 떨어짐)")
    a["tc"] = pd.qcut(a.debt, 3, labels=["low", "mid", "high"])
    P("  부채비율 3분위별: " + " | ".join(f"{k} 변동성 {v.vol_a:.0%} 급락일 {v.crash:+.2%} 보통일 {v.normal:+.3%}" for k, v in a.groupby("tc").mean(numeric_only=True).iterrows()))
    return a
a_all = q8(sd, "금융 포함"); a_nf = q8(sd[~fin_mask], "금융 제외")
fig, ax = plt.subplots(figsize=(7, 4.5)); ax.scatter(a_nf.debt, a_nf.vol_a * 100)
ax.set_xlabel("sector median debt ratio FY2023 (%)"); ax.set_ylabel("annualised volatility of sector return (%)"); ax.set_title("Q8: debt ratio vs volatility (financials excluded)")
plt.tight_layout(); plt.savefig("fin_q8_debt.png", dpi=130); plt.close()

# ================= Q9 PBR < 1 =================
P("\n## Q9. PBR이 1 아래인 종목은 2년 반 뒤 어떻게 됐는가")
bps0, roe0 = snap("bps", "Y2023.12", START_FIN), snap("roe", "Y2023.12", START_FIN)
bps1 = snap("bps", "Y2025.12", END)
p0 = PR.loc[START] if START in PR.index else PR.iloc[0]
last_date = PR.apply(lambda c: c.last_valid_index()); last_p = PR.apply(lambda c: c.dropna().iloc[-1] if c.notna().any() else np.nan)
u = pd.DataFrame({"p0": p0, "bps0": bps0, "roe0": roe0, "bps1": bps1, "p1": last_p, "cum": cum, "last": last_date, "sec": sector})
u = u.dropna(subset=["p0", "cum", "sec"])
P(f"시작일 가격이 있는 종목 {len(u)}; BPS 없음 {int(u.bps0.isna().sum())}, BPS≤0(자본잠식) {int((u.bps0<=0).sum())} 는 PBR 계산에서 제외")
u = u[u.bps0 > 0].copy(); u["pbr0"] = u.p0 / u.bps0; u["pbr1"] = u.p1 / u.bps1.where(u.bps1 > 0)
u["gone"] = u["last"] < END - pd.Timedelta(days=20)
u["pref"] = ~u.index.str.endswith("0")
def summ(x, tag):
    P(f"  {tag}: n={len(x)}, 평균 {x.cum.mean():+.1%}, 중앙값 {x.cum.median():+.1%}, 오른 비율 {(x.cum>0).mean():.0%}, 마지막 날 전에 사라진 종목 {x.gone.mean():.0%}")
def run(thr, excl_pref, tag):
    x = u[~u.pref] if excl_pref else u
    lo = x[(x.pbr0 < thr)]; hi = x[x.pbr0 >= thr]
    P(f"[{tag}] 저PBR(<{thr}) {len(lo)}종목 / 이상 {len(hi)}종목")
    summ(lo, "저PBR"); summ(hi, "그 외")
    summ(lo[lo.roe0 > 0], "저PBR & ROE>0"); summ(lo[lo.roe0 <= 0], "저PBR & ROE≤0")
    return lo
lo = run(1.0, False, "기본: PBR<1, 우선주 포함")
run(1.0, True, "(b1) 우선주 제외"); run(0.7, False, "(b2) 기준 0.7")
c = lo.groupby("sec").size().rename("low"); tot = u.groupby("sec").size().rename("all")
cc = pd.concat([c, tot], axis=1).fillna(0); cc = cc[cc["all"] >= 10]; cc["share"] = cc.low / cc["all"]
P("[저PBR이 몰린 업종: 업종 내 저PBR 비율 상위 8 (종목 10개 이상)]")
for s_, r_ in cc.sort_values("share", ascending=False).head(8).iterrows(): P(f"  {s_}: {r_.share:.0%} ({int(r_.low)}/{int(r_['all'])})")
lo2 = lo[lo.pbr1.notna()].copy(); lo2["resolved"] = lo2.pbr1 >= 1
P("[해소 여부: 끝 시점 PBR = 마지막 가격 / 끝 시점 최신 연간 BPS(2025.12)]")
for k, x in lo2.groupby("resolved"): summ(x, "해소(PBR≥1)" if k else "그대로(PBR<1)")
P("  주의: 해소는 주가가 오르면 기계적으로 생기므로, 해소 종목의 높은 수익률은 원인이 아니라 결과일 수 있다.")
fig, ax = plt.subplots(figsize=(7, 4.5))
for lab, x in [("PBR<1 & ROE>0", lo[lo.roe0 > 0]), ("PBR<1 & ROE<=0", lo[lo.roe0 <= 0]), ("PBR>=1", u[u.pbr0 >= 1])]:
    ax.hist(x.cum.clip(-1, 3) * 100, bins=40, alpha=.5, label=lab, density=True)
ax.set_xlabel("cumulative return (%, clipped at 300)"); ax.set_title("Q9: return by start-PBR and ROE"); ax.legend()
plt.tight_layout(); plt.savefig("fin_q9_pbr.png", dpi=130); plt.close()

open("fin_memo.md", "w", encoding="utf-8").write("\n".join(out))
print("\n끝: fin_memo.md, fin_q5_margin.png, fin_q6_car.png, fin_q7_error.png, fin_q8_debt.png, fin_q9_pbr.png")
