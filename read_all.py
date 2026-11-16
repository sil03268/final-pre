import os, glob, sys
from multiprocessing import Pool
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~"))
from parse_archive import parse_file, ROOT

MY = "방송과엔터테인먼트"   # 내 업종명 (다르면 수정)


def work(p):
    d = os.path.basename(os.path.dirname(p))
    c = os.path.basename(p)[:-5]
    try:
        r = parse_file(p)
        r.update(date=d, code=c)
        return r
    except Exception as e:
        return {"date": d, "code": c, "err": repr(e)[:80]}


if __name__ == "__main__":
    files = sorted(glob.glob(f"{ROOT}/*/*.html"))
    with Pool(8) as p:
        rows = p.map(work, files, chunksize=200)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    if "err" in df:
        bad = df[df["err"].notna()]
    else:
        bad = df.iloc[0:0]
    ok = df.drop(columns=["err"], errors="ignore").dropna(subset=["price"])
    bad.to_csv("failed.csv", index=False)
    ok.to_csv("all.csv", index=False)

    # 증거 요약
    last = ok.date.max()
    life = ok.groupby("code").date.agg(["min", "max"])
    print("파일", len(files), "| 성공 행", len(ok), "| 실패", len(bad))
    print("기간", ok.date.min().date(), "~", last.date(), "| 거래일", ok.date.nunique())
    print("종목 수", ok.code.nunique(),
          "| 상장폐지 추정", (life["max"] < last).sum(),
          "| 신규상장", (life["min"] > ok.date.min()).sum())
    print("업종 재분류 종목", (ok.groupby("code").upjong_no.nunique() > 1).sum())
    sp = ok.sort_values(["code", "date"]).copy()
    sp["sh"] = sp.groupby("code").shares.pct_change()
    print("액면분할/병합 의심", (sp.sh.abs() > 0.5).sum())

    # 그림 0: 날짜별 읽은 종목 수
    cnt = ok.groupby("date").code.nunique()
    fail = bad.groupby("date").size().reindex(cnt.index, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cnt.index, cnt.values, label="parsed")
    ax.plot(fail.index, fail.values, label="failed")
    ax.set_title("Stocks read per trading day")
    ax.legend()
    fig.savefig("fig0_coverage.png", dpi=150)

    # Q1: 업종 시총 순위
    s = ok.groupby(["date", "upjong"]).mktcap_eok.sum().unstack()
    rk = s.rank(axis=1, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    rk.plot(ax=ax, color="lightgray", legend=False)
    if MY in rk:
        rk[MY].plot(ax=ax, color="red", lw=2)
    ax.invert_yaxis()
    ax.set_title("Industry market-cap rank")
    fig.savefig("fig1_rank.png", dpi=150)
    

