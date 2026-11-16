#!/usr/bin/env python3
"""교수 아카이브(/mnt/data/stock/날짜/종목코드.html) -> 날짜별 CSV 파서 (서버에서 실행)

사용법
  python3 parse_archive.py --test 2026-06-23 005930        # 파일 1개 파싱 결과 출력(먼저 이것부터)
  python3 parse_archive.py --dates 2026-06-23              # 날짜 1개 전체 파싱(시간 측정용)
  nohup python3 parse_archive.py --step 5 > parse.log 2>&1 &   # 5거래일마다 표본
  nohup python3 parse_archive.py > parse.log 2>&1 &            # 전체(오래 걸림)
  python3 parse_archive.py --merge                          # ~/parsed/*.csv -> ~/all_stocks.csv, ~/parse_report.csv
날짜별 결과가 ~/parsed/날짜.csv로 저장되므로 끊겨도 이어서 실행하면 건너뛴다.
"""
import argparse, csv, os, re, sys, glob
from multiprocessing import Pool

ROOT = "/mnt/data/stock"
OUT = os.path.expanduser("~/parsed")
FIELDS = ["date", "code", "name", "upjong_no", "upjong", "price", "volume",
          "mktcap_eok", "market", "rank", "shares"]

def num(s):
    s = (s or "").replace(",", "").strip()
    return int(s) if s.isdigit() else None

def first(rx, text, flags=re.S):
    m = re.search(rx, text, flags)
    return m.group(1) if m else None

def parse_html(text):
    r = {}
    t = first(r"<title>\s*(.*?)\s*:\s*Npay", text)
    r["name"] = t.strip() if t else None
    m = re.search(r"type=upjong&(?:amp;)?no=(\d+)\">([^<]+)</a>", text)
    r["upjong_no"], r["upjong"] = (m.group(1), m.group(2).strip()) if m else (None, None)
    # 현재가: 상단 요약 dl 우선, 없으면 첫 no_today의 blind
    p = first(r"<dd>현재가\s*([\d,]+)", text)
    if p is None:
        blk = first(r'<p class="no_today">(.*?)</p>', text)
        p = first(r'<span class="blind">([\d,]+)</span>', blk) if blk else None
    r["price"] = num(p)
    r["volume"] = num(first(r"<dd>거래량\s*([\d,]+)</dd>", text))
    r["mktcap_eok"] = num(first(r"시가총액\(억\)</span></th>.*?<td>\s*([\d,]+)\s*</td>", text))
    m = re.search(r"(코스피|코스닥)\s*<em>(\d+)</em>위", text)
    r["market"], r["rank"] = (m.group(1), int(m.group(2))) if m else (None, None)
    r["shares"] = num(first(r"상장주식수</th>\s*<td><em>([\d,]+)</em>", text))
    return r

def parse_file(path):
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    return parse_html(text)

def do_date(date):
    out = os.path.join(OUT, date + ".csv")
    if os.path.exists(out):
        return date, -1, 0
    d = os.path.join(ROOT, date)
    ok = fail = 0
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".html"):
                continue
            try:
                r = parse_file(os.path.join(d, fn))
            except Exception:
                fail += 1
                continue
            r["date"], r["code"] = date, fn[:-5]
            if r["price"] is None:
                fail += 1          # 가격을 못 읽은 파일 = 읽지 못한 부분
            w.writerow(r)
            ok += 1
    os.replace(tmp, out)
    return date, ok, fail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs=2, metavar=("DATE", "CODE"))
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--procs", type=int, default=min(6, os.cpu_count() or 2))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.test:
        print(parse_file(os.path.join(ROOT, a.test[0], a.test[1] + ".html")))
        return
    if a.merge:
        import pandas as pd
        fs = sorted(glob.glob(os.path.join(OUT, "*.csv")))
        df = pd.concat((pd.read_csv(f, dtype={"code": str}) for f in fs), ignore_index=True)
        df.to_csv(os.path.expanduser("~/all_stocks.csv"), index=False, encoding="utf-8-sig")
        rep = df.groupby("date").agg(files=("code", "size"),
                                     price_ok=("price", lambda s: s.notna().sum()),
                                     upjong_ok=("upjong", lambda s: s.notna().sum()),
                                     cap_ok=("mktcap_eok", lambda s: s.notna().sum()))
        rep.to_csv(os.path.expanduser("~/parse_report.csv"), encoding="utf-8-sig")
        print("rows", len(df), "dates", df.date.nunique(), "stocks", df.code.nunique())
        print(rep.tail())
        return
    os.makedirs(OUT, exist_ok=True)
    dates = a.dates or sorted(x for x in os.listdir(ROOT) if re.fullmatch(r"20\d\d-\d\d-\d\d", x))
    if not a.dates:
        dates = dates[::a.step]
    with Pool(a.procs) as p:
        for date, ok, fail in p.imap_unordered(do_date, dates):
            print(date, "skip" if ok < 0 else f"ok={ok} fail={fail}", flush=True)

if __name__ == "__main__":
    main()
