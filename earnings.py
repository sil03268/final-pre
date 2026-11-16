# earnings.py - 종목별 "기업실적분석" 표가 바뀐 날 찾기
# 테스트(한 종목):   python3 earnings.py --test 005930
# 전체 실행:         nohup python3 earnings.py > earn.log 2>&1 &
# 결과: earnings_changes.csv (date, code, item, period, estimate, old, new)
import os, re, sys, html
from multiprocessing import Pool
import pandas as pd

ROOT = "/mnt/data/stock"
ITEMS = ["매출액", "영업이익", "당기순이익", "영업이익률"]


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_table(path):
    """HTML 한 개 -> {(항목, 기간): 값} 딕셔너리. 표를 못 찾으면 None."""
    h = open(path, encoding="utf-8", errors="ignore").read()
    i = h.find("th_cop_anal")
    if i < 0:
        return None
    start = h.rfind("<table", 0, i)
    end = h.find("</table>", i)
    if start < 0 or end < 0:
        return None
    tb = h[start:end]
    # 기간 제목 (2023.12, 2026.12(E) ...)
    periods = []
    for th in re.findall(r"<th[^>]*>(.*?)</th>", tb, flags=re.S):
        t = clean(th).replace(" ", "")
        if re.fullmatch(r"\d{4}\.\d{2}(\(E\))?", t):
            periods.append(t)
    periods = periods[:10]
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=re.S):
        m = re.search(r"<th[^>]*>(.*?)</th>", tr, flags=re.S)
        if not m:
            continue
        label = clean(m.group(1))
        if label not in ITEMS:
            continue
        vals = [clean(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        for p, v in zip(periods, vals):
            out[(label, p)] = v
    return out or None


def one_code(args):
    code, items = args          # items = [(date, path), ...] 날짜순
    prev, rows, nodata = None, [], 0
    for d, p in items:
        try:
            cur = parse_table(p)
        except Exception:
            cur = None
        if cur is None:
            nodata += 1
            continue
        if prev is not None:
            for k in set(prev) | set(cur):
                a, b = prev.get(k), cur.get(k)
                if a != b:
                    rows.append((d, code, k[0], k[1], "(E)" in k[1], a, b))
        prev = cur
    return rows, nodata, len(items)


def collect(only=None):
    by = {}
    for d in sorted(os.listdir(ROOT)):
        dp = os.path.join(ROOT, d)
        if not os.path.isdir(dp):
            continue
        for f in os.listdir(dp):
            if f.endswith(".html"):
                c = f[:-5].split("_")[-1]
                if only and c not in only:
                    continue
                by.setdefault(c, []).append((d, os.path.join(dp, f)))
    return by


COLS = ["date", "code", "item", "period", "estimate", "old", "new"]

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        code = sys.argv[2]
        by = collect({code})
        print("파일 수:", len(by.get(code, [])))
        first = by[code][-1][1]
        print("마지막 날 표 파싱 예시:")
        t = parse_table(first)
        for k in list(t)[:12] if t else []:
            print(" ", k, t[k])
        rows, nodata, n = one_code((code, by[code]))
        df = pd.DataFrame(rows, columns=COLS)
        print(f"표 없는 날 {nodata}/{n}, 변경 기록 {len(df)}건")
        print("변경된 날짜 수:", df.date.nunique())
        print(df.head(30).to_string())
    else:
        by = collect()
        print("종목 수:", len(by), flush=True)
        allrows, nod, tot = [], 0, 0
        with Pool(8) as pool:
            for rows, nodata, n in pool.imap_unordered(one_code, by.items(), chunksize=20):
                allrows += rows; nod += nodata; tot += n
        df = pd.DataFrame(allrows, columns=COLS).sort_values(["code", "date"])
        df.to_csv("earnings_changes.csv", index=False, encoding="utf-8-sig")
        print(f"전체 파일 {tot:,} | 표 없는 파일 {nod:,} | 변경 기록 {len(df):,}건")
        print("변경이 있었던 종목:", df.code.nunique())
        print("확정치 변경(추정 아님):", int((~df.estimate).sum()), "건")
        print("done: earnings_changes.csv")
