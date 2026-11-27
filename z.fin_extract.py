# fin_extract.py - 종목 페이지 "기업실적분석" 표 전체(모든 항목)를 읽어, 표가 바뀐 날만 기록
# earnings.py와 다른 점: (1) 모든 항목(ROE·부채비율·EPS·BPS·배당 등) (2) 연간(Y)/분기(Q) 열을 구분
#   -> earnings.py는 12월 연간과 12월 분기가 같은 이름이라 서로 덮어썼음
# 테스트(한 종목):  python3 fin_extract.py --test 005930
# 전체 실행:        nohup python3 fin_extract.py > fin_extract.log 2>&1 &
# 결과: fin_changes.csv (date, code, item, period, old, new, init)
#   period 예: Y2023.12 (연간 확정), Y2026.12(E) (연간 추정), Q2026.03 (분기 확정)
#   init=True : 그 종목 표가 처음 저장된 날의 값(변경 아님)
import os, re, sys, html
from multiprocessing import Pool
import pandas as pd

ROOT = "/mnt/data/stock"
PER = re.compile(r"\d{4}\.\d{2}(\(E\))?")

def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()

def parse_table(path):
    h = open(path, encoding="utf-8", errors="ignore").read()
    i = h.find("th_cop_anal")
    if i < 0:
        return None
    s = h.rfind("<table", 0, i); e = h.find("</table>", i)
    if s < 0 or e < 0:
        return None
    tb = h[s:e]
    periods = []
    for th in re.findall(r"<th[^>]*>(.*?)</th>", tb, flags=re.S):
        t = clean(th).replace(" ", "")
        if PER.fullmatch(t):
            periods.append(t)
    if len(periods) != 10:          # 연간 4 + 분기 6 이 아니면 믿지 않는다
        return ("BADLEN", len(periods))
    keys = [("Y" if k < 4 else "Q") + p for k, p in enumerate(periods)]
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=re.S):
        m = re.search(r"<th[^>]*>(.*?)</th>", tr, flags=re.S)
        if not m:
            continue
        label = clean(m.group(1))
        if not label or PER.fullmatch(label.replace(" ", "")):
            continue
        vals = [clean(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if len(vals) < 10:
            continue
        for k, v in zip(keys, vals):
            out[(label, k)] = v
    return out or None

def one_code(args):
    code, items = args
    prev, rows, nodata, bad = None, [], 0, 0
    for d, p in items:
        try:
            cur = parse_table(p)
        except Exception:
            cur = None
        if cur is None:
            nodata += 1; continue
        if isinstance(cur, tuple):
            bad += 1; continue
        if prev is None:
            for k, v in cur.items():
                rows.append((d, code, k[0], k[1], None, v, True))
        else:
            for k in set(prev) | set(cur):
                a, b = prev.get(k), cur.get(k)
                if a != b:
                    rows.append((d, code, k[0], k[1], a, b, False))
        prev = cur
    return rows, nodata, bad, len(items)

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

COLS = ["date", "code", "item", "period", "old", "new", "init"]

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        code = sys.argv[2]; by = collect({code})
        t = parse_table(by[code][-1][1])
        print("마지막 날 항목 이름(이 이름을 보고 fin_analysis.py 의 LABEL 을 확인):")
        if isinstance(t, dict):
            for lab in dict.fromkeys(k[0] for k in t):
                print("  ", repr(lab), "|", [t[(lab, p)] for (l, p) in t if l == lab][:10])
        else:
            print("  표 파싱 실패:", t)
        rows, nodata, bad, n = one_code((code, by[code]))
        df = pd.DataFrame(rows, columns=COLS)
        print(f"파일 {n} | 표 없음 {nodata} | 열 수 이상 {bad} | 기록 {len(df)}건")
    else:
        by = collect(); print("종목 수:", len(by), flush=True)
        allrows, nod, bad, tot = [], 0, 0, 0
        with Pool(8) as pool:
            for rows, a, b, n in pool.imap_unordered(one_code, by.items(), chunksize=20):
                allrows += rows; nod += a; bad += b; tot += n
        df = pd.DataFrame(allrows, columns=COLS).sort_values(["code", "date"])
        df.to_csv("fin_changes.csv", index=False, encoding="utf-8-sig")
        print(f"전체 파일 {tot:,} | 표 없음 {nod:,} | 열 수 이상 {bad:,} | 기록 {len(df):,}건")
        print("항목 이름 목록:", sorted(df.item.unique()))
        print("done: fin_changes.csv")
