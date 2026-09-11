"""
KW-MOON-0001 — 真实 LOLA 高程获取

按需下载 NASA PDS LOLA GDR 柱面栅格。**不提交进版本库**
（沿用仓库 `data/` 排除惯例：AGENTS.md §3）。

数据源（2026-09-11 实测可直连）：
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/
  ldem_16.img  5760×2880  int16 小端  比例因子 0.5 m  16 ppd  33,177,600 字节
  行首为 +90° 北纬，列首为 0° 经度，经度回绕。

用法：
  python verification/scripts/kw_moon_0001_fetch_lola.py            # 下载到 data/lola/
  python verification/scripts/kw_moon_0001_fetch_lola.py --check     # 只校验已有文件
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

BASE = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        "lrolol_1xxx/data/lola_gdr/cylindrical/img/")

PRODUCTS = {
    "ldem_16.img": {"bytes": 33177600, "ncols": 5760, "nrows": 2880, "ppd": 16},
    "ldem_4.img": {"bytes": 4147200, "ncols": 1440, "nrows": 720, "ppd": 4},
}


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="ldem_16.img", choices=sorted(PRODUCTS))
    ap.add_argument("--dest", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--proxy", default="", help="如 http://127.0.0.1:7897")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dest_dir = args.dest or os.path.join(root, "data", "lola")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, args.product)
    meta = PRODUCTS[args.product]

    if args.check or os.path.exists(path):
        if not os.path.exists(path):
            print(f"缺失：{path}")
            return 1
        size = os.path.getsize(path)
        ok = size == meta["bytes"]
        print(f"{path}\n  bytes = {size} (期望 {meta['bytes']}) -> {'OK' if ok else '尺寸不符'}")
        print(f"  sha256 = {_sha256(path)}")
        print(f"  分辨率 = {meta['ppd']} ppd, {meta['ncols']}×{meta['nrows']}")
        if not ok:
            print("  提示：文件不完整，请删除后重下。")
        return 0 if ok else 1

    url = BASE + args.product
    print(f"下载 {url}")
    print(f"  -> {path}  ({meta['bytes'] / 1e6:.1f} MB)")
    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
    urllib.request.urlretrieve(url, path)
    size = os.path.getsize(path)
    ok = size == meta["bytes"]
    print(f"  完成：{size} 字节 -> {'OK' if ok else '尺寸不符'}")
    print(f"  sha256 = {_sha256(path)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
