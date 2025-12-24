import os
import re
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


SN_RE = re.compile(r"^(\d{12})(?:_|$)")


def parse_sn12(filename: str) -> Optional[str]:
    """
    Expected: 12-digit SN at start of filename, e.g.
      123456789012_17363747.jpg
      123456789012.jpg

    If your files are like "SN_17363747.jpg" where SN is a placeholder,
    this will (correctly) NOT parse. You need the real 12 digits.
    """
    m = SN_RE.match(filename)
    return m.group(1) if m else None


def is_date_folder(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def dt_from_epoch(epoch: float) -> datetime:
    # local time
    return datetime.fromtimestamp(epoch)


def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def list_existing_date_dirs(root: Path) -> Dict[str, Path]:
    out = {}
    if not root.exists():
        return out
    for entry in root.iterdir():
        if entry.is_dir() and is_date_folder(entry.name):
            out[entry.name] = entry
    return out


def iter_files_recursive(folder: Path) -> Iterable[Path]:
    stack = [folder]
    while stack:
        cur = stack.pop()
        try:
            for entry in cur.iterdir():
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    yield entry
        except (FileNotFoundError, PermissionError):
            continue


def path_has_folder(path: Path, folder_name_lower: str) -> bool:
    return any(part.lower() == folder_name_lower for part in path.parts)


@dataclass(frozen=True)
class ScanEvent:
    sn12: str
    kind: str  # "ok" or "defect"
    minute_ts: datetime
    file_ts: datetime
    path: Path
    date_folder: str


def scan_date_dir(
    date_dir: Path,
    date_folder_name: str,
    ext_lower: str,
    ok_folder: str = "ok",
    error_folder: str = "error",
    time_source: str = "mtime",  # "mtime" or "ctime"
) -> List[ScanEvent]:
    events: List[ScanEvent] = []

    # date_dir contains shift folders; we scan everything under it,
    # but ignore anything inside any "error" folder subtree.
    for f in iter_files_recursive(date_dir):
        if f.suffix.lower() != ext_lower:
            continue
        if path_has_folder(f, error_folder):
            continue

        sn = parse_sn12(f.name)
        if not sn:
            continue

        try:
            ts_epoch = f.stat().st_mtime if time_source == "mtime" else f.stat().st_ctime
        except OSError:
            continue

        ts = dt_from_epoch(ts_epoch)
        minute_ts = floor_to_minute(ts)

        kind = "ok" if path_has_folder(f, ok_folder) else "defect"

        # If it’s under OK but also under error, we already excluded error.
        events.append(
            ScanEvent(
                sn12=sn,
                kind=kind,
                minute_ts=minute_ts,
                file_ts=ts,
                path=f,
                date_folder=date_folder_name,
            )
        )

    return events


def compute_metrics(
    root: Path,
    target_date: str,
    lookback_days: int,
    ext: str,
    time_source: str,
) -> dict:
    date_dirs = list_existing_date_dirs(root)
    if target_date not in date_dirs:
        raise FileNotFoundError(f"Target date folder not found: {root / target_date}")

    # Build lookback list: target_date - lookback_days .. target_date
    td = datetime.strptime(target_date, "%Y%m%d")
    wanted_dates = [(td - timedelta(days=d)).strftime("%Y%m%d") for d in range(lookback_days, -1, -1)]
    existing_wanted = [d for d in wanted_dates if d in date_dirs]

    # Scan all required date dirs
    all_events: List[ScanEvent] = []
    for d in existing_wanted:
        all_events.extend(
            scan_date_dir(
                date_dirs[d],
                d,
                ext_lower=ext.lower(),
                time_source=time_source,
            )
        )

    # Partition:
    # - OK events on target day (completions)
    ok_target = [e for e in all_events if e.kind == "ok" and e.date_folder == target_date]
    defect_target = [e for e in all_events if e.kind == "defect" and e.date_folder == target_date]
    defect_window = [e for e in all_events if e.kind == "defect"]  # across lookback window

    ok_sns: Set[str] = set(e.sn12 for e in ok_target)
    defect_sns_today: Set[str] = set(e.sn12 for e in defect_target)

    # Same-day repair success rate: among SNs that showed defects today, how many ended OK today?
    success_today_sns = ok_sns.intersection(defect_sns_today)
    success_rate = (len(success_today_sns) / len(defect_sns_today)) if defect_sns_today else None

    # For completions on target day, pick the latest OK timestamp per SN on that day
    ok_latest_ts: Dict[str, datetime] = {}
    for e in ok_target:
        prev = ok_latest_ts.get(e.sn12)
        if prev is None or e.file_ts > prev:
            ok_latest_ts[e.sn12] = e.file_ts

    # Build defect rounds per SN across the lookback window (minute-level uniqueness)
    defect_rounds_by_sn: Dict[str, Set[datetime]] = defaultdict(set)
    defect_file_count_by_sn: Dict[str, int] = defaultdict(int)
    for e in defect_window:
        defect_rounds_by_sn[e.sn12].add(e.minute_ts)
        defect_file_count_by_sn[e.sn12] += 1

    # Compute tries-before-success for each SN completed on target day:
    # tries = number of unique defect "round minutes" that happened before the completion OK time.
    tries_by_sn: Dict[str, int] = {}
    for sn, ok_ts in ok_latest_ts.items():
        rounds = defect_rounds_by_sn.get(sn, set())
        tries = sum(1 for r in rounds if r <= ok_ts)  # <= just in case same-minute ordering
        tries_by_sn[sn] = tries

    # Distribution
    dist = Counter(tries_by_sn.values())

    completed_repairs = len(ok_latest_ts)  # unique SN in OK that day
    defect_rounds_today = len({e.minute_ts for e in defect_target})  # workload rounds that day (unique minutes)
    defect_files_today = len(defect_target)  # raw file count (often > rounds due to multi-folder duplication)

    # Averages (for completions)
    if completed_repairs:
        avg_tries_all = sum(tries_by_sn.values()) / completed_repairs
        # Many teams prefer excluding "0 prior defects found" because it usually indicates missing history or a naming issue.
        nonzero = [v for v in tries_by_sn.values() if v > 0]
        avg_tries_nonzero = (sum(nonzero) / len(nonzero)) if nonzero else None
        zero_prior = dist.get(0, 0)
    else:
        avg_tries_all = None
        avg_tries_nonzero = None
        zero_prior = 0

    # Helpful “worst offenders”
    worst = sorted(tries_by_sn.items(), key=lambda kv: kv[1], reverse=True)[:15]

    return {
        "target_date": target_date,
        "lookback_days": lookback_days,
        "dates_scanned": existing_wanted,
        "completed_repairs_unique_sn": completed_repairs,
        "defect_unique_sn_today": len(defect_sns_today),
        "success_today_unique_sn": len(success_today_sns),
        "success_rate_same_day": success_rate,
        "defect_rounds_today_unique_minutes": defect_rounds_today,
        "defect_files_today_raw": defect_files_today,
        "avg_tries_before_success_all_completions": avg_tries_all,
        "avg_tries_before_success_excluding_zero": avg_tries_nonzero,
        "completions_with_no_prior_defect_found": zero_prior,
        "tries_distribution": dict(sorted(dist.items(), key=lambda kv: kv[0])),
        "worst_sn_by_tries": worst,
    }


def print_report(metrics: dict) -> None:
    print(f"\nRepair metrics for day folder: {metrics['target_date']}")
    print(f"Lookback window: {metrics['lookback_days']} days (scanned: {', '.join(metrics['dates_scanned'])})\n")

    print(f"Completed repairs (unique SN in OK): {metrics['completed_repairs_unique_sn']}")
    print(f"Unique SN with defects today:        {metrics['defect_unique_sn_today']}")
    print(f"Unique SN defect->OK same day:       {metrics['success_today_unique_sn']}")

    sr = metrics["success_rate_same_day"]
    if sr is None:
        print("Same-day repair success rate:        N/A (no defects found today)")
    else:
        print(f"Same-day repair success rate:        {sr:.2%}")

    print(f"\nRepair workload today (unique rounds/min): {metrics['defect_rounds_today_unique_minutes']}")
    print(f"Repair workload today (raw defect jpgs):  {metrics['defect_files_today_raw']}")

    avg_all = metrics["avg_tries_before_success_all_completions"]
    if avg_all is None:
        print("\nAvg tries before success: N/A (no completions in OK today)")
    else:
        print(f"\nAvg tries before success (all completions): {avg_all:.3f}")
        avg_nz = metrics["avg_tries_before_success_excluding_zero"]
        if avg_nz is not None:
            print(f"Avg tries before success (excluding 0):    {avg_nz:.3f}")
        print(f"Completions with 0 prior defects found:    {metrics['completions_with_no_prior_defect_found']}")

    print("\nDistribution: tries -> count")
    for tries, count in metrics["tries_distribution"].items():
        print(f"  {tries:>2} -> {count}")

    if metrics["worst_sn_by_tries"]:
        print("\nWorst SNs by tries (top 15):")
        for sn, tries in metrics["worst_sn_by_tries"]:
            print(f"  {sn}: {tries}")


def main():
    ap = argparse.ArgumentParser(description="Compute repair success/tries metrics from day-folder EL jpgs.")
    ap.add_argument("--root", required=True, help="Root folder containing date folders like 20251220, 20251221...")
    ap.add_argument("--date", required=True, help="Target date folder YYYYMMDD (e.g., 20251221)")
    ap.add_argument("--lookback", type=int, default=2, help="How many days back to search for defects (default 2)")
    ap.add_argument("--ext", default=".jpg", help="Image extension (default .jpg)")
    ap.add_argument("--time-source", choices=["mtime", "ctime"], default="mtime",
                    help="Timestamp source for grouping rounds: mtime (default) or ctime")
    args = ap.parse_args()

    metrics = compute_metrics(
        root=Path(args.root),
        target_date=args.date,
        lookback_days=args.lookback,
        ext=args.ext,
        time_source=args.time_source,
    )
    print_report(metrics)


if __name__ == "__main__":
    main()
