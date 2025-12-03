#!/usr/bin/env python3
"""
Simple concurrent distributor: push a list of source paths to 10 hosts,
evenly distributed across 4 SSD mountpoints (/data_0..3/mlcd) on each host.

Usage:
    python3 copy_distribute.py [--dry-run] [--max-parallel N] [--user USER]

Requirements:
    - rsync and ssh must be available on the machine running this script.
    - passwordless SSH (key-based) to target hosts for the specified user.
"""

import os
import sys
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----- Config (you can edit or pass via CLI) -----
hosts = [
    "172.16.5.19",
    "172.16.5.27",
    "172.16.5.81",
    "172.16.5.82",
    "172.16.5.85",
    "172.16.5.86",
    "172.16.5.87",
    "172.16.5.88",
    "172.16.5.89",
    "172.16.5.90",
    "172.16.5.91",
    "172.16.5.92",
    "172.16.5.93",
    "172.16.5.94",
    "172.16.5.95",
    "172.16.5.96",
]

list_coyo = [
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_001",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_002",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_003",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_004",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_005",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_006",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_007",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_008",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_009",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_010",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_011",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_012",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_013",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_014",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_015",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_016",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_017",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_018",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_019",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_020",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_021",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_022",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_023",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_024",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_025",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_026",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_027",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_028",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_029",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_030",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_031",
    "/rice_ocr/dataset_ocr/source/zero250m/obelics_00300_00600_part_032",
]


# ----- End config -----


def check_prereqs():
    from shutil import which

    if which("rsync") is None:
        print("Error: rsync not found in PATH. Install rsync and retry.")
        sys.exit(1)
    if which("ssh") is None:
        print("Error: ssh not found in PATH. Install OpenSSH client and retry.")
        sys.exit(1)


def build_assignment(sources, hosts):
    """
    Round-robin assign sources across hosts, then round-robin to specific disks within each host.
    Returns list of tuples: (src, host_ip, remote_target_dir)
    """
    assignments = []
    host_count = len(hosts)
    per_host_counter = [0] * host_count

    # Define the specific disks you want to use
    target_disks = [2, 3]

    for i, src in enumerate(sources):
        hidx = i % host_count

        # Calculate which disk index to use from your target list
        # The modulo operator (%) ensures we cycle 0, 1, 0, 1...
        disk_list_index = per_host_counter[hidx] % len(target_disks)
        disk_num = target_disks[disk_list_index]

        per_host_counter[hidx] += 1

        # Use the selected disk number (2 or 3)
        remote_base = f"/data_{disk_num}/llava_vit_ocr_zero250m"

        basename = os.path.basename(src.rstrip("/"))
        remote_dir = os.path.join(remote_base, basename)
        assignments.append((src, hosts[hidx], remote_dir))
    return assignments


def remote_mkdir(user, ip, path, ssh_opts):
    cmd = ["ssh"] + ssh_opts + [f"{user}@{ip}", f"mkdir -p '{path}'"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def rsync_push(user, ip, src, target_dir, ssh_opts, rsync_opts):
    # If src is dir -> sync directory (copy contents into target_dir/<basename>)
    if os.path.isdir(src):
        # We'll copy the directory (with trailing slash) into target_dir/
        src_spec = src.rstrip("/") + "/"
        remote = f"{user}@{ip}:'{target_dir}/'"
        cmd = ["rsync"] + rsync_opts + ["-e", "ssh " + " ".join(ssh_opts), src_spec, remote]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # If src is file -> copy file into target_dir/
    if os.path.isfile(src):
        remote = f"{user}@{ip}:'{target_dir}/'"
        cmd = ["rsync"] + rsync_opts + ["-e", "ssh " + " ".join(ssh_opts), src, remote]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # If prefix (.rec/.idx)
    rec = src + ".rec"
    idx = src + ".idx"
    results = []
    if os.path.isfile(rec):
        remote = f"{user}@{ip}:'{target_dir}/'"
        cmd = ["rsync"] + rsync_opts + ["-e", "ssh " + " ".join(ssh_opts), rec, remote]
        results.append(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    if os.path.isfile(idx):
        remote = f"{user}@{ip}:'{target_dir}/'"
        cmd = ["rsync"] + rsync_opts + ["-e", "ssh " + " ".join(ssh_opts), idx, remote]
        results.append(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    if not results:
        return None
    # Return last result for status
    return results[-1]


def task_push(item, user, ssh_opts, rsync_opts, dry_run=False):
    src, ip, remote_dir = item
    print(f"[ASSIGN] {src} -> {ip}:{remote_dir}")
    if dry_run:
        return {"item": item, "ok": True, "dry": True}

    # create remote dir
    r = remote_mkdir(user, ip, remote_dir, ssh_opts)
    if r.returncode != 0:
        return {"item": item, "ok": False, "err": f"mkdir failed: {r.stderr.decode().strip()}"}

    res = rsync_push(user, ip, src, remote_dir, ssh_opts, rsync_opts)
    if res is None:
        return {"item": item, "ok": False, "err": "source missing (.rec/.idx/dir/file not found)"}
    if res.returncode != 0:
        return {"item": item, "ok": False, "err": res.stderr.decode(errors="ignore")}
    return {"item": item, "ok": True, "out": res.stdout.decode(errors="ignore")}


def main():
    parser = argparse.ArgumentParser(description="Distribute data to multiple hosts (simple).")
    parser.add_argument("--dry-run", action="store_true", help="Only print assignment, do not copy")
    parser.add_argument("--max-parallel", type=int, default=50, help="Max concurrent rsync tasks")
    parser.add_argument("--user", type=str, default=os.getenv("USER", "root"), help="SSH user")
    args = parser.parse_args()

    check_prereqs()

    all_sources = list_coyo
    if not all_sources:
        print("No sources defined. Exiting.")
        return

    assignments = build_assignment(all_sources, hosts)
    print(f"Total items: {len(all_sources)}; Hosts: {len(hosts)}; Max parallel: {args.max_parallel}")
    print("Assignments preview")
    for a in assignments:
        print("  ", a)

    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    rsync_opts = ["-a", "--partial", "--inplace", "--no-compress", "--progress"]

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as exe:
            futures = {exe.submit(task_push, item, args.user, ssh_opts, rsync_opts, args.dry_run): item for item in assignments}
            for fut in as_completed(futures):
                res = fut.result()
                item = res["item"]
                if res.get("dry"):
                    print(f"[DRY] {item[0]} -> {item[1]}:{item[2]}")
                elif res["ok"]:
                    print(f"[OK ] {item[0]} -> {item[1]}:{item[2]}")
                else:
                    print(f"[ERR] {item[0]} -> {item[1]}:{item[2]}  : {res.get('err')}")
                results.append(res)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        return

    # Summary
    ok = sum(1 for r in results if r.get("ok"))
    err = len(results) - ok
    print(f"Done. Success: {ok}, Failed: {err}")


if __name__ == "__main__":
    main()
