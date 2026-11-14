# tools/fetch_eigen_test.py - download only the 697 Eigen test frames (image_02 png + velodyne bin)
# and the calibration files, by reading single files out of the remote KITTI raw zips
# (HTTP range requests, needs `pip install remotezip`). ~1.8 GB instead of the full drives.
# Usage (from an empty folder, which becomes evaluate_depth.py --data_path):
#   python fetch_eigen_test.py /path/to/monodepth2/splits/eigen/test_files.txt
# Pull only the Eigen test frames (left image + velodyne scan) out of the remote KITTI raw zips.
import os, sys, urllib.request, zipfile, io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
if len(sys.argv) != 2:
    sys.exit("usage: python fetch_eigen_test.py <monodepth2>/splits/eigen/test_files.txt")
lines = [l.split() for l in open(sys.argv[1])]
need = defaultdict(set)
for folder, idx, _ in lines:
    need[folder].add(int(idx))
dates = sorted({f.split("/")[0] for f in need})
for d in dates:                       # calibration files
    if not os.path.exists(f"{d}/calib_cam_to_cam.txt"):
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f"{BASE}/{d}_calib.zip").read()))
        z.extractall(".")
def get(folder):
    drive = folder.split("/")[1].replace("_sync", "")
    with RemoteZip(f"{BASE}/{drive}/{drive}_sync.zip") as z:
        for i in sorted(need[folder]):
            for n in (f"{folder}/image_02/data/{i:010d}.png", f"{folder}/velodyne_points/data/{i:010d}.bin"):
                if not os.path.exists(n):
                    z.extract(n, ".")
    return folder
with ThreadPoolExecutor(8) as ex:
    for f in ex.map(get, sorted(need)):
        print("done", f, flush=True)
