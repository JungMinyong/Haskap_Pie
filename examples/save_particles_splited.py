import os
import time
from pathlib import Path
import h5py
import numpy as np
import sys

# --------- SLURM rank info (works with srun -n N) ----------
rank = int(os.environ.get("SLURM_PROCID", 0))
nranks = int(os.environ.get("SLURM_NTASKS", 1))

print(f"[rank {rank:03d}] starting up, total ranks = {nranks}")


if len(sys.argv) !=5:
    print(sys.argv)
    print('Need simulation file path and code type to run as well as whether to pre-run refined region, ex: python shinbad.py simulation/folder/path ENZO save/path skip_number')
    sys.exit(1)

string = sys.argv[1]
code = sys.argv[2]
savestring = sys.argv[3]
skip = int(sys.argv[4])
print(string,code,savestring,skip)
fldn = 2019

if skip != 1:
    print('warning: skip is not 1, but this script does not support skipping. Proceeding anyway.')

    
fld_list = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[:,0]
#this give [/mnt/home/mjung/ceph/foggie_snaps/DD2035/DD2035, ...]

# --------- user config ----------
OUT_DIR  = Path(savestring) / 'resave_splited'  # put outputs somewhere
OUT_DIR.mkdir(parents=True, exist_ok=True)

DM_TYPE = 4

# snapshot range
RD_START = 2035
RD_END   = 2056  # inclusive

def iter_cpu_files(snap_prefix: Path):
    # snap_prefix: .../DD2035/DD2035
    snap_dir = snap_prefix.parent        # .../DD2035
    prefix   = snap_prefix.name          # DD2035
    return sorted(snap_dir.glob(f"{prefix}.cpu*"))

def process_snapshot(snap_name: str):
    snap_prefix = Path(snap_name)        # .../DD2035/DD2035
    snap_id     = snap_prefix.name       # DD2035

    cpu_files = iter_cpu_files(snap_prefix)

    if not cpu_files:
        if rank == 0:
            print(f"[{snap_id}] WARNING: no cpu files found near {snap_prefix}")
        return

    t0 = time.time()

    pos_local = []
    vel_local = []
    pindex_local = []
    ptype_local = []

    for i, fname in enumerate(cpu_files):
        if i % nranks != rank:
            continue

        with h5py.File(fname, "r") as f:
            for gname in f:
                if not gname.startswith("Grid"):
                    continue

                g = f[gname]
                if "particle_type" not in g:
                    continue

                ptype = g["particle_type"][...]
                msk = (ptype == 4) | (ptype == 1)
                if not np.any(msk):
                    continue

                px = g["particle_position_x"][...][msk]
                py = g["particle_position_y"][...][msk]
                pz = g["particle_position_z"][...][msk]
                vx = g["particle_velocity_x"][...][msk]
                vy = g["particle_velocity_y"][...][msk]
                vz = g["particle_velocity_z"][...][msk]
                pid = g["particle_index"][...][msk]
                ptype = g["particle_type"][...][msk]

                pos_local.append(np.vstack([px, py, pz]).T)
                vel_local.append(np.vstack([vx, vy, vz]).T)
                pindex_local.append(pid)
                ptype_local.append(ptype)

    pos_local = np.concatenate(pos_local) if pos_local else np.empty((0, 3), dtype=np.float64)
    vel_local = np.concatenate(vel_local) if vel_local else np.empty((0, 3), dtype=np.float64)
    pindex_local = np.concatenate(pindex_local) if pindex_local else np.empty((0,), dtype=np.int64)
    ptype_local = np.concatenate(ptype_local) if ptype_local else np.empty((0,), dtype=np.int64)

    out_path = OUT_DIR / f"{snap_id}_rank{rank:03d}.npz"
    np.savez(out_path, pos=pos_local, vel=vel_local, pindex=pindex_local, ptype=ptype_local)

    print(f"[rank {rank:03d}] {snap_id}: N={len(pindex_local)}  time={time.time()-t0:.2f}s")


def main():
    for rd in range(len(fld_list)):
        snap_name = fld_list[rd]
        process_snapshot(snap_name)

if __name__ == "__main__":
    main()