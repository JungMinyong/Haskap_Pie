from pathlib import Path
import sys
import numpy as np
import yt
import os

def merge_snapshot(I, snap_name: str, part_dict: dict):
    snap_prefix = Path(snap_name)        # .../DD2035/DD2035
    snap_id     = snap_prefix.name       # DD2035
    files = sorted(IN_DIR.glob(f"{snap_id}_rank*.npz"))

    if not files:
        print(f"[{snap_name}] WARNING: no rank files found")
        return

    pos_all = []
    vel_all = []
    pindex_all = []
    ptype_all = []

    for f in files:
        d = np.load(f)
        pos_all.append(d["pos"])
        vel_all.append(d["vel"])
        pindex_all.append(d["pindex"])
        ptype_all.append(d["ptype"])

    pos_all = np.concatenate(pos_all) if pos_all else np.empty((0, 3))
    vel_all = np.concatenate(vel_all) if vel_all else np.empty((0, 3))
    pindex_all = (
        np.concatenate(pindex_all) if pindex_all else np.empty((0,), dtype=np.int64)
    )
    ptype_all = np.concatenate(ptype_all) if ptype_all else np.empty((0,), dtype=np.int64)

    metadata = np.load(savestring +  '/metadata_%s.npz' % fldn)
    level_info = metadata['level_info']
    min_mass = float(metadata['min_mass'])

    msk_ptype4 = (ptype_all == 4)

    #pos_temp = pos_all[msk_mostrefined]
    #identify region encompassing ptype==4
    pos_temp = pos_all[msk_ptype4]
    ll = np.min(pos_temp,axis=0)
    ur = np.max(pos_temp,axis=0)
    buffer = 0.05 * (ur - ll)
    ll = ll - buffer
    ur = ur + buffer

    msk_inregion = (
        (pos_all[:,0] >= ll[0]) & (pos_all[:,0] <= ur[0]) &
        (pos_all[:,1] >= ll[1]) & (pos_all[:,1] <= ur[1]) &
        (pos_all[:,2] >= ll[2]) & (pos_all[:,2] <= ur[2])
    )
    pindex_all = pindex_all[msk_inregion]

    ds = ts[I]
    pos_all = ds.arr(pos_all[msk_inregion], "code_length").to("m")
    vel_all = ds.arr(vel_all[msk_inregion], "code_velocity").to("m/s")
    min_mass = ds.quan(min_mass, "Msun").to("kg").v #it's float for consistency with the original code
    

    #change it in reverse order
    i_level = len(level_info) - 1
    mass_all = np.full(len(vel_all), min_mass * (8 ** i_level))

    while i_level >= 0:
        I_start, I_end = level_info[i_level, 1], level_info[i_level, 2]
        msk_level = ( (pindex_all >= I_start) & (pindex_all <= I_end) )
        if np.sum(msk_level) == 0:
            raise RuntimeError("No particles found in level range, probably no maximum refinement level particles in the region")
        
        mass_all[msk_level] = min_mass * (8 ** i_level)
        i_level -= 1



    part = {}
    part['pos'], part['mass'], part['vel'], part['ids'] = pos_all, mass_all, vel_all, pindex_all

    np.save(OUT_DIR / f'part_{I}_0.npy', part)
    part_dict[I] = {
        'll': np.array([ll]),
        'ur': np.array([ur]),
    }
    OUT_REFINED_DIR = Path(savestring) / "Refined"
    OUT_REFINED_DIR.mkdir(parents=True, exist_ok=True)
    refined_path = OUT_REFINED_DIR / f'refined_region_{I}.npy'
    if not os.path.exists(refined_path):
        np.save(refined_path, np.array([ll, ur]))
    else:
        print(f"[{snap_name}] Refined region file already exists.")

    print(
        f"[{snap_name}] merged: "
        f"pos={pos_all.shape}, vel={vel_all.shape}, pindex={pindex_all.shape}"
    )




if __name__ == "__main__":
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

    fld_list = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[:,0]
    ts = yt.DatasetSeries(fld_list)
    IN_DIR  = Path(savestring) / "resave_splited"
    OUT_DIR = Path(savestring) / "particle_save"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    part_dict = {}
    for I, snap_name in enumerate(fld_list):
        merge_snapshot(I, snap_name, part_dict)
    if part_dict:
        np.save(OUT_DIR / 'part_dict.npy', part_dict)
