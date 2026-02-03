import yt
import sys
import numpy as np
import glob
import os
#====== make pfs ========

def make_pfs_allsnaps(folder, folder2, codetp, base_manual=None,
                     checkredshift=True, index=0):
    """
    Create a text file listing snapshot paths, redshifts, and current times,
    sorted by redshift (descending) or by filename index (ascending).

    Single-core version (no yt.parallel_objects / MPI required).
    """

    # ---- find snapshot configuration files ----
    snapshot_files = []

    if codetp in ("ENZO", "AGORA_ENZO"):
        bases = [["DD", "output_"],
                 ["DD", "output"],
                 ["DD", "data"],
                 ["DD", "DD"]]  # remove RD
        for b in bases:
            snapshot_files += glob.glob(f"{folder}/{b[0]}????/{b[1]}????")

    elif codetp in ("GADGET3", "GADGET4"):
        bases = [["snapshot_", "snapshot_"],
                 ["snapdir_", "snapshot_"]]
        for b in bases:
            snapshot_files += glob.glob(f"{folder}/{b[0]}???/{b[1]}???.0.hdf5")

    elif codetp == "AREPO":
        for b in ["snap_", "snapshot_"]:
            snapshot_files += glob.glob(f"{folder}/{b}???.hdf5")

    elif codetp == "GIZMO":
        snapshot_files += glob.glob(f"{folder}/snapshot_???.hdf5")

    elif codetp == "GEAR":
        snapshot_files += glob.glob(f"{folder}/snapshot_????.hdf5")

    elif codetp == "RAMSES":
        snapshot_files += glob.glob(f"{folder}/output_?????/info_?????.txt")

    elif codetp == "ART":
        snapshot_files += glob.glob(f"{folder}/10MpcBox_csf512_?????.d")

    elif codetp == "CHANGA":
        snapshot_files += glob.glob(f"{folder}/ncal-IV.??????")

    elif codetp == "manual":
        snapshot_files += glob.glob(base_manual)

    # ---- sorting / metadata ----
    if not checkredshift:
        raise NotImplementedError("checkredshift=False is not implemented yet.")
    else:
        snapshot_redshifts = np.empty(len(snapshot_files), dtype=float)
        current_time = np.empty(len(snapshot_files), dtype=float)

        for i, file_dir in enumerate(snapshot_files):
            ds = yt.load(file_dir)
            snapshot_redshifts[i] = float(ds.current_redshift)
            current_time[i] = float(ds.current_time.in_units("Myr").v)

        # sort by redshift descending
        arg_z = np.argsort(-snapshot_redshifts)
        snapshot_files = np.array(snapshot_files, dtype=str)[arg_z].tolist()
        snapshot_redshifts = snapshot_redshifts[arg_z]
        current_time = current_time[arg_z]

    # ---- write output ----
    outpath = f"{folder2}/pfs_allsnaps_{index}.txt"
    print(outpath)

    final_file = []
    for i in range(len(snapshot_files)):
        final_file.append([snapshot_files[i], snapshot_redshifts[i], current_time[i]])

    np.savetxt(outpath, final_file, fmt="%s")


if len(sys.argv) !=5:
    print(sys.argv)
    print('Need simulation file path and code type to run as well as whether to pre-run refined region, ex: python shinbad.py simulation/folder/path ENZO save/path skip_number')

string = sys.argv[1]
code = sys.argv[2]
savestring = sys.argv[3]
skip = int(sys.argv[4])
print(string,code,savestring,skip)
fldn = 2019
os.makedirs(savestring, exist_ok=True)

if not os.path.exists(savestring +  '/pfs_allsnaps_%s.txt' % fldn):
    fld_list = make_pfs_allsnaps(string,savestring,code,index=fldn)
else:
    print('pfs_allsnaps_%s.txt already exists' % fldn)


print('Making metadata file...')

savestring +  '/metadata_%s.npz' % fldn
if not os.path.exists(savestring +  '/metadata_%s.npz' % fldn):
    ds = yt.load(np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[0,0])
    ptype = ds.r['all','particle_type'].v
    pmass = ds.r['all','particle_mass'].to("Msun").v
    pid = ds.r['all','particle_index'].v

    msk_dm = (ptype == 1) | (ptype == 4)  # DM particle types in yt
    msk_dm = msk_dm & (pmass >= 0)
    pmass = pmass[msk_dm]
    pid = pid[msk_dm]
    
    min_mass = np.min(pmass)
    temp = pmass[(pmass < min_mass * 1.5) & (pmass > min_mass * 0.5)]
    min_mass = np.median(temp)  # to avoid outlier

    level_lst = []
    level = 0
    M1 = min_mass
    
    while True:
        pid_mostrefined = pid[(pmass < M1 * 1.5) & (pmass > M1 * 0.5)]
        if (len(pid_mostrefined) == 0):
            break
        idx_start = np.min(pid_mostrefined)
        idx_end = np.max(pid_mostrefined)
        level_lst.append((level, idx_start, idx_end))

        if idx_end - idx_start + 1 == len(pid_mostrefined):
            if idx_start < 100: #if they are likely to the top level ids. It was 0 in my test, but just in case
                print('likely top level ids, stop here')
                break
            raise RuntimeError("Unexpected: contiguous ids but not matching length")
        
        M1 = M1 * 8
        level += 1

    level_lst = np.array(level_lst)
    #save idx_start, idx_end, min_mass
    np.savez(savestring +  '/metadata_%s.npz' % fldn, min_mass = min_mass, level_info = level_lst)


else:
    print('metadata_%s.npz already exists' % fldn)
