#it is hard-coded in the case of Enzo and particle_type == 4

import os
import sys
import numpy as np
import glob

sys.path.append(os.path.dirname(__file__))
import run_haskap as rh
rh.yt.enable_parallelism()

#in the case of FOGGIE Maelstrom,
#StaticRefineRegionLeftEdge[3] = 0.4755859375 0.4892578125 0.4809570312
#StaticRefineRegionRightEdge[3] = 0.529296875 0.5166015625 0.5234375
box_left_edge = [0.4755859375, 0.4892578125, 0.4809570312]
box_right_edge = [0.529296875, 0.5166015625, 0.5234375]


def _refined_region_enzo_ptype4(reg):
    part_type = reg[('all', 'particle_type')].v.astype(np.int64)
    pos = reg[('all', 'particle_position')].to('code_length').v
    mask = part_type == 4
    if not np.any(mask):
        raise ValueError('No ENZO particle_type == 4 particles found for refined region.')
    pos4 = pos[mask]
    ll = pos4.min(axis=0)
    ur = pos4.max(axis=0)
    return ll, ur

def resave_particles_serial():
    skip = rh.skip
    save_part = rh.savestring + '/particle_save'
    if os.path.exists(save_part + '/part_dict.npy'):
        part_dict = np.load(save_part + '/part_dict.npy', allow_pickle=True).tolist()
    else:
        part_dict = {}
    if len(part_dict) == 0:
        interval, timelist = rh.inteval_timelist(skip, rh.fld_list)
        sto = {}
        rh.ensure_dir(save_part)
        minmass_floor = 100.0
        minmass_local = np.inf
        use_enzo_ptype4_refine = rh.refined and rh.code == 'ENZO' and getattr(rh, 'enzo_type4_only', False)
        ll_all, ur_all = None, None
        if rh.refined and not use_enzo_ptype4_refine:
            refined_times = np.array([])
            for times in timelist:
                len_now = len(timelist[timelist >= times])
                if (len_now % interval == 0 or times == rh.last_timestep) and times > 0:
                    refined_times = np.append(refined_times, times)
            ll_all, ur_all = np.array([1e89, 1e89, 1e89]), -1 * np.array([1e89, 1e89, 1e89])
            for times in refined_times:
                ll_o, ur_o = np.load(
                    rh.savestring + '/Refined/' + 'refined_region_%s.npy' % int(times),
                    allow_pickle=True
                ).tolist()
                ll_all = np.minimum(ll_all, ll_o)
                ur_all = np.maximum(ur_all, ur_o)
            buffer = (np.array(ur_all) - np.array(ll_all)) * 0.05
            ll_all, ur_all = np.array(ll_all) - buffer, np.array(ur_all) + buffer
        elif not rh.refined:
            ds_for_resave, _ = rh.open_ds(0, rh.code)
            ll_all = np.array(ds_for_resave.domain_left_edge)
            ur_all = np.array(ds_for_resave.domain_right_edge)
            del ds_for_resave
        for t in timelist:
            ds, meter = rh.open_ds(t, rh.code)
            print(t)
            if use_enzo_ptype4_refine:
                reg = ds.box(box_left_edge, box_right_edge)
                ll_ref, ur_ref = _refined_region_enzo_ptype4(reg)
                if rh.yt.is_root():
                    np.save(
                        rh.savestring + '/Refined/' + 'refined_region_%s.npy' % int(t),
                        np.array([ll_ref, ur_ref]),
                    )
                buffer = (ur_ref - ll_ref) * 0.05
                ll_all, ur_all = ll_ref - buffer, ur_ref + buffer
            else:
                reg = ds.box(ll_all, ur_all)
            ll = np.array([ll_all])
            ur = np.array([ur_all])
            if rh.find_dm is True and rh.find_stars is False:
                mass, pos, vel, ids = rh.pickup_particles(reg, rh.code, stars=rh.find_stars, find_dm=rh.find_dm)
                bool_reg = (np.sum(pos >= ll_all * meter, axis=1) == 3) * (np.sum(pos < ur_all * meter, axis=1) == 3)
                mass, pos, vel, ids = mass[bool_reg], pos[bool_reg], vel[bool_reg], ids[bool_reg]
                kg_to_msun = ds.mass_unit.in_units('Msun').v / ds.mass_unit.in_units('kg').v
                mass_msun = mass * kg_to_msun
                mass_msun = mass_msun[mass_msun > minmass_floor]
                if mass_msun.size > 0:
                    minmass_local = min(minmass_local, float(mass_msun.min()))
            elif rh.find_dm is False and rh.find_stars is True:
                spos, svel, sids = rh.pickup_particles(reg, rh.code, stars=rh.find_stars, find_dm=rh.find_dm)
                sbool_reg = (np.sum(spos >= ll_all * meter, axis=1) == 3) * (np.sum(spos < ur_all * meter, axis=1) == 3)
                spos, svel, sids = spos[sbool_reg], svel[sbool_reg], sids[sbool_reg]
            elif rh.find_dm is True and rh.find_stars is True:
                mass, pos, vel, ids, spos, svel, sids = rh.pickup_particles(
                    reg, rh.code, stars=rh.find_stars, find_dm=rh.find_dm
                )
                bool_reg = (np.sum(pos >= ll_all * meter, axis=1) == 3) * (np.sum(pos < ur_all * meter, axis=1) == 3)
                mass, pos, vel, ids = mass[bool_reg], pos[bool_reg], vel[bool_reg], ids[bool_reg]
                kg_to_msun = ds.mass_unit.in_units('Msun').v / ds.mass_unit.in_units('kg').v
                mass_msun = mass * kg_to_msun
                mass_msun = mass_msun[mass_msun > minmass_floor]
                if mass_msun.size > 0:
                    minmass_local = min(minmass_local, float(mass_msun.min()))
                sbool_reg = (np.sum(spos >= ll_all * meter, axis=1) == 3) * (np.sum(spos < ur_all * meter, axis=1) == 3)
                spos, svel, sids = spos[sbool_reg], svel[sbool_reg], sids[sbool_reg]
            sto[t] = {}
            sto[t]['ll'] = ll
            sto[t]['ur'] = ur
            for v in range(len(ll)):
                if rh.find_dm is True:
                    part = {}
                    bool_in = (np.sum(pos >= ll[v] * meter, axis=1) == 3) * (np.sum(pos < ur[v] * meter, axis=1) == 3)
                    part['pos'], part['mass'], part['vel'], part['ids'] = pos[bool_in], mass[bool_in], vel[bool_in], ids[bool_in]
                    if rh.find_stars is True:
                        sbool_in = (np.sum(spos >= ll[v] * meter, axis=1) == 3) * (np.sum(spos < ur[v] * meter, axis=1) == 3)
                        part['spos'], part['svel'], part['sids'] = spos[sbool_in], svel[sbool_in], sids[sbool_in]
                    #yt.is_root()
                    if rh.yt.is_root():
                        np.save(save_part + '/part_%s_%s.npy' % (t, v), part)
                elif rh.find_dm is False and rh.find_stars is True:
                    part = np.load(save_part + '/part_%s_%s.npy' % (t, v), allow_pickle=True).tolist()
                    sbool_in = (np.sum(spos >= ll[v] * meter, axis=1) == 3) * (np.sum(spos < ur[v] * meter, axis=1) == 3)
                    part['spos'], part['svel'], part['sids'] = spos[sbool_in], svel[sbool_in], sids[sbool_in]
                    if rh.yt.is_root():
                        np.save(save_part + '/part_%s_%s.npy' % (t, v), part)
            if rh.find_dm is True:
                mass, pos, vel, ids = 0, 0, 0, 0
            if rh.find_stars is True:
                spos, svel, sids = 0, 0, 0
            reg = 0
            ds = 0
        if rh.yt.is_root():
            np.save(save_part + '/part_dict.npy', sto)
        minmass_global = rh.comm.allreduce(minmass_local, op=rh.MPI.MIN)
        if not np.isfinite(minmass_global):
            minmass_global = None
        if rh.yt.is_root():
            np.save(save_part + '/minmass.npy', {'minmass': minmass_global, 'refined': rh.refined})
        return minmass_global
    return None



def main():
    if len(sys.argv) != 5:
        if rh.rank == 0:
            print(sys.argv)
            print('Need simulation file path and code type to run as well as whether to pre-run refined region, ex: python save_particles.py simulation/folder/path ENZO save/path skip_number')
        return

    string = sys.argv[1]
    code = sys.argv[2]
    savestring = sys.argv[3]
    skip = int(sys.argv[4])

    rh.string = string
    rh.code = code
    rh.savestring = savestring
    rh.skip = skip
    rh.fake = False
    rh.fldn = 2019
    if rh.fake:
        rh.fldn = 1948
    rh.organize_files = False
    rh.Numlength = 1000
    rh.non_sphere = True
    rh.ensure_dir(savestring)
    rh.ensure_dir(savestring + '/Refined')

    if not os.path.exists(savestring + '/pfs_allsnaps_%s.txt' % rh.fldn):
        rh.fld_list = rh.make_pfs_allsnaps(string, savestring, code, index=rh.fldn)
        rh.fld_list = rh.comm.bcast(rh.fld_list, root=0)

    rh.fld_list = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % rh.fldn, dtype=str)[:, 0]
    rh.time_list_1 = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % rh.fldn, dtype=str)[:, 2].astype(float)
    rh.red_list_1 = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % rh.fldn, dtype=str)[:, 1].astype(float)
    rh.u = rh.yt.units
    rh.oden_list = [80, 100, 150, 200, 250, 300, 500, 700, 1000]
    rh.oden1 = False
    rh.oden2 = False
    rh.offset = 0
    rh.make_tree = True
    rh.END = False
    rh.path = string
    rh.find_dm = True
    rh.find_stars = False
    rh.resave = True
    rh.last_timestep = len(rh.fld_list) - 1

    if rh.fake:
        rh.find_stars = False
        rh.resave = True
        rh.minmass, rh.refined = 21232, True
    else:
        rh.minmass, rh.refined = rh.minmass_calc(code)
        if rh.refined:
            if not (code == 'ENZO' and getattr(rh, 'enzo_type4_only', False)):
                refined_files = glob.glob(savestring + '/Refined/refined_region_*.npy')
                if len(refined_files) == 0:
                    rh.make_refined()

    if rh.rank == 0:
        print('Save iteration: ', rh.fldn)

    if not rh.refined:
        rh.resave = False

    if rh.resave:
        recomputed_minmass = resave_particles_serial()
        if recomputed_minmass is not None:
            rh.minmass = recomputed_minmass
        save_part = savestring + '/particle_save'
        if rh.yt.is_root():
            rh.ensure_dir(save_part)
            np.save(save_part + '/minmass.npy', {'minmass': rh.minmass, 'refined': rh.refined})
    else:
        if rh.rank == 0:
            print('Refined region not found; particle save skipped.')


if __name__ == '__main__':
    main()
