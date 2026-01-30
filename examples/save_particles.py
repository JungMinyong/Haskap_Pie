import os
import sys
import numpy as np
import glob
import gc

sys.path.append(os.path.dirname(__file__))
import run_haskap as rh
rh.yt.enable_parallelism()

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
        if rh.refined:
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
        else:
            ds_for_resave, _ = rh.open_ds(0, rh.code)
            ll_all = np.array(ds_for_resave.domain_left_edge)
            ur_all = np.array(ds_for_resave.domain_right_edge)
            del ds_for_resave
        for t in timelist:
            ds, meter = rh.open_ds(t, rh.code)
            print(t)
            ll = np.array([ll_all])
            ur = np.array([ur_all])
            reg = ds.box(ll_all, ur_all)
            if rh.find_dm is True and rh.find_stars is False:
                mass, pos, vel, ids = rh.pickup_particles(reg, rh.code, stars=rh.find_stars, find_dm=rh.find_dm)
                bool_reg = (np.sum(pos >= ll_all * meter, axis=1) == 3) * (np.sum(pos < ur_all * meter, axis=1) == 3)
                mass, pos, vel, ids = mass[bool_reg], pos[bool_reg], vel[bool_reg], ids[bool_reg]
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
            refined_files = glob.glob(savestring + '/Refined/refined_region_*.npy')
            if len(refined_files) == 0:
                rh.make_refined()

    if rh.rank == 0:
        print('Save iteration: ', rh.fldn)

    if not rh.refined:
        rh.resave = False

    if rh.resave:
        resave_particles_serial()
        save_part = savestring + '/particle_save'
        part_dict_path = save_part + '/part_dict.npy'
        if os.path.exists(part_dict_path):
            part_dict = np.load(part_dict_path, allow_pickle=True).tolist()
            meta = {}
            for t in sorted(part_dict.keys()):
                ds, meter = rh.open_ds(t, code)
                length_unit_m = ds.length_unit.in_units('m').v
                meta[t] = {
                    'current_time_s': ds.current_time.in_units('s').v,
                    'current_redshift': float(ds.current_redshift),
                    'domain_left_edge_m': ds.domain_left_edge.in_units('m').v,
                    'domain_right_edge_m': ds.domain_right_edge.in_units('m').v,
                    'length_unit_m': length_unit_m,
                    'mass_unit_kg': ds.mass_unit.in_units('kg').v,
                    'mass_unit_Msun': ds.mass_unit.in_units('Msun').v,
                    'hubble_constant': float(ds.hubble_constant),
                    'omega_matter': float(ds.omega_matter),
                    'omega_lambda': float(ds.omega_lambda),
                    'meter': float(meter),
                }
                del ds
                gc.collect()
            meta['_config'] = {
                'minmass': rh.minmass,
                'refined': rh.refined,
                'find_dm': rh.find_dm,
                'find_stars': rh.find_stars,
                'skip': rh.skip,
                'fldn': rh.fldn,
            }
            if rh.yt.is_root():
                np.save(save_part + '/particle_meta.npy', meta)
        else:
            if rh.rank == 0:
                print('Missing part_dict.npy; particle metadata not saved.')
    else:
        if rh.rank == 0:
            print('Refined region not found; particle save skipped.')


if __name__ == '__main__':
    main()
