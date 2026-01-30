import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))
import run_haskap as rh


def main():
    if len(sys.argv) != 5:
        if rh.rank == 0:
            print(sys.argv)
            print('Need simulation file path and code type to run as well as whether to pre-run refined region, ex: python run_halo_saved.py simulation/folder/path ENZO save/path skip_number')
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
        if rh.rank == 0:
            print('Missing pfs_allsnaps file; run save_particles.py first.')
        return

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

    meta_path = savestring + '/particle_save/particle_meta.npy'
    if not os.path.exists(meta_path):
        if rh.rank == 0:
            print('Missing particle metadata; run save_particles.py to create %s.' % meta_path)
        return

    rh.use_saved_particles_only = True
    rh.saved_particle_meta = rh.load_saved_particle_meta()
    meta_config = rh.saved_particle_meta.get('_config')
    if meta_config is None:
        if rh.rank == 0:
            print('Missing _config in particle_meta; rerun save_particles.py to regenerate metadata.')
        return

    rh.minmass = meta_config['minmass']
    rh.refined = meta_config['refined']
    rh.find_dm = meta_config.get('find_dm', rh.find_dm)
    rh.find_stars = meta_config.get('find_stars', rh.find_stars)

    if rh.rank == 0:
        print('Running halo finder using saved particles only.')

    rh.Evolve_Tree(plot=False, codetp=code, skip_large=False, verbose=False, from_tree=False,
        last_timestep=rh.last_timestep, multitree=True, refined=rh.refined, video=False,
        trackbig=False, tracktime=True)


if __name__ == '__main__':
    main()
