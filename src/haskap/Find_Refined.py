import numpy as np
import yt as yt
from yt.data_objects.unions import ParticleUnion
from yt.data_objects.particle_filters import add_particle_filter
import time,os
from collections import Counter

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

class Find_Refined():
    """
    This class returns the refined region of a simulation snapshot.
    The refined region is defined as the region that contains the smallest DM mass and some allowance on the second smallest DM mass.
    For ENZO, an optional mode can instead return the bounding box of particle_type == 4 only.

    Parameters
    ----------
    codetp : str
        The simulation code type. The available options are 'ENZO', 'GEAR', 'GADGET3', 'AREPO', 'GIZMO', 'RAMSES', 'ART', 'CHANGA'

    fld_list : np.array()
        A text file containing the directory to all simulation snapshots

    timestep : int
        The index of the examined snapshot in fld_list

    lim_index : int (default = 0)
        The index of the minimum DM mass that we want to include in the refined region.
        The default value is 0, meaning that we will include the smallest mass and some allowance on the second smallest mass.
        If the value is 1, we will include the second smallest mass and some allowance on the third smallest mass, and so on.
    enzo_type4_only : bool (default = False)
        If True and codetp == 'ENZO', ignore particle masses and return the bounding box
        that encloses particle_type == 4 only.

    Returns
    -------
    self.refined_region : np.array() (2x3)
        The coordinates of the refined region. The first element is the lower-left corner, and the second element is the upper-right corner.
    """
    def __init__(self, codetp, fld_list, timestep, savestring, lim_index = 0, enzo_type4_only = False):
        self.codetp = codetp
        self.lim_index = lim_index
        self.savestring = savestring
        self.timestep = timestep
        self.enzo_type4_only = bool(enzo_type4_only) and self.codetp == 'ENZO'
        if not os.path.exists(self.savestring + '/' + 'refined_region_%s.npy' % (self.timestep)):
            self.open_ds(fld_list)
            self.find_refined_region()
        else:
            self.refined_region = np.load(self.savestring + '/' + 'refined_region_%s.npy' % (self.timestep),allow_pickle=True).tolist()

    def open_ds(self, fld_list):
        """
        This function loads the simulation snapshot and create a Dark Matter particle field (if necessary)
        """
        # The conversion factor from code_length to physical unit is not correct in AGORA’s GADGET3 and AREPO
        if self.codetp == 'GADGET3' or self.codetp == 'AREPO':
            self.ds = yt.load(fld_list[self.timestep], unit_base = {"length": (1.0, "Mpccm/h")})
            #self.ds = yt.load(self.path,  unit_base = {"length": (1.0, "Mpccm/h")})
        else:
            self.ds = yt.load(fld_list[self.timestep])
            #self.ds = yt.load(self.path)

        #filter out dark matter particles for ENZO
        if self.codetp == 'ENZO':
            def darkmatter_init(pfilter, data):
                filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
                filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
                return filter_darkmatter
            add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
            self.ds.add_particle_filter("DarkMatter")
        #combine less-refined particles and refined-particles into one field for GEAR, GIZMO, AREPO, and GADGET3
        if self.codetp == 'GEAR':
            dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
            self.ds.add_particle_union(dm)
        if self.codetp == 'GADGET3':
            dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
            self.ds.add_particle_union(dm)
        if self.codetp == 'AREPO' or self.codetp == 'GIZMO':
            dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
            self.ds.add_particle_union(dm)

    def _enzo_type4_positions(self, reg):
        part_type = reg[("all","particle_type")].v.astype(np.int64)
        pos = reg[("all","particle_position")].to("code_length").v
        return pos[part_type == 4]

    def _reduce_range_enzo_type4(self, numsegs = 10):
        ll_all = self.ds.domain_left_edge.to('code_length').v
        ur_all = self.ds.domain_right_edge.to('code_length').v
        xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                    np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
        _,segdist = np.linspace(ll_all[0],ur_all[0],numsegs,retstep=True)
        ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
        ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
        ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
        ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
        jobs,sto = self.job_scheduler_ref(np.arange(len(ll)))
        ll_reduced = np.array([np.inf, np.inf, np.inf])
        ur_reduced = np.array([-np.inf, -np.inf, -np.inf])
        found = False
        ranks = np.arange(nprocs)
        for rank_now in ranks:
            if rank == rank_now:
                for i in jobs[rank]:
                    buffer = (segdist/100)
                    reg = self.ds.box(ll[i] - buffer ,ur[i] + buffer)
                    pos = self._enzo_type4_positions(reg)
                    if len(pos) > 0:
                        sto[i]["ll"] = pos.min(axis=0)
                        sto[i]["ur"] = pos.max(axis=0)
        for rank_now in ranks:
            for t in jobs[rank_now]:
                sto[t] = comm.bcast(sto[t], root=rank_now)
                if "ll" in sto[t]:
                    ll_reduced = np.minimum(ll_reduced, sto[t]["ll"])
                    ur_reduced = np.maximum(ur_reduced, sto[t]["ur"])
                    found = True
        if not found:
            raise ValueError("No ENZO particle_type == 4 particles found for refined region.")
        return np.vstack((ll_reduced, ur_reduced))

    def job_scheduler_ref(self,out_list):
        '''
        Function to schedule jobs for each rank. This is the implementation of MPI to run parallel loops. Works with any given list.
        Parameters:
            out_list (list): List of jobs to be done
        Returns:
            tuple: Dictionary of jobs for each rank, and a dictionary to store the results
        '''
        ranks = np.arange(nprocs)
        jobs = {i: [] for i in ranks}
        sto = {t: {} for t in out_list}
        if rank == 0:
            count = 0
            while count < len(out_list):
                out_list_2 = np.copy(ranks)
                np.random.shuffle(out_list_2)
                for o in ranks:
                    if count + out_list_2[o] < len(out_list):
                        i = count + out_list_2[o]
                        jobs[o].append(out_list[i])
                count += max(ranks) + 1
            for o in jobs:
                np.random.shuffle(jobs[o])
        jobs = comm.bcast(jobs, root=0)
        return jobs, sto

    def reduce_range(self, numsegs = 10):
        """
        This function divides a specific domain (given by ll_all and ur_all) into (numsegs-1)^3 sub-regions to speed up the loading process.

        Parameters
        ----------
        numsegs : int (default = 10)
            The number of segments that the domain will be divided into. This is chosen according to the number of processors used.

        Return
        -------
        reduced_region : np.array() (2x3)
            The coordinates of the reduced region. The first element is the lower-left corner, and the second element is the upper-right corner.
            The reduced region is the minimum box that contains all the particles at the very most refined level.

        center : np.array() (3)
            The center of mass of the most refined particles in the reduced region. This is the starting point for the extension process.
        """
        ll_all = self.ds.domain_left_edge.to('code_length').v
        ur_all = self.ds.domain_right_edge.to('code_length').v
        xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                    np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))

        _,segdist = np.linspace(ll_all[0],ur_all[0],numsegs,retstep=True)

        ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
        ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
        ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
        ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
        #Runnning parallel to obtain the refined region for each snapshot

        jobs,sto = self.job_scheduler_ref(np.arange(len(ll)))

        ranks = np.arange(nprocs)
        for rank_now in ranks:
            if rank == rank_now:
                for i in jobs[rank]:
                    buffer = (segdist/100) #set buffer when loading the box
                    reg = self.ds.box(ll[i] - buffer ,ur[i] + buffer)

                    lr_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter', 'AREPO': 'DarkMatter', 'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
                    lr_pos = reg[(lr_name_dict[self.codetp],'particle_position')].to('code_length').v
                    lr_m = reg[(lr_name_dict[self.codetp],'particle_mass')].to('Msun').v
                    lr_m = np.ceil(lr_m).astype(np.int64)
                    values, counts = np.unique(lr_m, return_counts=True)
                    #mset_max = np.max(lr_m)

                    posset = {}
                    for j in range(len(values)):
                        posset[values[j]] = np.vstack((lr_pos[lr_m == values[j]].min(axis=0), lr_pos[lr_m == values[j]].max(axis=0)))

                    mset = dict(zip(values, counts))

                    #sto.result = {}
                    sto[i][0] = posset #the volume including all particles of certain mass in the box
                    sto[i][1] = mset
            #sto.result[3] = mset_max

        mset_list = Counter()
        #mset_max_list = np.array([])


        for rank_now in ranks:
            for t in jobs[rank_now]:
                    sto[t] = comm.bcast(sto[t], root=rank_now)
                    mset_list += Counter(sto[t][1])
            #mset_max_list = np.append(mset_max_list,vals[3])

        mset_list = dict(mset_list)
        unique_mass, count = np.array(list(mset_list.keys())), np.array(list(mset_list.values()))
        #Sometimes ENZO has a bug and create very small dark matter particle masses (~ 0 or << most refined particle mass) -> This step removes them
        unique_mass = unique_mass[count/np.sum(count) > 0.005] #this value is tested with AGORA ENZO and box_3_z_1 to contains all the levels
        self.all_m_list = np.sort(unique_mass)

        ll_reduced = np.array([np.inf, np.inf, np.inf])
        ur_reduced = np.array([-np.inf, -np.inf, -np.inf])
        for rank_now in ranks:
            for t in jobs[rank_now]:
                    if self.all_m_list[self.lim_index] in list(sto[t][1].keys()):
                        ll_reduced = np.minimum(ll_reduced, sto[t][0][self.all_m_list[self.lim_index]][0])
                        ur_reduced = np.maximum(ur_reduced, sto[t][0][self.all_m_list[self.lim_index]][1])
                    if self.all_m_list[self.lim_index + 1] in list(sto[t][1].keys()):
                        ll_reduced = np.minimum(ll_reduced, sto[t][0][self.all_m_list[self.lim_index+1]][0])
                        ur_reduced = np.maximum(ur_reduced, sto[t][0][self.all_m_list[self.lim_index+1]][1])
        reduced_region = np.vstack((ll_reduced, ur_reduced))

        return reduced_region

    def extend_initial_refined_region(self, surrounded_box, numsegs = 10):
        """
        This function extends the initial refined region until encountering less-refined particles or reaching the boundary of the surrounded boxex.
        This is done by locating all the less-refined particles in the surrounded box.
        We start from the initial refined region, and keep expanding the box in all six directions a less-refined particle appears in the refined box.

        Parameters
        ----------
        center : np.array() (3)
            The center of mass of the most refined particles in the reduced region. This is the starting point for the extension process.

        surrounded_box : np.array() (2x3)
            The coordinates of the surrounded box. The first element is the lower-left corner, and the second element is the upper-right corner.
            This is the smallest box that contains all the particles at the very most refined level.
            This is the boundary that the refined region should not exceed, and the extension process stops when reaching this boundary.

        numsegs : int (default = 10)
            The number of segments that the surrounded_box will be divided into. This is chosen according to the number of processors used.

        Return
        -------
        output : np.array() (2x3)
            The coordinates of the refined region. The first element is the lower-left corner, and the second element is the upper-right corner.

        """
        ll_all = surrounded_box[0]
        ur_all = surrounded_box[1]
        xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                    np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))

        _,segdist = np.linspace(ll_all[0],ur_all[0],numsegs,retstep=True)

        ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
        ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
        ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
        ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
        #Obtain all the less refined particle positions in the surrounded_box
        my_storage = {}
        for sto, i in yt.parallel_objects(range(len(ll)), nprocs, storage = my_storage, dynamic = False):
            #buffer = (segdist/100) #set buffer when loading the box
            buffer = 0 #no buffer to make it consistent between different numsegs
            reg = self.ds.box(ll[i] - buffer ,ur[i] + buffer)
            dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter', 'AREPO': 'DarkMatter', 'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
            dm_pos = reg[dm_name_dict[self.codetp],'particle_position'].to('code_length').v
            dm_m = reg[dm_name_dict[self.codetp],'particle_mass'].to('Msun').v
            dm_m = np.ceil(dm_m).astype(np.int64)
            most_refined_pos_each = dm_pos[dm_m == self.all_m_list[self.lim_index]]
            most_refined_pos_sum = np.sum(most_refined_pos_each, axis=0)
            most_refined_pos_len = len(most_refined_pos_each)
            lr_pos_each = dm_pos[dm_m > (self.all_m_list[self.lim_index + 2]-10)]
            sto.result = {}
            sto.result[0] = lr_pos_each
            sto.result[1] = most_refined_pos_sum
            sto.result[2] = most_refined_pos_len
        #Get all the less-refined particle information
        lr_pos = np.empty((0,3))
        for c, vals in sorted(my_storage.items()):
            if vals != None:
                lr_pos = np.vstack((lr_pos, vals[0]))
        #Determine the center of the init_refined_region
        most_refined_pos_sum_all = np.array([0,0,0])
        most_refined_pos_len_all = 0
        for c, vals in sorted(my_storage.items()):
            if vals != None:
                most_refined_pos_sum_all = most_refined_pos_sum_all + vals[1]
                most_refined_pos_len_all += vals[2]

        center = most_refined_pos_sum_all/most_refined_pos_len_all
        self.center = center
        #Run the expansion
        my_storage = {}
        for sto, i in yt.parallel_objects(range(1), nprocs, storage = my_storage):
            #Compute the initial box to expand from. This initial box is a perfect cube.
            extend_width = (self.ds.domain_right_edge.to('code_length').v[0] - self.ds.domain_left_edge.to('code_length').v[0])/20
            init_refined_region = np.array([center - extend_width, center + extend_width])
            init_boolean = np.all((lr_pos > init_refined_region[0]) & (lr_pos < init_refined_region[1]), axis=1)
            while np.any(init_boolean) == True: #this means that there are less-refined particles in the box
                extend_width /= 2
                init_refined_region = np.array([center - extend_width, center + extend_width])
                init_boolean = np.all((lr_pos > init_refined_region[0]) & (lr_pos < init_refined_region[1]), axis=1)

            box = np.array([init_refined_region[0],init_refined_region[1]])
            # print('Expansion begins')
            spacing = extend_width/8
            spacing_lim = (self.ds.domain_right_edge.to('code_length').v[0] - self.ds.domain_left_edge.to('code_length').v[0])/10000
            # Define the six directions as 3D vectors
            directions = np.array([
                [[-spacing, 0, 0],[0,0,0]],  # min_x
                [[0, -spacing, 0],[0,0,0]],  # min_y
                [[0, 0, -spacing],[0,0,0]],  # min_z
                [[0,0,0],[spacing, 0, 0]],   # max_x
                [[0,0,0],[0, spacing, 0]],   # max_y
                [[0,0,0],[0, 0, spacing]]    # max_z
            ])
            # Initialize the stop flags
            stop_flags = np.array([False, False, False, False, False, False])
            # While not all directions are stopped
            while not np.all(stop_flags):
                # For each direction
                for i in range(6):
                    # If this direction is not stopped
                    if stop_flags[i] == False:
                        # Try to expand the box in this direction
                        new_box = box + directions[i]
                        #If the new box expand beyond the surrounded boundary (in a direction), set the value of the new box equal to the surrounded boundary (in that direction)
                        loc = int(i/3), i % 3 #loc is the location of the direction in the box (remember the box is set by np.array([minx, miny, minz], [maxx, maxy, maxz]))
                        if i < 3: #for the minimum (lower left) expansion
                            if new_box[loc[0]][loc[1]] < surrounded_box[loc[0]][loc[1]]:
                                new_box[loc[0]][loc[1]] = surrounded_box[loc[0]][loc[1]]
                                stop_flags[i] = True
                                # print('Min expasion prohibited in direction',i)
                        elif i >= 3: #for the maximum (upper right) expansion
                            if new_box[loc[0]][loc[1]] > surrounded_box[loc[0]][loc[1]]:
                                new_box[loc[0]][loc[1]] = surrounded_box[loc[0]][loc[1]]
                                stop_flags[i] = True
                                # print('Max expasion prohibited in direction',i)
                        # Check if there are any less-refined particles in the expanded region
                        boolean = np.all((lr_pos > new_box[0]) & (lr_pos < new_box[1]), axis=1)
                        # If there are no less-refined particles, update the box
                        if not np.any(boolean):
                            box = new_box
                        # Otherwise, stop this direction
                        else: #if encounter the less-refined particles, decrease the extension spacing by 1.5, then try again. However, if the spacing is less than the limit, stop the direction
                            while abs(directions[i].sum()) >= spacing_lim or np.any(boolean) == True:
                                directions[i] /= 1.5
                                new_box = box + directions[i]
                                boolean = np.all((lr_pos > new_box[0]) & (lr_pos < new_box[1]), axis=1)
                            if abs(directions[i].sum()) >= spacing_lim:
                                box = new_box
                            else:
                                stop_flags[i] = True
                                # print('Stop because less-refined particles appear in direction',i)
            sto.result = {}
            sto.result[0] = box

        output = []
        for c, vals in sorted(my_storage.items()):
            if vals != None:
                output = vals[0]

        self.refined_region = output
        return output

    def find_refined_region(self):
        """
        The main function that combines all the previous functions.
        """
        start_time = time.time()
        if nprocs < 128:
            numsegs = 10
        else:
            numsegs = int(np.ceil((nprocs*5)**(1/3)) + 1)
        if self.enzo_type4_only:
            self.refined_region = self._reduce_range_enzo_type4(numsegs)
            del self.ds
            if yt.is_root():
                np.save(self.savestring + '/' + 'refined_region_%s.npy' % (self.timestep),self.refined_region)
            return
        reduced_region = self.reduce_range(numsegs)
        # if yt.is_root():
        #     #print(reduced_region)
        #     #print(center)
        #     print('All mass levels:', self.all_m_list)
        #     print('Time for reduce_range: ', time.time() - start_time)
        self.extend_initial_refined_region(reduced_region, numsegs = numsegs)
        del self.ds
        if yt.is_root():
            # print('All mass levels:', self.all_m_list)
            # print('Reduced region:',reduced_region)
            # print('Center:',self.center)
            # print('Refined region:',self.refined_region)
            # print('Time for extend_initial_region: ', time.time() - start_time)
            np.save(self.savestring + '/' + 'refined_region_%s.npy' % (self.timestep),self.refined_region)
