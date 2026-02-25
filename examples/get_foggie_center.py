import pandas as pd
import numpy as np

def get_foggie_center(
    haloname,
    snapname, #DD1234 
    ):
    
    halo_dict = {   '002392'  :  'Hurricane' ,
                    '002878'  :  'Cyclone' ,
                    '004123'  :  'Blizzard' ,
                    '005016'  :  'Squall' ,
                    '005036'  :  'Maelstrom' ,
                    '008508'  :  'Tempest' }
    halo_dict_rev = {v: k for k, v in halo_dict.items()}

    path_df = f"/mnt/home/mjung/foggie/foggie/halo_infos/{halo_dict_rev[haloname]}/nref11c_nref9f/halo_c_v"
    df = pd.read_csv(path_df, sep="|", engine="python", skipinitialspace=True)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    df["name"] = df["name"].str.strip()

    for col in df.columns:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    L = 143884.89208633  # kpccm
    #snapname = f"DD{int(snapnum):04d}"
    row = df.loc[df["name"] == snapname]
    if row.empty:
        raise ValueError(f"{snapname} not found in metadata.")
    row = row.iloc[0]

    center = np.array(row[['x_c', 'y_c', 'z_c']])
    redshift = row['redshift']
    center_unitary = (center * (1 + redshift)) / L    
    return center_unitary

