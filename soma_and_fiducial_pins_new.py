from sqlalchemy import create_engine
import pandas as pd
import os
import datetime
import SimpleITK as sitk
import psycopg2
from morph_utils import query 

# CONNECTION_STRING = 'host=limsdb2 dbname=lims2 user=limsreader password=limsro'

# def query(sql, args):
#     conn = psycopg2.connect(CONNECTION_STRING)
#     cur = conn.cursor()

#     cur.execute(sql, args)
#     results = cur.fetchall()

#     cur.close()
#     conn.close()

#     return results

def process_json( slide_specimen_id, jblob, annotation, structures ) :
    
    locs = []
     
    for m in jblob['markups'] :

        info = {}
        info['slide_specimen_id'] = slide_specimen_id
        info['specimen_name'] = m['name']
        
        #print(m['name'])

        if m['markup']['type'] != 'Fiducial' :
            continue
            
        if 'controlPoints' not in m['markup'] :
            print(info)
            print("WARNING: no control point found, skipping")
            continue
            
        if m['markup']['controlPoints'] == None :
            print(info)
            print("WARNING: control point list empty, skipping")
            continue
            
        if len(m['markup']['controlPoints']) > 1 :
            print(info)
            print("WARNING: more than one control point, using the first")

        #
        # Cell Locator is LPS(RAI) while CCF is PIR(ASL)
        #
        pos = m['markup']['controlPoints'][0]['position']
        info['x'] =  1.0 * pos[1]
        info['y'] = -1.0 * pos[2]
        info['z'] = -1.0 * pos[0]
        
        if (info['x'] < 0 or info['x'] > 13190) or \
            (info['y'] < 0 or info['y'] > 7990) or \
            (info['z'] < 0 or info['z'] > 11390) :
            print(info)
            print("WARNING: ccf coordinates out of bounds")
            continue
        
        # Read structure ID from CCF
        point = (info['x'], info['y'], info['z'])
        
        # -- this simply divides cooordinates by resolution/spacing to get the pixel index
        pixel = annotation.TransformPhysicalPointToIndex(point)
        sid = annotation.GetPixel(pixel)
        info['structure_id'] = sid
        
        if sid not in structures.index :
            print(info)
            print("WARNING: not a valid structure - skipping")
            continue
        
        info['structure_acronym'] = structures.loc[sid]['acronym']

        locs.append(info)

    return locs


def get_soma_and_fiducial_pins(output_folder = r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\ccf_slice_registration\cell_soma_structure_and_coords'):

    '''
    Enter your input parameters here, then run the entire notebook
    '''

    #Where you want the file saved
    # output_folder = r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\lims_queries\cell_soma_structure_and_coords'

    ##If you only wqant specific ids in the output file. If you set 'specific_ids' to False, this will be ignored
    id_csv = r'' #place location insise quotation marks
    specific_ids = False #Set this to True if you have specific ids


    # ---------------------------
    # (1) Get structure information from LIMS - this is only needed for validataion
    # (2) Open up CCF annotation volume
    # (3) Get json blob of the cells the be matched
    # (4) For each cell, convert Cell Locator to CCF coordinates and find annotation
    # (5) Close LIMS connection
    # (6) Write output to file
    # ----------------------------

    # ---------------------------
    # (1) Get structure information from LIMS - this is only needed for validataion
    # ----------------------------

    # conn = pg8000.connect(user='atlasreader', host='limsdb2', port=5432, database='lims2', password='atlasro')
    # db_str = "postgresql+psycopg2://atlasreader:atlasro@limsdb2:5432/lims2"
    # engine = sqlalchemy.create_engine(db_str)
    # # engine = create_engine(db_str)

    # structures = pd.read_sql(
    # "SELECT * FROM structures where ontology_id = 1 ",
    # con = engine)

    # sql = """SELECT * FROM structures where ontology_id = 1"""
    # results = 

    

    # structures.set_index('id', inplace=True)


    #get structure info from lims
    structures = pd.DataFrame(query.get_structures())
    structures.set_index('id', inplace=True)


    # --------------------------------
    # (2) Open up CCF annotation volume
    # ------------------------------

    model_directory = r'\\allen\programs\celltypes\production\0378\informatics\model_september_2017\P56\atlases\MouseCCF2017'
    annotation_file = os.path.join( model_directory, 'annotation_10.nrrd' )

    annotation = sitk.ReadImage( annotation_file )

    # ---------------------------
    # (3) Get json blob of the cells the be matched
    # ----------------------------

    # pins = pd.read_sql(
    #     #ephys pin location
    # #"SELECT sm.* FROM specimen_metadata sm WHERE sm.current = 't' AND sm.kind = 'IVSCC cell locations'",
    #     #QC'ed location 
    #     "SELECT sm.* FROM specimen_metadata sm WHERE sm.current = 't' AND sm.kind = 'IVSCC tissue review'",
    #     con = engine
    # )
    pins = pd.DataFrame(query.query_pinning_info_cell_locator())

    cell_info = []


    # ---------------------------
    # (4) For each cell, convert Cell Locator to CCF coordinates and find annotation
    # ---------------------------

    for index, row in pins.iterrows() :    
        
        jblob = row['data']
        #print("== input: %s" % row['specimen_id'])
        #print(jblob)
        
        processed = process_json( row['specimen_id'], jblob, annotation, structures )
        #print("== output: ")
        #print(processed)
        
        cell_info.extend(processed)

    # ---------------------------
    # (5) Close LIMS connection
    # ----------------------------

    # engine.dispsose()

    # ---------------------------
    # (6) Write output to file
    # ----------------------------

    df = pd.DataFrame(cell_info)
    x = datetime.datetime.now()
    fdt = str(x).split(' ')[0]
    y, m, d = fdt.split('-')
    dt = '{}{}{}'.format(y, m, d)

    output_file = "cell_soma_information_{}.csv".format(dt)

    df.to_csv(os.path.join(output_folder, output_file), index=False)

    if specific_ids:
        id_dat = pd.read_csv(id_csv)
        name_dat = pd.DataFrame()
        for sp_id in id_dat.cell_specimen_id.values:

            sql = """
            SELECT sp.name as sp, sp.id
            FROM specimens sp
            WHERE sp.id = '{}'
            """.format(sp_id)
            curr = pd.read_sql(
                sql,
                con = engine
            )
            name_dat = pd.concat([name_dat, curr])

        print('Missing: {}'.format([i for i in name_dat.sp.values if i not in df.specimen_name.values]))
        df = df[df.specimen_name.isin(name_dat.sp.values)]
        df.to_csv(os.path.join(output_folder, output_file), index=False)


    return os.path.join(output_folder, output_file) #path to file 