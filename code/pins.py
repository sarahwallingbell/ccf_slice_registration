import os
import pandas as pd
import datetime
import SimpleITK as sitk
import psycopg2
from sqlalchemy import create_engine
from morph_utils import query 

def process_json( slide_specimen_id, jblob, annotation, structures ) :
    
    locs = []
     
    for m in jblob['markups'] :

        info = {}
        info['slide_specimen_id'] = slide_specimen_id
        info['specimen_name'] = m['name']
        
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

def get_soma_and_fiducial_pins(output_folder, annotation_file, query_engine=None):

    # ---------------------------
    # (1) Get structure information from LIMS - this is only needed for validataion
    # (2) Open up CCF annotation volume
    # (3) Get json blob of the cells the be matched
    # (4) For each cell, convert Cell Locator to CCF coordinates and find annotation
    # (5) Write output to file
    # ----------------------------

    # ---------------------------
    # (1) Get structure information from LIMS - this is only needed for validataion
    # ----------------------------

    #get structure info from lims
    structures = pd.DataFrame(query.get_structures(query_engine=query_engine))
    structures.set_index('id', inplace=True)

    # --------------------------------
    # (2) Open up CCF annotation volume
    # ------------------------------

    # model_directory = r'\\allen\programs\celltypes\production\0378\informatics\model_september_2017\P56\atlases\MouseCCF2017'
    # annotation_file = os.path.join( model_directory, 'annotation_10.nrrd' )

    annotation = sitk.ReadImage( annotation_file )

    # ---------------------------
    # (3) Get json blob of the cells the be matched
    # ----------------------------

    pins = pd.DataFrame(query.query_pinning_info_cell_locator(query_engine=query_engine))

    # ---------------------------
    # (4) For each cell, convert Cell Locator to CCF coordinates and find annotation
    # ---------------------------

    cell_info = []
    for index, row in pins.iterrows() :    
        
        jblob = row['data']
        processed = process_json( row['specimen_id'], jblob, annotation, structures )
        cell_info.extend(processed)

    # ---------------------------
    # (5) Write output to file
    # ----------------------------

    df = pd.DataFrame(cell_info)
    output_file = "soma_pins.csv"
    df.to_csv(os.path.join(output_folder, output_file), index=False)

    return os.path.join(output_folder, output_file) #path to file 

def get_pins(output_folder, 
            annotation_file,
            query_engine):

    """
    Extract soma and fiducial pins.
    """

    pins_path = get_soma_and_fiducial_pins(output_folder, annotation_file, query_engine)
    pins = pd.read_csv(pins_path)

    pins['specimen_name'] = pins['specimen_name'].str.strip() #strip any erroneous white space from the start and end fo specimen names
    pins = pins[pins['specimen_name'] != 'Point'] #remove one random pin on a cortical slice. 

    #pins contains fiducial pins (specimen name ends in a letter) and soma pins (specimen name ends in a number)
    #break up pins into fiducial vs soma pin dataframes
    fiducials_dict = {}
    somas_dict = {}
    for i, p in pins.iterrows():
        last_pin_char = p['specimen_name'][-1]
        if last_pin_char.isalpha(): fiducials_dict[i] = p #the last char of the pin name is a letter, so this is a fiducial (not a soma) pin 
        else: somas_dict[i] = p #the last char of the pin name is a number, so this is a soma (not a fiducial) pin
            
    somas = pd.DataFrame.from_dict(somas_dict, orient='index').reset_index().drop(['index'], axis=1)
    fiducials = pd.DataFrame.from_dict(fiducials_dict, orient='index').reset_index().drop(['index'], axis=1)

    slices = fiducials.slide_specimen_id.unique() #get the slices that have fiducials

    return somas, slices