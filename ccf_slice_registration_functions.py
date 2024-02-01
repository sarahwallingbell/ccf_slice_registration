import os 
import glob
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import psycopg2
import matplotlib.pyplot as plt
from numpy.linalg import inv
import requests
import shutil
import csv
import re
import lims_utils
import allensdk.core.swc as swc
import logging

AUTOTRACE_PATH = r'\\allen\programs\celltypes\workgroups\mousecelltypes\AutotraceReconstruction'

CONNECTION_STRING = 'host=limsdb2 dbname=lims2 user=limsreader password=limsro'
IMAGE_SERVICE_STRING = 'http://lims2/cgi-bin/imageservice'


def query(sql, args):
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    cur.execute(sql, args)
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

def visualize_landmarks( image_2d, ax, landmarks = []) :
    
    # visualize a 2D image with landmarks superimposed

    left = (-0.5) * image_2d.GetSpacing()[0]
    right = (image_2d.GetSize()[0] - 0.5) * image_2d.GetSpacing()[0]
    bottom = (image_2d.GetSize()[1] - 0.5) * image_2d.GetSpacing()[1]
    top =  (- 0.5) * image_2d.GetSpacing()[1]

    slice = sitk.GetArrayViewFromImage(image_2d)
    ax.imshow(slice, extent = (left,right,bottom,top))
    
    for p in landmarks :
        ax.plot(p[0],p[1],'r+',markersize=10)


def flatten( lm ) :    
    # flatten a set of points as input to SimpleITK landmark based registration

    return [ c for p in lm for c in p]

def initialize_transform( transform_info ) :
    
    # initialize a SimpltITK 3D affine transform from json blob
    arr = np.array( transform_info )
    lut = [0, 1, 2, 4, 5, 6, 8, 9, 10, 3, 7, 11]
    parameters = arr[lut]

    m = np.reshape(parameters[0:9],(3,3))
    t = np.reshape(parameters[9:12],(3,1))
    
    # flip from RAS to LPS  
    # because pinning tool is in RAS but ITK does transforms in LPS
    x = [-1,0,0,0,-1,0,0,0,1]
    x = np.reshape(x,(3,3))
    m = np.matmul(x,m)
    t = np.matmul(x,t)
    
    # flip both axes of the virtual view and move to center
    vm = np.reshape([1,0,0,0,-1,0,0,0,1],(3,3)) #TODO only flipping one axis... why
    vt = np.reshape([-6250,-6250,0],(3,1))
    m = np.matmul(m,vm)
    t = np.add(np.matmul(m,vt),t) 

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(m.flat[:])
    affine.SetTranslation(t.flat[:])
    
    return affine


def get_ccf_coord(specimen_id):
    sql = """
        select x, y, z from cell_soma_locations
        where specimen_id = {}
        """.format(specimen_id)
    x,y,z = query(sql,())[0]
    return x,y,z


from pandas import DataFrame, Series

def query_lims_for_layers(specimen_id):
    sql = """
    SELECT sp.id as specimen_id, sp.name AS specimen, sp.cell_depth,
        imt.name AS image_type, agl.name AS drawing_layer, polygon.id AS polygon_id,
        bp.biospecimen_id,
        polygon.path, layer.mag, polygon.display_attributes, sc.resolution, struct.acronym
    FROM specimens sp JOIN specimens spp ON spp.id=sp.parent_id
    JOIN image_series iser ON iser.specimen_id=spp.id AND iser.type = 'FocalPlaneImageSeries' AND iser.is_stack = 'f'
    JOIN sub_images si ON si.image_series_id=iser.id
    JOIN avg_graphic_objects layer ON layer.sub_image_id=si.id
    JOIN avg_graphic_objects polygon ON polygon.parent_id=layer.id
    LEFT JOIN biospecimen_polygons bp ON polygon.id = bp.polygon_id
    JOIN images im ON im.id=si.image_id
    JOIN image_types imt ON imt.id=im.image_type_id
    JOIN scans sc ON sc.slide_id=im.slide_id
    LEFT JOIN structures struct ON struct.id = polygon.cortex_layer_id
    JOIN avg_group_labels agl ON layer.group_label_id=agl.id
    WHERE sp.id = %s
    ORDER BY 1, 4, 5, 6
    """

    results = query(sql, (specimen_id, ))
    df = DataFrame(results, columns=["specimen_id", "specimen_name", "cell_depth",
                                         "img_type", "draw_type", "poly_id", "biospecimen_id", "poly_coords",
                                         "mag", "dispattr", "res", "layer_acronym"]).drop_duplicates(subset="poly_coords")

#     keep only draw types we are interested in
    used_draw_types = ["Pia", "White Matter", "Soma", "Cortical Layers"]
    df = df.loc[df["draw_type"].isin(used_draw_types), :].drop_duplicates(subset=["biospecimen_id", "poly_coords"])
    
#     keep soma, pia, and wm for specimen only
    mask_out = df["draw_type"].isin(["Pia", "White Matter", "Soma"]).values & (df["biospecimen_id"].values != int(specimen_id))
    df = df.loc[~mask_out, :]

    return df

def get_20x_info(sp_name):
    sql = """
    SELECT  slice.name, subimg.id, subimg.width, subimg.height, sc.resolution,  img.treatment_id
    FROM specimens slice 
    JOIN image_series iser on iser.specimen_id = slice.id -- image series is associated with the slice
    JOIN sub_images subimg on subimg.image_series_id = iser.id -- there are two subimages in the imageseries
    JOIN images img on subimg.image_id = img.id -- each subimage belongs to an image
    JOIN scans sc on sc.image_id = img.id -- an image is output of a scan
    WHERE slice.name = '{}' AND img.treatment_id = 300080909
    """.format(sp_name)

    x = query(sql, ())[0]
    return x
    

def get_20x_img(sub_image, specimen_name, working_directory):
    sql ="""
    SELECT sd.storage_directory, img.zoom FROM slides sd
    JOIN images img ON img.slide_id = sd.id
    JOIN sub_images si ON si.image_id = img.id 
    WHERE si.id = '{}'
    """.format(sub_image)
    
    result = query(sql, ())[0]
    sd = result[0]
    aff = result[1]
    aff_path = os.path.join(sd, aff)

    image_path = os.path.join(working_directory, '{}_overview.jpg'.format(specimen_name))
    downsample = 0
    quality = 100
    url = r'{0}?path={1}&'\
                   'downsample={2}&'\
                   'quality={3}'.format(IMAGE_SERVICE_STRING, 
                                        aff_path, downsample, quality)
    # print('result: ', result)
    # print('sd: ', sd)
    # print('aff: ', aff)
    # print('aff_path: ', aff_path)
    # print('url: ', url)



    response = requests.get(url, stream=True)

    with open(image_path, 'wb') as image_file:
        shutil.copyfileobj(response.raw, image_file)
    del response
    
    
def get_name_by_id(sp_id):
    sql = """
    SELECT sp.name as sp, sp.id
    FROM specimens sp
    WHERE sp.id = '{}'
    """.format(sp_id)
    
    x = query(sql, ())[0][0]
    return x

def get_id_by_name(sp_name):
    sql = """
    SELECT sp.name as sp, sp.id
    FROM specimens sp
    WHERE sp.name = '{}'
    """.format(sp_name)
    
    x = query(sql, ())[0][1]
    return x

def get_pinning_old(sp_name): #how it was in this code. 
    sql = """
    SELECT sp.name AS sp, sm.id, sm.specimen_id, sm.updated_at ,sm.data
    FROM
    specimens sp
    JOIN specimen_metadata sm ON sm.specimen_id=sp.id
    WHERE
    sp.name = '{}' AND
    current = 't' AND kind = 'IVSCC cell locations'
    ORDER BY sm.id DESC;""".format(sp_name)
    
    x = query(sql, ())[0]
    return x

#kind = 'IVSCC cell locations' : the ephys rig person did a rough pinning of location 
#kind = 'IVSCC tissue review' : someone on our team reviewed the rough pinning and validated it. <USE THIS ONE ONLY> in code!

def get_pinning(sp_name): #how it should be in this code (SWB)
    sql = """
    SELECT sp.name AS sp, sm.id, sm.specimen_id, sm.updated_at ,sm.data
    FROM
    specimens sp
    JOIN specimen_metadata sm ON sm.specimen_id=sp.id
    WHERE
    sp.name = '{}' AND
    sm.current = 't' AND sm.kind = 'IVSCC tissue review' 
    ORDER BY sm.id DESC;""".format(sp_name)

    #   'IVSCC cell locations'
    
    x = query(sql, ())[0]
    return x

def get_children(spec_id):
    sql = """
        SELECT sp.id as specimen_id, sp.name AS specimen
        FROM specimens sp JOIN specimens spp ON spp.id=sp.parent_id
        WHERE sp.parent_id = {}
        """.format(spec_id)

    results = query(sql, ())
    children = []
    for r in results:
        if len(r[1].split('.')) > 3:
            children.append(list(r))
            
            
    return children

def get_soma_polygons(spec_id):

    children = get_children(spec_id)
    df = pd.DataFrame()
    for c in children:
        spec = c[0]
        print('spec: {}'.format(spec))
        this_cell = query_lims_for_layers(str(spec))
        df = pd.concat([df, this_cell])

    return df
    
def compute_center_from_polyline (aa) :
    aa = aa.split(',')
    aa = [int(x) for x in aa]
    aa = np.reshape(aa,(int(len(aa)/2),2))
    return np.mean( aa , axis = 0)

def get_landmark_ids(spec_id):
    sql = """
        SELECT sp.id as specimen_id, sp.name AS specimen
        FROM specimens sp JOIN specimens spp ON spp.id=sp.parent_id
        WHERE sp.parent_id = {}
        """.format(spec_id)

    results = query(sql, ())
    children = []
    for r in results:
        if len(r[1].split('.')) == 3:
            children.append(list(r))
    
    return children

def get_landmark_location(spec_id):
    sql = """
        SELECT sp.id as specimen_id, sp.name AS specimen, sp.cell_depth,
            imt.name AS image_type, agl.name AS drawing_layer, polygon.id AS polygon_id,
            bp.biospecimen_id,
            polygon.path, layer.mag, polygon.display_attributes, sc.resolution, struct.acronym
        FROM specimens sp JOIN specimens spp ON spp.id=sp.parent_id
        JOIN image_series iser ON iser.specimen_id=spp.id AND iser.type = 'FocalPlaneImageSeries' AND iser.is_stack = 'f'
        JOIN sub_images si ON si.image_series_id=iser.id
        JOIN avg_graphic_objects layer ON layer.sub_image_id=si.id
        JOIN avg_graphic_objects polygon ON polygon.parent_id=layer.id
        LEFT JOIN biospecimen_polygons bp ON polygon.id = bp.polygon_id
        JOIN images im ON im.id=si.image_id
        JOIN image_types imt ON imt.id=im.image_type_id
        JOIN scans sc ON sc.slide_id=im.slide_id
        LEFT JOIN structures struct ON struct.id = polygon.cortex_layer_id
        JOIN avg_group_labels agl ON layer.group_label_id=agl.id
        WHERE sp.id = '%s' AND agl.name = 'Fiducial' AND bp.biospecimen_id = %s
        """

    results = query(sql, (spec_id, spec_id))

    d = pd.DataFrame(results, columns = ['specimen_id', 'specimen_name', 'cell_depth', 'img_type', 'draw_type',
       'poly_id', 'biospecimen_id', 'poly_coords', 'mag', 'dispattr', 'res','layer_acronym'])
    d.drop_duplicates(subset = 'poly_coords', inplace = True)
    return d
            

# == transform cell morphs to ccf ==





def to_dict(swc_file):
    nodes = {}
    with open(swc_file, "r") as f:
        for line in f:
            if line.lstrip().startswith('#'):
                continue
            toks = line.split()
            node_dict = {
                'id' : int(toks[0]),
                'type' : int(toks[1]),
                'x' : float(toks[2]),
                'y' : float(toks[3]),
                'z' : float(toks[4]),
                'radius' : float(toks[5]),
                'parent' : int(toks[6].rstrip())
            }
            nodes[int(toks[0])] = node_dict
    return nodes




def shift(x, y, z, morpho):
	for node in morpho.keys():
		morpho[node]['x'] += x
		morpho[node]['y'] += y
		morpho[node]['z'] += z
		
	return morpho
    

def read_marker_file(file_name):
    """ read in a marker file and return a list of dictionaries """
    markers = []
    with open(file_name, "r") as f:
        for line in f:
            if line.lstrip().startswith('#'):
                continue
            toks = line.split(',')
#             print(toks)
            marker_dict = {

                'x' : float(toks[0]),
                'y' : float(toks[1]),
                'z' : float(toks[2]),
                'name' : int(toks[5])
            }
            markers.append(marker_dict)
    return markers


def get_swc_from_lims(specimen_id):
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    SQL = "SELECT f.filename, f.storage_directory FROM \
     neuron_reconstructions n JOIN well_known_files f ON n.id = f.attachable_id \
     AND n.specimen_id = %s AND n.manual AND NOT n.superseded AND f.well_known_file_type_id = 303941301"
    cur.execute(SQL, (specimen_id,))
    result = cur.fetchone()

    if result is None:
        # print("\t\tNo SWC file found for specimen ID {}".format(specimen_id))
        return

    swc_filename = result[0]
    swc_path = result[1] + result[0]

    cur.close()
    conn.close()
    return swc_filename, swc_path


def get_marker_file_from_lims(specimen_id):
    SQL = "SELECT f.filename, f.storage_directory FROM \
     neuron_reconstructions n JOIN well_known_files f ON n.id = f.attachable_id \
     AND n.specimen_id = %s AND n.manual AND NOT n.superseded AND f.well_known_file_type_id = 486753749"
    
    try: 
        result = query(SQL, (specimen_id,))[0]
        marker_path = result[1] + result[0]
        return marker_path
    except:
        #no marker file found for this cell
        return None

def identify_soma_marker(markers):

    try: 
        soma_markers = [m for m in markers if m["name"] == 30] # 30 is the code for soma marker
        soma_marker = soma_markers[0]
        return soma_marker
    except: 
        #no soma markers found 
        return None


def scale_factor(specimen_id, morph):
    cut_thickness=350
 
    sql = f"""
    select sp.id, sp.cell_depth from specimens sp
    where sp.id = {specimen_id}
    """
    result = query(sql, specimen_id)[0]
    cell_depth = list(result)[1]#['cell_depth']

    marker_path = get_marker_file_from_lims(str(specimen_id))

    if marker_path: 
        marker_path = edit_path(marker_path)
        markers = read_marker_file(marker_path)
    else: markers = []

    soma = morph[1]

    soma_marker = identify_soma_marker(markers)

    if (soma_marker is not None) and (cell_depth is not None):
        z_level = soma_marker["z"]
        fixed_depth = np.abs(soma["z"] - z_level)

        if np.allclose(fixed_depth, 0):
            return np.nan

        scale = cell_depth / fixed_depth
        all_z = [c["z"] for c in morph.values()]
        max_z_extent = np.max(all_z) - np.min(all_z)
        min_slice_thickness = max_z_extent * scale

        if min_slice_thickness <= cut_thickness:
            corrected_scale = scale
        else:
            corrected_scale = cut_thickness / max_z_extent
    else:
        all_z = [c["z"] for c in morph.values()]
        max_z_extent = np.max(all_z) - np.min(all_z)
        corrected_scale = cut_thickness / max_z_extent
    return corrected_scale

def edit_path(p):
    p = p.replace('\\', '/')
    p = p.replace('/', '//', 1)
    return p


def convert_coords_str(coords_str):
    vals = coords_str.split(',')
    x = np.array(vals[0::2], dtype=float)
    y = np.array(vals[1::2], dtype=float)
    return x, y



def dict_to_swc(neuron_dict, filename):
	"""
	Takes a neuron dictionary and converts to a swc file
	"""
	f = open(filename, 'w')
	f.write("# id,type,x,y,z,r,pid\n")
	for l, vals in neuron_dict.items():
		f.write("%d %d " % (vals['id'] , vals['type']))
		f.write("%.4f " % vals['x'])
		f.write("%.4f " % vals['y'])
		f.write("%.4f " % vals['z'])
		f.write("%.4f " % vals['radius'])
		f.write("%d\n" % vals['parent'])
	f.close()
        


#get soma loc in ccf from json, not from lims. 
#This should work for cells pinned with new pinning tool.
def get_ccf_coord_jblob(name, cell_soma_info):
    row = cell_soma_info[cell_soma_info.specimen_name == name].iloc[0]
    return row.x, row.y, row.z


def slice_flipped(ldf):
    # determins if the order of fiducial pins is the same (along x axis) between 20x overview and virtual slice
    # if they are the same, no flip (False). If they are not the same, flip (True). 
    for i, (idx, row) in enumerate(ldf.iterrows()):
        if i == 0:
            #overiew slice first pin 
            min_pin_overview = row.specimen_name
            min_pin_overview_val = row.overview_coordinate[0]
            #virtual slice first pin 
            min_pin_virtual_slice = row.specimen_name
            min_pin_virtual_slice_val = row.virtual_slice_coordinate[0]
        else:
            #get overciew slice min x pin 
            if row.overview_coordinate[0] < min_pin_overview_val:
                min_pin_overview = row.specimen_name
                min_pin_overview_val = row.overview_coordinate[0]
            #get virtual slice min x pin 
            if row.virtual_slice_coordinate[0] < min_pin_virtual_slice_val:
                min_pin_virtual_slice = row.specimen_name
                min_pin_virtual_slice_val = row.virtual_slice_coordinate[0]

    if min_pin_overview == min_pin_virtual_slice: flip = False #no flip
    else: flip = True #flip

    # print('min pin overview: ', min_pin_overview_val)
    # print('min pin overview: ', min_pin_overview)
    # print('min pin virtual: ', min_pin_virtual_slice_val)
    # print('min pin virtual: ', min_pin_virtual_slice)
    # print('flip: ', flip)

    return flip

def query_for_z_resolution(specimen_id):
    sql = """
    select ss.id, ss.name, shs.thickness from specimens ss
    join specimen_blocks sb on ss.id = sb.specimen_id
    join blocks bs on bs.id = sb.block_id
    join thicknesses shs on shs.id = bs.thickness_id 
    where ss.id = {}
    """.format(specimen_id)
    
    res = lims_utils.query(sql,())
    try:
        return res[0][-1]
    except:
        return None
    
def convert_pixel_to_um(morph, specimen_id):
    anisotropy_value = query_for_z_resolution(specimen_id)
    # print('anisotropy_value = {}'.format(anisotropy_value))
    for no in morph.nodes():
        no['x'] = no['x']*0.1144
        no['y'] = no['y']*0.1144
        no['z'] = no['z']*anisotropy_value
    return morph

def convert_pixel_to_um_dictnrn(morph, specimen_id):
    anisotropy_value = query_for_z_resolution(specimen_id)
    # print('anisotropy_value = {}'.format(anisotropy_value))
    for k, v in morph.items():
        morph[k]['x'] = v['x']*0.1144
        morph[k]['y'] = v['y']*0.1144
        morph[k]['z'] = v['z']*anisotropy_value
    return morph

def get_autotrace_pp_path(spec, step):
    cell_path = os.path.join(AUTOTRACE_PATH, str(spec))
    swcs_path = os.path.join(cell_path, 'SWC')
    postproc_path = os.path.join(swcs_path, 'PostProcess')
    swc_path = os.path.join(postproc_path, '{}_Aspiny1.0_C.0.step{}_SortTreeIDs.swc'.format(spec, step))
    return swc_path

def get_autotrace_raw_path(spec): #TODO rigid way of getting raw file, doesn't account for diff model nums etc. 
    cell_path = os.path.join(AUTOTRACE_PATH, str(spec))
    swcs_path = os.path.join(cell_path, 'SWC')
    postproc_path = os.path.join(swcs_path, 'Raw')
    swc_path = os.path.join(postproc_path, '{}_Aspiny1.0_0.1.2_1.0.swc'.format(spec))
    return swc_path

def register_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, cell_soma_info, overview_to_virtual_slice_transform, virtual_slice_to_ccf_transform):

    # 1) save original swc to path 
    print('\t\tStarting step 1')
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'.swc'))
    soma = morph[1]

    # 3) if needed correct for shrinkage - this should only affect the z coordinate!
    try: 
        print('\t\tStarting step 3')
        z_scale = scale_factor(sp_id, morph)
        for k,v in morph.items():
            cz = v['z']
            morph[k]['z'] = cz * z_scale
    except: print('\t\t\tcould not correct for shrinkage') #TODO currently shrinkage correction relies on there being a manual reconstruction. 

    # 4) translate the (x,y) coorinates of the morphology such that the soma node is in 
    #    the corresponding position in the overview image
    print('\t\tStarting step 4')
    x_shift = lims_soma[0] - soma['x']
    y_shift = lims_soma[1] - soma['y']
    morph = shift( x_shift, y_shift, 0, morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_shifted_soma.swc'))

    # 5) transform the morphology to match the virtual slice
    # ---- apply "overview_to_virtual_slice_transform" to the (x,y) coordinates of the morphology
    print('\t\tStarting step 5')
    for k,v in morph.items():
        point = [v['x'], v['y']]
        tpoint = list(overview_to_virtual_slice_transform.TransformPoint(point))
        morph[k]['x'] =  tpoint[0]
        morph[k]['y'] =  tpoint[1]
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_to_virtual.swc'))

    # 6) transform from virtual slice to CCF space
    # ---- apply "virtual_slice_to_ccf_transform" to the (x,y,z) coordinates of the morphology
    print('\t\tStarting step 6')
    for k,v in morph.items():
        point = [v['x'], v['y'], v['z']]
        tpoint = list(virtual_slice_to_ccf_transform.TransformPoint(point))
        morph[k]['x'] =  tpoint[0]
        morph[k]['y'] =  tpoint[1]
        morph[k]['z'] =  tpoint[2]
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reoriented.swc'))

    # 7) transform data to PIR coords (what ccf is in) from LPS (what simpleITK is in)
    print('\t\tStarting step 7')
    x = [0,1,0,
         0,0,-1,
        -1,0,0]
    x = np.reshape(x,(3,3))
    for k,v in morph.items():
        point = [v['x'], v['y'], v['z']]
        point = np.reshape(point, (3,1))
        pir_point = np.matmul(x,point)
        morph[k]['x'] =  pir_point[0]
        morph[k]['y'] =  pir_point[1]
        morph[k]['z'] =  pir_point[2]
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reoriented_pir.swc'))

    
    # 8) shift to provided ccf soma coordinate
    print('\t\tStarting step 8')
    soma = morph[1]
    ccf_soma = list(get_ccf_coord_jblob(sp_name, cell_soma_info))
    x_shift = ccf_soma[0] - soma['x']
    y_shift = ccf_soma[1] - soma['y']
    z_shift = ccf_soma[2] - soma['z']
    morph = shift( x_shift, y_shift, z_shift, morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reoriented_pir_shifted.swc'))


    print('\t\tRegistration complete!')
    