import os 
# import glob
# import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import psycopg2
# import matplotlib.pyplot as plt
# from numpy.linalg import inv
import requests
import shutil
import copy
# import csv
# import re
# import allensdk.core.swc as swc
# import logging
from neuron_morphology.morphology import Morphology
from neuron_morphology.swc_io import morphology_to_swc
from neuron_morphology.transforms.affine_transform import AffineTransform
from morph_utils.query import query_for_z_resolution
from morph_utils.ccf import move_soma_to_left_hemisphere
from morph_utils.modifications import resample_morphology, normalize_position


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

    # keep only draw types we are interested in
    used_draw_types = ["Pia", "White Matter", "Soma", "Cortical Layers"]
    df = df.loc[df["draw_type"].isin(used_draw_types), :].drop_duplicates(subset=["biospecimen_id", "poly_coords"])
    
    # keep soma, pia, and wm for specimen only
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

def get_pinning(sp_name):

    """
    Get pins from LIMS

    A note about specimen_metadata.kind: 
        sm.kind = 'IVSCC cell locations' : the ephys rig person did a rough pinning of location --> Don't use/trust this version. 
        sm.kind = 'IVSCC tissue review' : someone on our team reviewed the rough pinning and validated it. --> Only use/trust this version.
    
    """

    sql = """
    SELECT sp.name AS sp, sm.id, sm.specimen_id, sm.updated_at ,sm.data
    FROM
    specimens sp
    JOIN specimen_metadata sm ON sm.specimen_id=sp.id
    WHERE
    sp.name = '{}' AND
    sm.current = 't' AND sm.kind = 'IVSCC tissue review' 
    ORDER BY sm.id DESC;""".format(sp_name)
    
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

def dict_to_morphology(morph_dict):
    """ 
    Given a dictionary of morphology nodes (to_dict()) returns a neuron_morphology object 
    """

    nodes = list(morph_dict.values())
    for node in nodes:
        node["parent"] = int(node["parent"])
        node["id"] = int(node["id"])
        node["type"] = int(node["type"])
        if isinstance(node['x'], np.ndarray): node['x'] = node['x'][0]
        if isinstance(node['y'], np.ndarray): node['y'] = node['y'][0]
        if isinstance(node['z'], np.ndarray): node['z'] = node['z'][0]

    return Morphology(nodes,
                      node_id_cb=lambda node: node["id"],
                      parent_id_cb=lambda node: node["parent"])


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
        #no SWC file found for this cell 
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
    cell_depth = list(result)[1]

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
        

def get_ccf_coord_jblob(name, cell_soma_info):
    """
    Get soma loc in ccf from json, not from lims. 
    This should work for cells pinned with new pinning tool.
    """

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

    return flip

  
def convert_pixel_to_um(morph, specimen_id):
    anisotropy_value = query_for_z_resolution(specimen_id)
    for no in morph.nodes():
        no['x'] = no['x']*0.1144
        no['y'] = no['y']*0.1144
        no['z'] = no['z']*anisotropy_value
    return morph

def convert_pixel_to_um_dictnrn(morph, specimen_id):
    anisotropy_value = query_for_z_resolution(specimen_id)
    for k, v in morph.items():
        morph[k]['x'] = v['x']*0.1144
        morph[k]['y'] = v['y']*0.1144
        morph[k]['z'] = v['z']*anisotropy_value
    return morph

def get_autotrace_pp_path(spec, step):
    cell_path = os.path.join(AUTOTRACE_PATH, str(spec))
    swcs_path = os.path.join(cell_path, 'SWC')
    postproc_path = os.path.join(swcs_path, 'PostProcess')
    #look for D path first, else use C path 
    swc_path = os.path.join(postproc_path, '{}_Aspiny1.0_D.0.step{}_SortTreeIDs.swc'.format(spec, step))
    if not os.path.exists(swc_path):
        swc_path = os.path.join(postproc_path, '{}_Aspiny1.0_C.0.step{}_SortTreeIDs.swc'.format(spec, step))
    return swc_path

def get_autotrace_raw_path(spec): #TODO rigid way of getting raw file, doesn't account for diff model nums etc. 
    cell_path = os.path.join(AUTOTRACE_PATH, str(spec))
    swcs_path = os.path.join(cell_path, 'SWC')
    postproc_path = os.path.join(swcs_path, 'Raw')
    swc_path = os.path.join(postproc_path, '{}_Aspiny1.0_0.1.2_1.0.swc'.format(spec))
    return swc_path

def shrinkage_correct(morph, sp_id):
    try: 
        z_scale = scale_factor(sp_id, morph)
        for k,v in morph.items():
            cz = v['z']
            morph[k]['z'] = cz * z_scale
        return morph
    except: 
        # Currently shrinkage correction relies on there being a manual reconstruction. 
        return morph 

def overview_to_virtual_slice_transform_morphology(morph, overview_to_virtual_slice_transform):
    # apply "overview_to_virtual_slice_transform" to the (x,y) coordinates of the morphology
    for k,v in morph.items():
        point = [v['x'], v['y']]
        tpoint = list(overview_to_virtual_slice_transform.TransformPoint(point))
        morph[k]['x'] =  tpoint[0]
        morph[k]['y'] =  tpoint[1]
    return morph 

def virtual_slice_to_ccf_transform_morphology(morph, virtual_slice_to_ccf_transform):
    # apply "virtual_slice_to_ccf_transform" to the (x,y,z) coordinates of the morphology
    for k,v in morph.items():
        point = [v['x'], v['y'], v['z']]
        tpoint = list(virtual_slice_to_ccf_transform.TransformPoint(point))
        morph[k]['x'] =  tpoint[0]
        morph[k]['y'] =  tpoint[1]
        morph[k]['z'] =  tpoint[2]
    return morph

def lps_to_pir_tranform_morphology(morph):
    #  transform data from LPS (what simpleITK is in) to PIR coords (what ccf is in) 
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
    return morph 

def register_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, cell_soma_info, overview_to_virtual_slice_transform, virtual_slice_to_ccf_transform, resolution, volume_shape, z_midline):

    # 0) save original swc 
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'.swc'))

    # 1) correct for shrinkage - this should only affect the z coordinate! - currently only possible for cells with a manual reconstruction.
    morph = shrinkage_correct(morph, sp_id)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_shrinkage_corrected.swc'))

    # 2) translate the (x,y) coorinates of the morphology such that the soma node is in the corresponding position in the overview image
    soma = morph[1]
    x_shift = lims_soma[0] - soma['x']
    y_shift = lims_soma[1] - soma['y']
    morph = shift( x_shift, y_shift, 0, morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reg_shift_soma.swc'))

    # 3) transform the morphology to match the virtual slice
    morph = overview_to_virtual_slice_transform_morphology(morph, overview_to_virtual_slice_transform)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reg_to_virtual.swc'))

    # 4) transform from virtual slice to CCF space
    morph = virtual_slice_to_ccf_transform_morphology(morph, virtual_slice_to_ccf_transform)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reg.swc'))

    # 5) transform data from LPS (what simpleITK is in) to PIR coords (what ccf is in) 
    morph = lps_to_pir_tranform_morphology(morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reg_pir.swc'))

    # 6) shift to provided ccf soma coordinate
    soma = morph[1]
    ccf_soma = list(get_ccf_coord_jblob(sp_name, cell_soma_info))
    x_shift = ccf_soma[0] - soma['x']
    y_shift = ccf_soma[1] - soma['y']
    z_shift = ccf_soma[2] - soma['z']
    morph = shift( x_shift, y_shift, z_shift, morph)
    # dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reg_pir_shifted.swc'))

    # 7) resample morphology for even node spacing and save registered cell in pir coords 
    spacing = 1.144 
    morph_obj = dict_to_morphology(copy.deepcopy(morph))
    morph_obj = resample_morphology(morph_obj, spacing)
    morphology_to_swc(morph_obj, os.path.join(swc_path, swc_name+'_registered_pir.swc'))

    # 8) flip cells to the left hemisphere to align medial/lateral axis and save registered cell in pim coords 
    morph_obj = dict_to_morphology(copy.deepcopy(morph))
    morph_obj = move_soma_to_left_hemisphere(morph_obj, resolution, volume_shape, z_midline)
    morph_obj = resample_morphology(morph_obj, spacing)
    morphology_to_swc(morph_obj, os.path.join(swc_path, swc_name+'_registered_pim.swc'))



def upright_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, cell_soma_info, overview_to_virtual_slice_upright_transform, virtual_slice_to_ccf_transform,
                  resolution, volume_shape, z_midline):

    # 0) save original swc 
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'.swc'))

    # 1) if needed correct for shrinkage - this should only affect the z coordinate!
    morph = shrinkage_correct(morph, sp_id)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_shrinkage_corrected_upright_new.swc'))

    # 2) translate the (x,y) coorinates of the morphology such that the soma node is in the corresponding position in the overview image
    soma = morph[1]
    x_shift = lims_soma[0] - soma['x']
    y_shift = lims_soma[1] - soma['y']
    morph = shift( x_shift, y_shift, 0, morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_shifted_soma_upright_new.swc'))

    # 3) transform the morphology to match the virtual slice - UPRIGHT transform, only rotates to match dorsal/ventral axis. 
    morph = overview_to_virtual_slice_transform_morphology(morph, overview_to_virtual_slice_upright_transform)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_to_virtual_upright_new.swc'))

    # 4) transform from virtual slice to CCF space
    morph = virtual_slice_to_ccf_transform_morphology(morph, virtual_slice_to_ccf_transform)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reoriented_upright_new.swc'))

    # 5) transform data from LPS (what simpleITK is in) to PIR coords (what ccf is in)
    morph = lps_to_pir_tranform_morphology(morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reoriented_pir_upright_new.swc'))

    # 6) shift to provided ccf soma coordinate
    soma = morph[1]
    ccf_soma = list(get_ccf_coord_jblob(sp_name, cell_soma_info))
    x_shift = ccf_soma[0] - soma['x']
    y_shift = ccf_soma[1] - soma['y']
    z_shift = ccf_soma[2] - soma['z']
    morph = shift( x_shift, y_shift, z_shift, morph)
    dict_to_swc(morph, os.path.join(swc_path, swc_name+'_reoriented_pir_shifted_upright_new.swc'))

    # 7) normalize soma location, resample, and save upright_pir
    spacing = 1.144 
    morph_obj = dict_to_morphology(copy.deepcopy(morph))
    morph_obj = normalize_position(morph_obj) #center soma at origin 
    morph_obj = resample_morphology(morph_obj, spacing)
    morphology_to_swc(morph_obj, os.path.join(swc_path, swc_name+'_upright_pir.swc'))

    # 8) shift all somas to left hemisphere, mirror across dorsal/ventral axis. now the z axis which was previously 'right' is actually 'medial'
    morph_obj = dict_to_morphology(copy.deepcopy(morph)) #has soma at correct ccf location, essential for move_soma_to_left_hemisphere fn 
    morph_obj = move_soma_to_left_hemisphere(morph_obj, resolution, volume_shape, z_midline)
    morph_obj = normalize_position(morph_obj) #center soma at origin 
    morph_obj = resample_morphology(morph_obj, spacing)
    morphology_to_swc(morph_obj, os.path.join(swc_path, swc_name+'_upright_pim.swc'))

    # 9) coords change PIM --> MDP (x-->medial, y-->dorsal, z-->posterior)
    soma = morph_obj.get_soma()
    x = [0, 0,1, 
         0,-1,0,
         1, 0,0, 
         0, 0,0] 
    translate_transform = AffineTransform.from_list(x)
    morph_obj = translate_transform.transform_morphology(morph_obj) # if you need the original object to remain unchanged do morph.clone()
    # morph_obj = normalize_position(morph_obj) #center soma at origin 
    # morph_obj = resample_morphology(morph_obj, spacing)
    morphology_to_swc(morph_obj, os.path.join(swc_path, swc_name+'_upright_mdp.swc'))


def _calculate_rotation_angle_2d(from_vector, from_origin, to_vector, to_origin):
    """
    Calculate the rotation angle in radians to rotate 'from_vector' from 'from_origin'
    to 'to_vector' from 'to_origin'.
    """
    # Calculate vectors from origins
    from_vector_shifted = from_vector - from_origin
    to_vector_shifted = to_vector - to_origin
    
    # Normalize the vectors
    from_vector_shifted_norm = from_vector_shifted / np.linalg.norm(from_vector_shifted)
    to_vector_shifted_norm = to_vector_shifted / np.linalg.norm(to_vector_shifted)
    
    # Calculate the angle using the arctan2 function
    theta = np.arctan2(to_vector_shifted_norm[1], to_vector_shifted_norm[0]) - np.arctan2(from_vector_shifted_norm[1], from_vector_shifted_norm[0])

    return theta #radians 


def get_upright_transformation(downsampled_overview, ccf_to_virtual_slice_transform, virtual_slice_to_overview_transform, flip):

    #a) define inferior vector in PIR coordinates in ccf
    origin_ccf =    [0.0, 0.0, 0.0] # [x, y, z]
    inferior_ccf =  [0.0, 1.0, 0.0] # [x, y, z] #points towards inferior (from origin: 0,0,0)

    #b) transform from PIR coords (what ccf is in) --> LPS coords (what simpleITK is in) 
    x = [0, 0,-1,
         1, 0, 0,
         0,-1, 0]
    x = np.reshape(x,(3,3))
    origin_lps =    np.matmul(x,origin_ccf)
    inferior_lps =  np.matmul(x,inferior_ccf) 

    #c) transform into virtual slice coordinates 
    origin_vslice =     list(ccf_to_virtual_slice_transform.TransformPoint(origin_lps))
    inferior_vslice =   list(ccf_to_virtual_slice_transform.TransformPoint(inferior_lps))

    #d) tranform into 20x overview slice coordinates
    origin_20x =    list(virtual_slice_to_overview_transform.TransformPoint(origin_vslice[0:2]))
    inferior_20x =  list(virtual_slice_to_overview_transform.TransformPoint(inferior_vslice[0:2]))

    #e) Get angle (radians) to make inferior in 20x overview match inferior axis in virtual slice  
    from_point = np.array(inferior_20x[0:2])      
    from_origin = np.array(origin_20x[0:2])     
    to_point = np.array(inferior_vslice[0:2])     
    to_origin = np.array(origin_vslice[0:2])    
    rotation_angle = _calculate_rotation_angle_2d(from_point, from_origin, to_point, to_origin)

    #f) find the center of the downsampled_overview to rotate around
    width_height = (downsampled_overview.GetWidth() // 2, downsampled_overview.GetHeight() // 2)
    spacing = downsampled_overview.GetSpacing()
    center = tuple(x * y for x, y in zip(width_height, spacing))

    #g) determine if this slice needs to be flipped around the y axis (flip x axis sign) 
    flip_x_transform = sitk.AffineTransform(2)
    flip_x_transform.SetCenter(center)
    if flip: flip_x_transform.Scale([-1, 1])
    else: flip_x_transform.Scale([1, 1])

    #h) Create the Euler 2D transform to rotate so inferior axes match 
    overview_slice_to_upright_rotation = sitk.Euler2DTransform(center, -rotation_angle) #Resample function rotates the other way 

    #i) composit transform to both rotate and flip 
    composite_transform = sitk.CompositeTransform(2)
    composite_transform.AddTransform(overview_slice_to_upright_rotation) #first rotate
    composite_transform.AddTransform(flip_x_transform) #second flip if needed

    return composite_transform
