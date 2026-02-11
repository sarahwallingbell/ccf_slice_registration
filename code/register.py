import os
import pandas as pd
import numpy as np
import copy
import SimpleITK as sitk
from query import get_id_by_name, get_swc_from_lims, get_scale_factor
from neuron_morphology.transforms.affine_transform import AffineTransform
from neuron_morphology.morphology import Morphology
from neuron_morphology.swc_io import morphology_to_swc
from morph_utils.query import query_for_z_resolution
from morph_utils.ccf import move_soma_to_left_hemisphere
from morph_utils.modifications import resample_morphology, normalize_position

def edit_path(p):
    p = p.replace('\\', '/')
    p = p.replace('/', '//', 1)
    return p

def get_ccf_coord_jblob(name, cell_soma_info):
    """
    Get soma loc in ccf from json, not from lims. 
    This should work for cells pinned with new pinning tool.
    """

    row = cell_soma_info[cell_soma_info.specimen_name == name].iloc[0]
    return row.x, row.y, row.z

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
        z_scale = get_scale_factor(sp_id, morph)
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

    # 1.5) resample 
    spacing = 1.144 
    morph_r = dict_to_morphology(copy.deepcopy(morph))
    morph_r = resample_morphology(morph_r, spacing)
    morphology_to_swc(morph_r, os.path.join(swc_path, swc_name+'_shrinkage_corrected_resampled.swc'))

    # 2) translate the (x,y) coorinates of the morphology such that the soma node is in the corresponding position in the overview image
    soma = morph[1]
    x_shift = lims_soma[0] - soma['x']
    y_shift = lims_soma[1] - soma['y']
    morph = shift( x_shift, y_shift, 0, morph)

    # 3) transform the morphology to match the virtual slice
    morph = overview_to_virtual_slice_transform_morphology(morph, overview_to_virtual_slice_transform)

    # 4) transform from virtual slice to CCF space
    morph = virtual_slice_to_ccf_transform_morphology(morph, virtual_slice_to_ccf_transform)

    # 5) transform data from LPS (what simpleITK is in) to PIR coords (what ccf is in) 
    morph = lps_to_pir_tranform_morphology(morph)

    # 6) shift to provided ccf soma coordinate
    soma = morph[1]
    ccf_soma = list(get_ccf_coord_jblob(sp_name, cell_soma_info))
    x_shift = ccf_soma[0] - soma['x']
    y_shift = ccf_soma[1] - soma['y']
    z_shift = ccf_soma[2] - soma['z']
    morph = shift( x_shift, y_shift, z_shift, morph)

    # 7) resample morphology for even node spacing and save registered cell in pir coords 
    # spacing = 1.144 
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

    # 1) if needed correct for shrinkage - this should only affect the z coordinate!
    morph = shrinkage_correct(morph, sp_id)

    # 2) translate the (x,y) coorinates of the morphology such that the soma node is in the corresponding position in the overview image
    soma = morph[1]
    x_shift = lims_soma[0] - soma['x']
    y_shift = lims_soma[1] - soma['y']
    morph = shift( x_shift, y_shift, 0, morph)

    # 3) transform the morphology to match the virtual slice - UPRIGHT transform, only rotates to match dorsal/ventral axis. 
    morph = overview_to_virtual_slice_transform_morphology(morph, overview_to_virtual_slice_upright_transform)

    # 4) transform from virtual slice to CCF space
    morph = virtual_slice_to_ccf_transform_morphology(morph, virtual_slice_to_ccf_transform)

    # 5) transform data from LPS (what simpleITK is in) to PIR coords (what ccf is in)
    morph = lps_to_pir_tranform_morphology(morph)

    # 6) shift to provided ccf soma coordinate
    soma = morph[1]
    ccf_soma = list(get_ccf_coord_jblob(sp_name, cell_soma_info))
    x_shift = ccf_soma[0] - soma['x']
    y_shift = ccf_soma[1] - soma['y']
    z_shift = ccf_soma[2] - soma['z']
    morph = shift( x_shift, y_shift, z_shift, morph)

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

def register_morphologies(sp_name, sl_name, out, somas, resolution, volume_shape, z_midline):

    """
        sp_name = cell specimen name 
        sl_name = slice name 
    """

    registered_cells = []
    uprighted_cells = []

    #get cell id
    sp_id = get_id_by_name(sp_name)

    #check if there's a transform for this slice 
    slice_path = os.path.join(out, sl_name)
    if not os.path.isfile(os.path.join(slice_path, 'overview_to_virtual_slice_transform.txt')): 
        return 'ERROR: overview_to_virtual_slice_transform.txt not found', registered_cells, uprighted_cells
    if not os.path.isfile(os.path.join(slice_path, 'virtual_slice_to_ccf_transform.txt')): 
        return 'ERROR: virtual_slice_to_ccf_transform.txt not found', registered_cells, uprighted_cells
    if not os.path.isfile(os.path.join(slice_path, 'alignment_output.csv')): 
        return 'ERROR: alignment_output.csv not found', registered_cells, uprighted_cells

    #get soma loc in 20x 
    alignment_output = pd.read_csv(os.path.join(slice_path, 'alignment_output.csv'))
    this_cell_alignment = alignment_output.query("draw_type == 'Soma'")[alignment_output.specimen_name == sp_name]
    if len(this_cell_alignment) == 0: 
        return 'ERROR: no soma pin', registered_cells, uprighted_cells
    if len(this_cell_alignment) > 1: 
        cells_with_issues[sp_name] = 'Multiple soma pins'
        return 'ERROR: multiple soma pins', registered_cells, uprighted_cells
    lims_soma = this_cell_alignment['center_micron'].values
    lims_soma = lims_soma[0][1:-2].split(' ')
    lims_soma = [float(i) for i in lims_soma if len(i) > 0]


    #make folder to save registered swcs
    swc_path = os.path.join(slice_path, 'SWC')
    if not os.path.exists(swc_path): os.mkdir(swc_path)


    #load affine transforms
    overview_to_virtual_slice_transform =           sitk.ReadTransform(os.path.join(slice_path, 'overview_to_virtual_slice_transform.txt'))
    overview_to_virtual_slice_upright_transform =   sitk.ReadTransform(os.path.join(slice_path, 'overview_to_virtual_slice_upright_transform.txt'))
    virtual_slice_to_ccf_transform =                sitk.ReadTransform(os.path.join(slice_path, 'virtual_slice_to_ccf_transform.txt'))


    #register manual swc 
    try: 
        swc_name = '{}'.format(sp_id)
        lims_path = list(get_swc_from_lims(str(sp_id)))[1]
        lims_path = edit_path(lims_path)
        lims_path = '/root/capsule/data/aibs-isilon-bucket-01' + lims_path #Code Ocean s3 data asset location (from Isilon location)

        #register to CCF
        morph = to_dict(lims_path)
        register_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, somas, 
                    overview_to_virtual_slice_transform, virtual_slice_to_ccf_transform,
                    resolution, volume_shape, z_midline)
        registered_cells = registered_cells + [sp_name+'_manual']
        
        #upright 
        morph = to_dict(lims_path)
        upright_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, somas, 
                    overview_to_virtual_slice_upright_transform, virtual_slice_to_ccf_transform,
                    resolution, volume_shape, z_midline)
        uprighted_cells = uprighted_cells + [sp_name+'_manual']
            
    except: 
        return 'NOTE: no manual swc to register', registered_cells, uprighted_cells


    #TODO add autotrace registration back when they are migrated to S3

    # autotrace_registered = False
    # #register autotrace post processed step 14 swc
    # try: 
    #     pp14_path = get_autotrace_pp_path(sp_id, 14)
    #     swc_name = pp14_path.rsplit('\\',1)[1].split('.swc',1)[0]

    #     #register to CCF
    #     morph = to_dict(pp14_path)
    #     morph = convert_pixel_to_um_dictnrn(morph, sp_id)
    #     register_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, somas, 
    #             overview_to_virtual_slice_transform, virtual_slice_to_ccf_transform,
    #             resolution, volume_shape, z_midline)
    #     registered_cells = registered_cells + [sp_name+'_pp14']

    #     #upright 
    #     morph = to_dict(pp14_path)
    #     morph = convert_pixel_to_um_dictnrn(morph, sp_id)
    #     upright_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, somas, 
    #                 overview_to_virtual_slice_upright_transform, virtual_slice_to_ccf_transform,
    #                 resolution, volume_shape, z_midline)
    #     uprighted_cells = uprighted_cells + [sp_name+'_pp14']

    #     autotrace_registered = True
    # except: print('\t\tno pp14 autotrace swc to register')


    # #register raw autotrace swc if no post processed version 
    # if not autotrace_registered:
    #     try: 
    #         raw_path = get_autotrace_raw_path(sp_id)
    #         swc_name = raw_path.rsplit('\\',1)[1].split('.swc',1)[0]

    #         #register to CCF
    #         morph = to_dict(raw_path)
    #         morph = convert_pixel_to_um_dictnrn(morph, sp_id)
    #         register_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, somas, 
    #                 overview_to_virtual_slice_transform, virtual_slice_to_ccf_transform,
    #                 resolution, volume_shape, z_midline)
    #         registered_cells = registered_cells + [sp_name+'_raw']

    #         #upright 
    #         morph = to_dict(raw_path)
    #         morph = convert_pixel_to_um_dictnrn(morph, sp_id)
    #         upright_morph(sp_name, sp_id, lims_soma, morph, swc_path, swc_name, somas, 
    #                     overview_to_virtual_slice_upright_transform, virtual_slice_to_ccf_transform,
    #                     resolution, volume_shape, z_midline)
    #         uprighted_cells = uprighted_cells + [sp_name+'_raw']

    #     except: print('\t\tno raw autotrace swc to register')

    return None, registered_cells, uprighted_cells
