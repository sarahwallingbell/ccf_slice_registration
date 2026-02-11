import os 
import numpy as np
import pandas as pd
import psycopg2
import requests
import shutil

# AUTOTRACE_PATH = r'\\allen\programs\celltypes\workgroups\mousecelltypes\AutotraceReconstruction'

CONNECTION_STRING = 'host=lims2.private-allenneuraldynamics.org dbname=lims2 user={} password={}'.format(os.environ["DATABASE_USERNAME"], os.environ["DATABASE_PASSWORD"])
IMAGE_SERVICE_STRING = 'http://lims2/cgi-bin/imageservice'

# def query_engine_code_ocean(sql):
#     conn = psycopg2.connect(database="lims2",
#                         user=os.environ["DATABASE_USERNAME"],
#                         password=os.environ["DATABASE_PASSWORD"], 
#                         host="lims2.private-allenneuraldynamics.org",
#                         port="5432")
#     return pd.read_sql(sql, conn)


def query_engine_code_ocean(sql):
    with psycopg2.connect(
        database="lims2",
        user=os.environ["DATABASE_USERNAME"],
        password=os.environ["DATABASE_PASSWORD"], 
        host="lims2.private-allenneuraldynamics.org",
        port="5432"
    ) as conn:
        df = pd.read_sql(sql, conn)
    return df

def query(sql, args):
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    cur.execute(sql, args)
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

def get_ccf_coord(specimen_id):
    sql = """
        select x, y, z from cell_soma_locations
        where specimen_id = {}
        """.format(specimen_id)
    x,y,z = query(sql,())[0]
    return x,y,z

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
    df = pd.DataFrame(results, columns=["specimen_id", "specimen_name", "cell_depth",
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
    
def get_20x_img(sub_image, specimen_name, working_directory, code_ocean=True):
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

    if code_ocean:
        url = url.replace('/lims2/', '/lims2.private-allenneuraldynamics.org/')
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

def _identify_soma_marker(markers):

    try: 
        soma_markers = [m for m in markers if m["name"] == 30] # 30 is the code for soma marker
        soma_marker = soma_markers[0]
        return soma_marker
    except: 
        #no soma markers found 
        return None

def _read_marker_file(file_name):
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

def get_scale_factor(specimen_id, morph):
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
        markers = _read_marker_file(marker_path)
    else: markers = []

    soma = morph[1]

    soma_marker = _identify_soma_marker(markers)

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

# def convert_coords_str(coords_str):
#     vals = coords_str.split(',')
#     x = np.array(vals[0::2], dtype=float)
#     y = np.array(vals[1::2], dtype=float)
#     return x, y
        
