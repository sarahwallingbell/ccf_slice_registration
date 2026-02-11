import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from query import get_name_by_id, get_pinning, get_20x_info, get_20x_img, get_soma_polygons, get_landmark_ids, get_landmark_location

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

def slice_flipped(ldf):
    """
    Determines if the order of fiducial pins is flipped along the x-axis between
    the overview image and the virtual slice, based on pin names.

    Returns True if flipped, False otherwise.
    """
    # Find pins with min and max x in overview
    min_overview_name = ldf.set_index('specimen_name')['overview_coordinate'].apply(lambda x: x[0]).idxmin()
    max_overview_name = ldf.set_index('specimen_name')['overview_coordinate'].apply(lambda x: x[0]).idxmax()

    # Find pins with min and max x in virtual slice
    min_virtual_name = ldf.set_index('specimen_name')['virtual_slice_coordinate'].apply(lambda x: x[0]).idxmin()
    max_virtual_name = ldf.set_index('specimen_name')['virtual_slice_coordinate'].apply(lambda x: x[0]).idxmax()

    # If either first or last pin names match, it's not flipped
    if min_overview_name == min_virtual_name or \
       max_overview_name == max_virtual_name:
        return False
    else:
        return True
  
def flatten( lm ) :    
    # flatten a set of points as input to SimpleITK landmark based registration
    return [ c for p in lm for c in p]

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

def compute_center_from_polyline (aa) :
    aa = aa.split(',')
    aa = [int(x) for x in aa]
    aa = np.reshape(aa,(int(len(aa)/2),2))
    return np.mean( aa , axis = 0)

def calculate_rotation_angle_2d(from_vector, from_origin, to_vector, to_origin):
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
    rotation_angle = calculate_rotation_angle_2d(from_point, from_origin, to_point, to_origin)

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

def slice_transform_to_ccf(specimen_id, specimen_name, out, ccf):

    """
    specimen_id is slice id.
    """
    
    #make folder to store transform for this slice 
    working_directory = os.path.join(out, specimen_name)
    if not os.path.exists(working_directory):
        os.mkdir(working_directory)

    # read in pinning tool output - virtual slice definition and cell ccf locations
    pinning_info = get_pinning(specimen_name)[4]

    # virtual_slice_to_ccf_transform: transform a 3D point (in micron) in the virtual slice to ccf (in micron)
    # ccf_to_virtual_slice_transform: transform a 3D point in ccf (in micron) to virtual slice (in micron)
    # (note: each point can have its own orientation - we are taking the first one only)
    virtual_slice_to_ccf_transform = initialize_transform( pinning_info['markups'][0]['orientation'] )
    ccf_to_virtual_slice_transform = virtual_slice_to_ccf_transform.GetInverse()

    # write out the virtual_slice_to_ccf_transform
    file = os.path.join(working_directory, 'virtual_slice_to_ccf_transform.txt')
    sitk.WriteTransform( virtual_slice_to_ccf_transform, file )

    # generate a virtual slice from the transforms
    #   virtual_slice_3d is a 3D volume with a single z slice
    #   virtual_slice    is a 2D volume created by extracting out the single slice
    virtual_slice_size = [1250,1250,1] # 3D volume with one slice
    virtual_slice_spacing = ccf.GetSpacing()
    virtual_slice_3d = sitk.Resample(ccf, 
                                    virtual_slice_size, 
                                    virtual_slice_to_ccf_transform, 
                                    sitk.sitkLinear,
                                    [0,0,0], 
                                    virtual_slice_spacing, 
                                    [1,0,0,0,1,0,0,0,1], 
                                    0.0, 
                                    ccf.GetPixelID())#, False )


    virtual_slice = virtual_slice_3d[:,:,0] # 2D image version

    # read in the 2D overview image and associate metadata
    img_info = [list(get_20x_info(specimen_name))]
    odf = pd.DataFrame(img_info, columns = ['specimen_name', 'sub_image_id', 'width', 'height', 'resolution', 'treatment_id'])
    overview_info = odf.loc[0]
    sub_image = odf['sub_image_id'].values[0]
    url = get_20x_img(sub_image, specimen_name, working_directory) 
    file = os.path.join(working_directory, '{}_overview.jpg'.format(specimen_name))
    overview = sitk.ReadImage( file )
    overview_spacing = [overview_info['resolution'],overview_info['resolution']]
    overview.SetSpacing(overview_spacing)

    # downsample the 2D overview image
    downsampled_overview = sitk.BinShrink( sitk.VectorIndexSelectionCast(overview,0), [25,25])

    # write the downsampled overview to file
    file = os.path.join(working_directory,'downsampled_overview.nii.gz')
    sitk.WriteImage( downsampled_overview, file, True )
    # Read in drawn soma polygons to create matching landmarks set
    df = get_soma_polygons(specimen_id)

    # For each cell
    #  - compute cell soma from polyline in pixels
    #  - convert cell soma location to microns
    #  - join with cell soma location in CCF
    #  - compute cell soma location in virtual slice
    df['center_pixel'] = [compute_center_from_polyline(p) for p in df['poly_coords']]
    df['center_micron'] = [np.multiply(p,overview_spacing) for p in df['center_pixel']]

    jdict = {}
    for m in pinning_info['markups'] :
        jdict[m['name'].strip()] = m['markup']['controlPoints'][0]['position']
        
    df['ccf_coordinate'] = [jdict[p] for p in df['specimen_name']]
    df['virtual_slice_coordinate'] = [ ccf_to_virtual_slice_transform.TransformPoint(p)[:2] for p in df['ccf_coordinate'] ]
    df.to_csv(os.path.join(working_directory, 'alignment_output.csv'), index=False)

    lndmrks = get_landmark_ids(specimen_id)
    ldf = pd.DataFrame()
    for c in lndmrks:
        # print(c)
        this_lndmrk = get_landmark_location(c[0])
        ldf = pd.concat([ldf, this_lndmrk])
        
    ldf['center_pixel'] = [compute_center_from_polyline(p) for p in ldf['poly_coords']]
    ldf['overview_coordinate'] = [np.multiply(p,overview_spacing) for p in ldf['center_pixel']]

    ldf['ccf_coordinate'] = [jdict[p] for p in ldf['specimen_name']]
    ldf['virtual_slice_coordinate'] = [ ccf_to_virtual_slice_transform.TransformPoint(p)[:2] for p in ldf['ccf_coordinate'] ]

    if len(ldf) < 3:
        # slices_with_issues[specimen_name] = 'not enough fiducials'
        return f'Error: not enough fiducials, {len(ldf)} fiducials found'
    

    # determine if slice was flipped, if so fix virtual to ccf transform 
    flip = False
    if slice_flipped(ldf):
        transform = virtual_slice_to_ccf_transform.GetParameters()
        transform = list(transform)
        idx = -1
        if pinning_info['referenceView'].lower() == 'sagittal': idx = 2 #flip coronal axis in PIL to LPS transform 
        if pinning_info['referenceView'].lower() == 'coronal': idx = 5 #flip saggital axis in RIA to LPS transform 
        if idx > -1: 
            transform[idx] = transform[idx]*-1 #flip axis
            flip = True 
        transform = tuple(transform)
        virtual_slice_to_ccf_transform.SetParameters(transform)

        # write out the virtual_slice_to_ccf_transform
        file = os.path.join(working_directory, 'virtual_slice_to_ccf_transform.txt')
        sitk.WriteTransform( virtual_slice_to_ccf_transform, file )
    
    # Concatentate landmarks (cell soma + additional) for registration
    # virtual slice (fixed) landmarks
    fixed_landmarks = [tuple(p) for p in df['virtual_slice_coordinate']]
    fixed_landmarks.extend([tuple(p) for p in ldf['virtual_slice_coordinate']] )

    # overview (moving) landmarks
    moving_landmarks = [tuple(p) for p in df['center_micron']]
    moving_landmarks.extend([tuple(p) for p in ldf['overview_coordinate']] )

    # virtual_slice_to_overview_transform: transform a 2D point (in microns) in the virtual slice to overview image (in microns)
    # overview_to_virtual_slice_transform: transform a 3D point in overview image (in microns) to virtual slice (in microns)
    virtual_slice_to_overview_transform = sitk.LandmarkBasedTransformInitializer( sitk.AffineTransform(2), flatten(fixed_landmarks), flatten(moving_landmarks) )
    overview_to_virtual_slice_transform = virtual_slice_to_overview_transform.GetInverse()

    # write out the overview_to_virtual_slice_transform
    file = os.path.join(working_directory,'overview_to_virtual_slice_transform.txt')
    sitk.WriteTransform( overview_to_virtual_slice_transform, file )

    # generate resampled overview image
    resampled_overview = \
        sitk.Resample(downsampled_overview, virtual_slice, virtual_slice_to_overview_transform, \
                    sitk.sitkLinear, 0, downsampled_overview.GetPixelID())

    # write the downsampled overview to file
    file = os.path.join(working_directory,'resampled_overview.nii.gz')
    sitk.WriteImage( resampled_overview, file, True ) 

    # get the upright-only transformation
    virtual_slice_to_overview_upright_transform = get_upright_transformation(downsampled_overview, ccf_to_virtual_slice_transform, virtual_slice_to_overview_transform, flip)
    overview_to_virtual_slice_upright_transform = virtual_slice_to_overview_upright_transform.GetInverse()
    moving_landmarks_upright = [overview_to_virtual_slice_upright_transform.TransformPoint(xy) for xy in moving_landmarks]

    # upright downsampled overview and save to file
    upright_overview = \
        sitk.Resample(downsampled_overview, virtual_slice_to_overview_upright_transform, \
                    sitk.sitkLinear, 0.0, downsampled_overview.GetPixelID())
    file = os.path.join(working_directory,'upright_overview.nii.gz')
    sitk.WriteImage( upright_overview, file, True ) 

    #save the upright transform to use on morphologies 
    file = os.path.join(working_directory,'overview_to_virtual_slice_upright_transform.txt')
    sitk.WriteTransform( overview_to_virtual_slice_upright_transform, file )

    #save slice transform overview image
    fig, axes = plt.subplots(nrows=1,ncols=4,figsize=(20,5)) 
    visualize_landmarks(downsampled_overview, axes[0], moving_landmarks)
    axes[0].set_title('20x with fiducials')
    visualize_landmarks( virtual_slice, axes[1], fixed_landmarks)
    axes[1].set_title('virtual slice with fiducials')
    visualize_landmarks( resampled_overview, axes[2], fixed_landmarks)
    axes[2].set_title('resampled 20x')
    visualize_landmarks( upright_overview, axes[3], moving_landmarks_upright)
    axes[3].set_title('upright 20x')
    plt.savefig(os.path.join(working_directory, 'transformation_overview.jpg'))
    # plt.show() 
    plt.clf()
