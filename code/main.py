import os 
import numpy as np
import pandas as pd
import argschema as ags
import SimpleITK as sitk
from pins import get_pins
from query import query_engine_code_ocean, get_id_by_name, get_name_by_id
from transform import slice_transform_to_ccf
from register import register_morphologies
from tqdm import tqdm


class RegisterCCFSchema(ags.ArgSchema):
    out = ags.fields.OutputDir(dump_default="/root/capsule/results/test", metadata={'description':"Output folder for results"})
    # TODO add S3 bucket to write results to
    ccf_annotation_file = ags.fields.String( dump_default="/root/capsule/data/ccf/annotation_10.nrrd", metadata={'description':"Path to CCF annotation file"})
    ccf_template_file = ags.fields.String(dump_default="/root/capsule/data/ccf/average_template_10.nii.gz", metadata={'description': 'Path to CCF file'})
    ccf_resolution = ags.fields.Int(dump_default=10, metadata={'description':"CCF voxel resolution in microns"})
    slice_file = ags.fields.String(dump_default=None, metadata={'description': 'File with slice names to regsiter'})
    #TODO add cell file option?

def main(args):


    #1. Get pins 

    print('\n\nGetting pins...')
    somas, slices = get_pins(output_folder = args['out'], 
                             annotation_file = args['ccf_annotation_file'],
                             query_engine = query_engine_code_ocean)


    if args['slice_file'] is not None and os.path.exists(args['slice_file']):
        # Filter list to only register the input slices/cells
        with open(args['slice_file'], "r") as f:
            slice_names = [line.strip() for line in f]
        slices = [get_id_by_name(x) for x in slice_names]
        somas = somas[somas.slide_specimen_id.isin(slices)]          


    #2. Load CCF

    print('\n\nLoading CCF...')
    ccf = sitk.ReadImage( args['ccf_template_file'] )
    ccf_volume_shape = ccf.GetSize()
    ccf_z_size = ccf_volume_shape[2] * args['ccf_resolution']
    ccf_z_midline = ccf_z_size / 2


    #3. Get slice transforms to CCF 

    slices_with_issues = {}
    for specimen_id in tqdm(slices, desc="Calculating slice transforms"):
        specimen_id = int(specimen_id)
        specimen_name = get_name_by_id(specimen_id)
        #loop through all slices, finding transforms to register and upright them to the ccf 
        try:
            slice_error = slice_transform_to_ccf(specimen_id, specimen_name, args['out'], ccf)
            if not slice_error is None:
                slices_with_issues[specimen_name] = slice_error
        except: 
            slices_with_issues[specimen_name] = 'issue with this slice'
            continue
    slices_with_issues_df = pd.DataFrame.from_dict(slices_with_issues.items())
    slices_with_issues_df.to_csv(os.path.join(args['out'], 'slices_with_issues.csv'), index=False)


    #4. Register morphologies to CCF

    registered_cells = []
    uprighted_cells = []
    cells_with_issues = {}
    for idx, cell in tqdm(somas.iterrows(), total=len(somas), desc="Registering morphologies"):

        sp_name = cell['specimen_name']
        sl_name = sp_name.rsplit('.', 1)[0]
        try:
            cell_error, this_registered_cells, this_uprighted_cells = register_morphologies(sp_name, sl_name, args['out'], somas, args['ccf_resolution'], ccf_volume_shape, ccf_z_midline)
            if not cell_error is None:
                cells_with_issues[sp_name] = cell_error
            registered_cells = registered_cells + this_registered_cells
            uprighted_cells = uprighted_cells + this_uprighted_cells
        except: 
            cells_with_issues[sp_name] = 'Issue with this cell'
            continue
    cells_with_issues_df = pd.DataFrame.from_dict(list(cells_with_issues.items())) 
    cells_with_issues_df.to_csv(os.path.join(args['out'], 'cells_with_issues.csv'), index=False)

    registered_cells_df = pd.DataFrame(registered_cells)
    registered_cells_df.to_csv(os.path.join(args['out'], 'registered_cells.csv'), index=False)


    # TODO 5. Write results to S3 (need persistant programatic read/write access via IAM user)


    # TODO do we want the files organized differently in the end? i.e. all the upright swcs pooled together?


    print('\n\nRegistration Complete!\n\n')


if __name__ == "__main__":

    mod = ags.ArgSchemaParser(schema_type=RegisterCCFSchema)
    main(mod.args)

