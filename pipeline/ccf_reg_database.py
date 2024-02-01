import sqlite3
import pandas as pd

def ensure_tables_exist(database_file):
    con = sqlite3.connect(database_file)
    cur = con.cursor()

    # check that the registered_slices table exists
    res = cur.execute("SELECT name FROM sqlite_master")
    if ("registered_slices",) not in res.fetchall():
        cur.execute(
            "CREATE TABLE registered_slices(slice_id, slice_name, registered, overview_to_virtual_slice_transform_path, virtual_slice_to_ccf_transform_path, registration_timestamp, recommend_review)")

    # check that the registered_cells table exists
    res = cur.execute("SELECT name FROM sqlite_master")
    if ("registered_cells",) not in res.fetchall():
        cur.execute(
            "CREATE TABLE registered_cells(specimen_id, specimen_name, slice_id, has_manual, manual_registered, manual_swc_path, manual_registration_timestamp, has_raw_autotrace, raw_autotrace_registered, raw_autotrace_swc_path, raw_autotrace_registration_timestamp, has_post_processed_step_14_autotrace, post_processed_step_14_autotrace_registered, post_processed_step_14_autotrace_swc_path, post_processed_step_14_autotrace_registration_timestamp, has_post_processed_step_22_autotrace, post_processed_step_22_autotrace_registered, post_processed_step_22_autotrace_swc_path, post_processed_step_22_autotrace_registration_timestamp)")

    cur.close()
    con.close()



def get_registered_slices(database_file):
    con = sqlite3.connect(database_file)
    cur = con.cursor()

    #select registered slices and put in df 
    existing_slice_registrations = pd.read_sql_query("SELECT * \
                                                  FROM registered_slices rs \
                                                  WHERE rs.registered == 1", con)
    cur.close()
    con.close()

    return existing_slice_registrations


def get_registered_cells(database_file):
    con = sqlite3.connect(database_file)
    cur = con.cursor()

    #select cells where all existing swcs are registered and put in df 
    existing_cell_registrations = pd.read_sql_query("SELECT * \
                                                    FROM registered_cells rc \
                                                    WHERE rc.has_manual == rc.manual_registered \
                                                    AND rc.has_raw_autotrace == rc.raw_autotrace_registered \
                                                    AND rc.has_post_processed_step_14_autotrace == post_processed_step_14_autotrace_registered \
                                                    AND rc.has_post_processed_step_22_autotrace == post_processed_step_22_autotrace_registered", con)
    cur.close()
    con.close()

    return existing_cell_registrations


def add_slice_registration(database_file, slice):
    con = sqlite3.connect(database_file)
    cur = con.cursor()

    sql = ''' INSERT INTO registered_slices(slice_id, slice_name, registered, overview_to_virtual_slice_transform_path, virtual_slice_to_ccf_transform_path, registration_timestamp,recommend_review)
              VALUES(?,?,?,?,?,?,?) '''
    cur.execute(sql, slice)
    con.commit()

    cur.close()
    con.close()


def add_cell_registration(database_file, cell):
    con = sqlite3.connect(database_file)
    cur = con.cursor()

    sql = ''' INSERT INTO registered_cells(specimen_id, specimen_name, slice_id, has_manual, manual_registered, manual_swc_path, manual_registration_timestamp, has_raw_autotrace, raw_autotrace_registered, raw_autotrace_swc_path, raw_autotrace_registration_timestamp, has_post_processed_step_14_autotrace, post_processed_step_14_autotrace_registered, post_processed_step_14_autotrace_swc_path, post_processed_step_14_autotrace_registration_timestamp, has_post_processed_step_22_autotrace, post_processed_step_22_autotrace_registered, post_processed_step_22_autotrace_swc_path, post_processed_step_22_autotrace_registration_timestamp)
              VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '''
    cur.execute(sql, cell)
    con.commit()

    cur.close()
    con.close()



