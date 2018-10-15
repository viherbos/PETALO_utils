import numpy as np
import scipy as sp
import sys
sys.path.append("../PETALO_DAQ_infinity/SimLib")
sys.path.append("../PETALO_analysis")
import os
import pandas as pd
from SimLib import config_sim as CFG
from SimLib import HF_files as HF



def main():
    # Argument parser for config file name
    parser = argparse.ArgumentParser(description='PETALO Output Comparator.')
    parser.add_argument("-f", "--json_file", action="store_true",
                        help="Control File (json)")
    parser.add_argument('arg1', metavar='N', nargs='?', help='')
    parser.add_argument("-d", "--directory", action="store_true",
                        help="Work directory")
    parser.add_argument('arg2', metavar='N', nargs='?', help='')
    args = parser.parse_args()

    if args.json_file:
        file_name = ''.join(args.arg1)
    else:
        file_name = "sim_config"
    if args.directory:
        path = ''.join(args.arg2)
    else:
        path="./"

    config_file = file_name + ".json"

    CG = CFG.SIM_DATA(filename = path + config_file, read = True)
    CG = CG.data

    DAQ_outfile = HF.DAQ_IO(
                    path         = CG['ENVIRONMENT']['path_to_files'],
                    daq_filename = CG['ENVIRONMENT']['file_name'],
                    ref_filename = CG['ENVIRONMENT']['file_name']+"0.h5",
                    daq_outfile  = CG['ENVIRONMENT']['out_file_name']+"_"+ file_name + ".h5")

    FASTDAQ_outfile = HF.DAQ_IO(
                    path         = CG['ENVIRONMENT']['path_to_files'],
                    daq_filename = CG['ENVIRONMENT']['file_name'],
                    ref_filename = CG['ENVIRONMENT']['file_name']+"0.h5",
                    daq_outfile  = CG['ENVIRONMENT']['MC_out_file_name']+"_"+ file_name + ".h5")


    data_DAQ    ,sensors = DAQ_outfile.read()
    data_FASTDAQ,sensors = FASTDAQ_outfile.read()

    i = 0
    for event_F in data_FASTDAQ:

        if (np.any(np.abs(event_F-data_DAQ[i]) != 0)):
            print ("Event %d is different in FASTDAQ and DAQ simulation" % i)

        i += 1
    

if __name__ == "__main__":
    main()
