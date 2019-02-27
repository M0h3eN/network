import os

from argparse import ArgumentParser
from dataImport.commons.basicFunctions import assembleData
from fitModel.pre_processing import raw_neuronal_data_info_compute


parser = ArgumentParser(description='This is a Python program for inserting neurons information in mongoDB')

parser.add_argument('-d', '--data',  action='store',
                    dest='data', help='Raw data directory')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-H', '--host', action='store',
                    dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port',action='store',
                    dest='port', help='MongoDB port number')

args = parser.parse_args()

# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData(dirr)

# get neural data information
raw_neuronal_data_info_compute(allNeurons, args)

print('**** data ingestion completed ****')
