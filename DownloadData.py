#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
'stream'    : "oper",
    'levtype'   : "sfc",
    'param'     : "165.128",
    'dataset'   : "interim",
    'step'      : "0",
    'grid'      : "0.5/0.5",
    'time'      : "00/06/12/18",
    'date'      : "2016-08-01/to/2016-11-30",
    'type'      : "an",
    'class'     : "ei",
    'area'      : "35.5/-85.5/34.5/-84.5",
    'format'    : "netcdf",
    'target'    : "Chattanooga.nc"})

