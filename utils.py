from dolfin import *
from dolfin_adjoint import *

import sys
import numpy as np


class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def load_xdmf_mesh(meshFile, dimension = 3, facet_flag = 0):
    # input mesh file name and path witout file extension
    # dimension = max spatial dimension of the mesh
    # (typically 2 or 3 for 2D or 3D)
    # read in the mesh

    mesh = Mesh()
    with XDMFFile(meshFile + ".xdmf") as infile:
        infile.read(mesh)

    if facet_flag:
        # read in the facet boundaries
        mvc = MeshValueCollection("size_t", mesh, dimension - 1)
        with XDMFFile(meshFile + "_facet.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        facet_regions = cpp.mesh.MeshFunctionSizet(mesh,mvc)

    # read in the defined physical regions
    mvc2 = MeshValueCollection("size_t", mesh, dimension)
    with XDMFFile(meshFile + ".xdmf") as infile:
        infile.read(mvc2, "name_to_read")
    physical_regions = cpp.mesh.MeshFunctionSizet(mesh, mvc2)

    if facet_flag:
        return mesh, facet_regions, physical_regions
    else:
        return mesh, physical_regions


def generate_xdmf_result_file(file_name):
    # can not be used to save mixelement result
    result_file = XDMFFile(MPI.comm_world, file_name)
    result_file.parameters["rewrite_function_mesh"] = False
    result_file.parameters["functions_share_mesh"] = True
    result_file.parameters["flush_output"] = True
    # result_file.write_checkpoint(result, result_name, result_step, append=append)
    return result_file


def load_xdmf_result(file_name, func, func_name, func_step = 0):
    file_read = XDMFFile(file_name)
    file_read.read_checkpoint(func, func_name, func_step)


def sample_gaussain_2d(pad: int):
    '''
    pad = 1, 3 X 3
    pad = 2, 5 X 5
    ...
    '''
    assert pad >= 0
    if pad == 0:
        return np.array([[gaussian_2d(0)]])
    sampling_window = np.zeros((2*pad+1, 2*pad+1))
    # right bottom
    sampling_quarter = np.zeros((pad+1, pad+1))
    for i in range(pad+1):
        for j in range(i+1):
            sampling_quarter[i, j] = gaussian_2d(i**2+j**2)
            sampling_quarter[j, i] = sampling_quarter[i, j]
    sampling_window[pad:, pad:] = sampling_quarter
    # up & down
    sampling_window[pad - 1::-1, pad:] = sampling_window[pad+1:, pad:]
    # left & right
    sampling_window[:, pad-1::-1] = sampling_window[:, pad+1:]
    return sampling_window


def gaussian_2d(r_squared):
    '''
    mu = 0
    sigma = 1
    '''
    return np.exp(-r_squared/2)/2/np.pi
