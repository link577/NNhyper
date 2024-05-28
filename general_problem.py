from dolfin import *
from dolfin_adjoint import *

import numpy as np

from utils import generate_xdmf_result_file, load_xdmf_mesh

class GeneralCase(object):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.mesh_file = None
        self.post_file = None
        pass

    def read_mesh(self, dimension = 2):
        self.mesh, self.facet_regions, self.physical_regions = load_xdmf_mesh(self.mesh_file,
                                                                              dimension=dimension,
                                                                              facet_flag=True)

        self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.physical_regions)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.facet_regions)
        self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.facet_regions)  # internal boundary

        self.dim = self.mesh.topology().dim()
        self.facet_dim = self.dim - 1
        self.normal_vector = FacetNormal(self.mesh)
        self.coorX = SpatialCoordinate(self.mesh)

    def check_boundary(self, func_space=None, bcs=None, default_value=np.pi,
                       file_name=None):
        if file_name is None:
            file_name = self.post_file + f"boundary.xdmf"
        result_file = generate_xdmf_result_file(file_name)
        func = Function(func_space, name="boundary")
        func.vector()[:] = default_value * np.ones_like(func.vector()[:])
        [bc.apply(func.vector()) for bc in bcs]
        result_file.write_checkpoint(func, "boundary", 0, append=False)

    def write_log(self, info="", filename=None, clear=False, print_out=True, end="\n"):
        if filename is None:
            filename = self.post_file+f"log.txt"
        with open(filename, "a") as f:
            if clear:
                f.truncate(0)
            f.write(f"{info}{end}")
        if print_out:
            print(info)