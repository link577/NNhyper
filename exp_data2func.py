import dolfin.function.functionspace
import fenics_adjoint.types.function
from dolfin import *
from dolfin_adjoint import *

import numpy as np
import pandas as pd
# import scipy.io as sio
# import matplotlib.pyplot as plt

from utils import generate_xdmf_result_file, load_xdmf_mesh, sample_gaussain_2d

# A class that convert experiment data to fenics function:
# its functionalities are:
# Take the experiment data file and the relevant parameters as input
# Generate and return the ground truth functions and the evaluating functions


class Exp2Func():
    def __init__(self, V: dolfin.function.functionspace.FunctionSpace, kwargs):
        self.kwargs = kwargs
        self.exp_data_file = self.kwargs.get("exp_data_file")
        # displacement data of all frame and all configuration
        self.disp = np.load(self.exp_data_file, allow_pickle=True).item()
        print("Data have been loaded")
        # x_min, y_min, x_max, y_max
        self.ROI_coor = self.kwargs.get("ROI_coor", [0, 0, 25.088, 13.328])
        # row_min, column_min, row_max, column_max
        self.ROI_index_range = self.kwargs.get("ROI_index_range", [25, 10, 110, 170])
        self.pixel = self.kwargs.get("pixel", 0.0392)
        self.right_boundary = self.kwargs.get("right_boundary", 13.328)
        self.load_file = self.kwargs.get("load_file", "../DIC/camera_Force.xlsx")
        self.force_load_scale = self.kwargs.get("force_load_scale", 1e2)  # N 1e-2 N
        self.force_load, self.first_frame_of_load, self.disp_load = self.load_loads()
        self.boundary_pad = self.kwargs.get("boundary_pad", 5)
        self.sampling_pad = self.kwargs.get("sampling_pad", 2)
        # function space
        self.V = V
        self.generate_map_matrix()

    def exp2func(self, step_list, func_name="displacement", save_result=True,
                 save_file="./displacement.xdmf",
                 bc_type="force"):
        '''
        :param bc_type: boundary type
        :param save_file: fittingly named
        :param func_name: name for save
        :param step_list: numpy array which stores the DIC steps to be projected
        :param save_result: whether save the projected dolfin function
        :return: a list which contains the projected dolfin function
        '''
        # 46 -- 800 frame
        assert bc_type in ["force", "disp"]
        # assert step_list[0] >= 45 and step_list[-1] <= 799
        results = []
        load_list = np.zeros_like(step_list, dtype=float)
        u = Function(self.V, name=func_name)
        if save_result:
            result_file = generate_xdmf_result_file(save_file)
        for i, step in enumerate(step_list):
            self.current_step = step
            exp_ref_ux, hole_indices, exp_ref_ux_repad, load_right_x = \
                self.data_process(step, "plot_u_ref_formatted", "u")
            exp_ref_uy, _, exp_ref_uy_repad, _ = self.data_process(step, "plot_v_ref_formatted", "v")
            if bc_type == "disp":
                # load_list[i] = load_right_x
                load_list[i] = self.disp_load[step-self.first_frame_of_load]
            elif bc_type == "force":
                load_list[i] = self.force_load[step - self.first_frame_of_load]/self.right_boundary
            # disp: ux
            u.vector()[self.V.sub(0).dofmap().dofs()] = \
                np.array([self.get_exp_data(exp_ref_ux, hole_indices, exp_ref_ux_repad, index_y, index_x)
                          if self.index_in_ROI(index_x, index_y) else 0.
                          for index_x, index_y in self.map_matrix[self.V.sub(0).dofmap().dofs(), :]])
            # disp_uy
            u.vector()[self.V.sub(1).dofmap().dofs()] = \
                np.array([self.get_exp_data(exp_ref_uy, hole_indices, exp_ref_uy_repad, index_y, index_x)
                          if self.index_in_ROI(index_x, index_y) else 0.
                          for index_x, index_y in self.map_matrix[self.V.sub(1).dofmap().dofs(), :]])
            # u.vector()[self.V.sub(1).dofmap().dofs()] = \
            #     exp_ref_uy[self.map_matrix[self.V.sub(1).dofmap().dofs(), 1],
            #                self.map_matrix[self.V.sub(1).dofmap().dofs(), 0]]
            results.append((u.copy(deepcopy=True),
                            self.force_load[step - self.first_frame_of_load]))
            if save_result:
                result_file.write_checkpoint(u, func_name, load_list[i], append=(i != 0))

            # print(f"index {i} step {step}  complete")
        return results, load_list

    def generate_map_matrix(self):
        # 2-dimensional problem
        assert self.V.mesh().topology().dim() == 2
        self.map_matrix = np.zeros((self.V.dim(), 2), dtype=int)
        data_raw = self.load_mat(0)
        data_unpadded = self.data_unpad(data_raw)
        self.exp_h_steps, self.exp_l_steps = data_unpadded.shape
        dof_coordinates = self.V.tabulate_dof_coordinates()
        x_min, y_min, x_max, y_max = self.ROI_coor
        exp_data_interval = (x_max - x_min)/(self.exp_l_steps-1)
        print(f"data_step [mm] FEM: {exp_data_interval}, DIC: {self.pixel*4}")
        # x direction
        self.map_matrix[:, 0] = \
            np.array([round((dof_coordinates[i, 0] - x_min)/exp_data_interval)
                      for i in range(self.V.dim())], dtype=int)
        # y direction
        self.map_matrix[:, 1] = \
            np.array([round((dof_coordinates[i, 1] - y_min)/exp_data_interval)
                      for i in range(self.V.dim())], dtype=int)

    def index_in_ROI(self, index_x, index_y):
        if (0 <= index_x <= self.exp_l_steps-1 and
                0 <= index_y <= self.exp_h_steps - 1):
            return True
        else:
            # print(index_x, index_y)
            return False

    def load_mat(self, dic_step, key="plot_u_ref_formatted"):
        return self.disp[key][dic_step].T

    def load_loads(self):
        load = pd.read_excel(self.load_file, usecols=(0, 2, 3)).values
        # N --> KN
        force_load = (load[:, 1] - load[0, 1]) * self.force_load_scale
        disp_load = load[:, 2] - load[0, 2]
        first_frame_of_force = int(load[0, 0] - 1)
        return force_load, first_frame_of_force, disp_load

    # def data_unpad(self, data_ref):
    #     '''
    #     :param data_ref: a 2D numpy array that stores the displacement data on the reference configuration
    #     :return: unpadded data (assume the padding number is 0)
    #     '''
    #     scale = 1e15
    #     row_indices = np.sum(scale*data_ref**2, axis=1) > 0
    #     column_indices = np.sum(scale*data_ref**2, axis=0) > 0
    #     temp_row = data_ref[row_indices, :]
    #     data_unpadded = temp_row[:, column_indices]
    #     return data_unpadded

    def data_unpad(self, data_ref):
        row_min, column_min, row_max, column_max = \
            self.ROI_index_range
        data_unpaded = data_ref[row_min:row_max+1, column_min:column_max+1]

        return data_unpaded

    def data_trim(self, data_unpadded, mode):
        '''
        Handling boundaries of the dic data
        current strategy:
        for u: Take the average value of the left and right boundaries of the DIC data
        and use them as the translation value and boundary value respectively.
        for v: DIC[0,0] = right bottom corner
        '''
        # record the hole/lost region hole = 0
        assert mode in ["u", "v"]
        scale = 1e15
        hole_indices = np.zeros_like(data_unpadded, dtype=int)
        hole_tag = scale * data_unpadded == 0
        hole_indices[~hole_tag] = int(1)
        hole_indices = np.pad(hole_indices, ((self.boundary_pad, self.boundary_pad),
                                             (self.boundary_pad, self.boundary_pad)),
                              mode='constant', constant_values=0)
        # sio.savemat("./test0.mat", {"array": hole_indices})
        # data process for u
        if mode == "u":
            left_mean = np.mean(data_unpadded[:, 0])
            # TODO: optimize this
            # data_trimed = data_unpadded - left_mean
            data_trimed = data_unpadded
            # the displacement boundary value
            right_mean = np.mean(data_trimed[:, -1])
            data_trimed[hole_tag] = 0
        # data process for v
        elif mode == "v":
            data_trimed = data_unpadded
            right_mean = np.mean(data_trimed[:, -1])
            data_trimed[hole_tag] = 0
        # padding data for sampling
        data_repad = np.pad(data_trimed, ((self.boundary_pad, self.boundary_pad),
                                          (self.boundary_pad, self.boundary_pad)),
                            mode='constant', constant_values=0)
        return data_trimed, hole_indices, data_repad, right_mean

    def data_process(self, dic_step, key, mode):
        '''
        load data, unpad data, trim data
        '''
        data = self.load_mat(dic_step, key)
        data = self.data_unpad(data)
        data, hole_indices, data_repad, load_right = self.data_trim(data, mode)
        return data, hole_indices, data_repad, load_right

    def get_exp_data(self, data, hole_indices, data_repad, index_i: int, index_j: int):
        if hole_indices[index_i+self.boundary_pad, index_j+self.boundary_pad]:
            return data[index_i, index_j]
        else:
            # print("sampling")
            index_i += self.boundary_pad
            index_j += self.boundary_pad
            for pad in np.arange(self.sampling_pad, self.boundary_pad+1, 1):
                trust_window = hole_indices[index_i - pad:index_i + pad + 1,
                               index_j - pad:index_j + pad + 1]
                # all elements in sample window is untrustworthy
                if np.sum(trust_window, (0, 1)) == 0 and pad < self.boundary_pad:
                    print(f"A {2*pad+1} X {2*pad+1} sampling window is insufficient in position ",
                          f"[{index_i-self.boundary_pad}, {index_j - self.boundary_pad}] ",
                          f"of {self.current_step}th frame\n",
                          f"A {2*pad+3} X {2*pad+3} sampling window is used")
                    continue
                if np.sum(trust_window, (0, 1)) == 0 and pad == self.boundary_pad:
                    print(f"A {2 * pad + 1} X {2 * pad + 1} sampling window is insufficient in position ",
                          f"[{index_i - self.boundary_pad}, {index_j - self.boundary_pad}] ",
                          f"of {self.current_step}th frame\n",
                          f"Maximum pad reached")
                data_window = data_repad[index_i - pad:index_i + pad + 1,
                              index_j - pad:index_j + pad + 1]
                sampling_window = sample_gaussain_2d(pad)
                weight_window = sampling_window * trust_window
                scale = np.sum(weight_window, (0, 1)) \
                    if np.sum(weight_window, (0, 1)) != 0 else 1
                weight_window = weight_window / scale
                break
            return np.sum(weight_window * data_window, (0, 1))

            # trust_window = hole_indices[index_i - self.sampling_pad:index_i + self.sampling_pad + 1,
            #                index_j - self.sampling_pad:index_j + self.sampling_pad + 1]
            # data_window = data_repad[index_i-self.sampling_pad:index_i+self.sampling_pad+1,
            #               index_j-self.sampling_pad:index_j+self.sampling_pad+1]
            # weight_window = self.sampling_window*trust_window
            # scale = np.sum(weight_window, (0, 1)) \
            #     if np.sum(weight_window, (0, 1)) != 0 else 1
            # weight_window = weight_window/scale
            # return np.sum(weight_window*data_window, (0, 1))


if __name__ == "__main__":
    # circle
    exp_kwargs = {"exp_data_file": "../DIC1113/BLACK/data.npy",
                  "ROI_coor": [0, 0, 23.72, 11.86],
                  "ROI_index_range": [34, 59, 73, 137],
                  "pixel": 0.076016,
                  "right_boundary": 14.37,
                  "load_file": "../DIC1113/BLACK/camera_Force.xlsx",
                  "force_load_scale": 1e2,
                  "boundary_pad": 5,
                  "sampling_pad": 2}
    mesh, facet_regions, physical_regions = load_xdmf_mesh("../mesh/dic1113_black_square_subdomain_intact", dimension=2, facet_flag=True)
    dx = Measure('dx', domain=mesh, subdomain_data=physical_regions)
    V = VectorFunctionSpace(mesh, "CG", 2)
    # print(V.dim())
    exp2func = Exp2Func(V, exp_kwargs)
    results, load_list = exp2func.exp2func(np.linspace(20, 825, 100, dtype=int), func_name="test",
                                           save_file="../DIC1113/BLACK/exp2func_result_CG2/test.xdmf",
                                           bc_type="disp")

    for i in range(len(results)):
        print(results[i][-1], load_list[i])

    # # no circle
    # exp_kwargs = {"exp_data_file": "../DIC0825/2023_08_25_nocircle/NO_CIRCLE/data.npy",
    #               "ROI_coor": [0,0,13.84,10.0],
    #               "ROI_index_range": [122,105,169,170],
    #               "pixel": 0.05322,
    #               "right_boundary": 15.22,
    #               "load_file": "../DIC0825/2023_08_25_nocircle/NO_CIRCLE/camera_Force.xlsx",
    #               "force_load_scale": 1e2,
    #               "boundary_pad": 5,
    #               "sampling_pad": 2}
    # mesh, facet_regions, physical_regions = load_xdmf_mesh("../mesh/dic0825_nocircle", dimension=2, facet_flag=True)
    # dx = Measure('dx', domain=mesh, subdomain_data=physical_regions)
    # V = VectorFunctionSpace(mesh, "CG", 2)
    # # print(V.dim())
    # exp2func = Exp2Func(V, exp_kwargs)
    # results, load_list = exp2func.exp2func(np.linspace(45, 1000, 1000-45+1, dtype=int), func_name="test1000",
    #                                        save_file="../DIC0825/2023_08_25_nocircle/NO_CIRCLE/exp2func_result_CG2/test1000.xdmf",
    #                                        bc_type="disp")

    # np.savetxt("../DIC/exp2func_result_CG2/loadlist.txt", load_list)
    # plt.plot(np.arange(45, 1104), load_list)
    # plt.xlabel("Frame")
    # plt.ylabel("Mean $u_x$ [mm] in right boundary")
    # plt.savefig("../DIC/exp2func_result_CG2/load.png")
    # plt.show()
    # plt.close()


    # for i in range(1104):
    #     data1 = exp2func.load_mat(i)
    #     data2 = exp2func.data_unpad(data1)
    #     if data2.shape != (86, 161):
    #         print(i)
