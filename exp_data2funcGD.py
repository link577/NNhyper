import dolfin.function.functionspace
import exp_data2func as e2f
# used for generating initial value
from nnhyper_plane_stress import *
from dolfin import *
from dolfin_adjoint import *

import numpy as np
import pandas as pd
import os, time
from utils import generate_xdmf_result_file


class Exp2Func_GD():
    def __init__(self, V: dolfin.function.functionspace.FunctionSpace, kwargs):
        self.kwargs = kwargs
        self.exp_data_file = self.kwargs.get("exp_data_file")
        # displacement data of all frame and all configuration
        # self.disp = np.load(self.exp_data_file, allow_pickle=True).item()
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
        self.preprocess_file = self.kwargs.get("preprocess_file", None)
        # function space
        self.V = V
        self.dof_coordinate = self.V.tabulate_dof_coordinates()
        self.u_index = self.V.sub(0).dofmap().dofs()
        self.v_index = self.V.sub(1).dofmap().dofs()
        self.dof_neighbor = self.generate_dof_neighbor()


        self.e2f = e2f.Exp2Func(self.V, self.kwargs)
        self.time_start = time.time()

    def exp2func_gd(self, step_list, func_name="displacement", save_result=True,
                    save_file="./displacement.xdmf",
                    bc_type="force"):

        result, load_list = self.e2f.exp2func(step_list,
                                              func_name=func_name,
                                              save_result=False,
                                              bc_type=bc_type)
        if self.preprocess_file is not None:
            file_read = XDMFFile(self.preprocess_file)
            for i, step in enumerate(step_list):
                file_read.read_checkpoint(result[i][0], "dic_disp_gd", i)

        if self.preprocess_file is None:
            print("warning: preprocess is about to take place")
            self.time_start = time.time()
            save_pre_file = f"{os.path.split(self.exp_data_file)[0]}/dic_disp_gd.xdmf"
            result_pre_file = generate_xdmf_result_file(save_pre_file)
            for i, step in enumerate(step_list):
                predict_func = result[i][0]
                gd_func = self.gradient_decent(predict_func, step)
                result[i][0].vector()[:] = gd_func.vector()[:]
                result_pre_file.write_checkpoint(gd_func, "dic_disp_gd", load_list[i], append=(i != 0))

        if save_result:
            result_file = generate_xdmf_result_file(save_file)
            for i, l in enumerate(load_list):
                result_file.write_checkpoint(result[i][0], func_name, l, append=(i != 0))

        return result, load_list



    def load_loads(self):
        load = pd.read_excel(self.load_file, usecols=(0, 2, 3)).values
        # N --> KN
        force_load = (load[:, 1] - load[0, 1]) * self.force_load_scale
        disp_load = load[:, 2] - load[0, 2]
        first_frame_of_force = int(load[0, 0] - 1)
        return force_load, first_frame_of_force, disp_load

    def data_unpad(self, data_ref):
        row_min, column_min, row_max, column_max = \
            self.ROI_index_range
        data_unpaded = data_ref[row_min:row_max+1, column_min:column_max+1]
        return data_unpaded

    def load_mat(self, dic_step, key="plot_u_ref_formatted"):
        return self.disp[key][dic_step].T

    def calculate_neighbor_index(self, coor_x, coor_y):
        i_min, j_min, i_max, j_max = \
            self.ROI_index_range
        i_max -= i_min
        j_max -= j_min
        # i_min, j_min = 0, 0
        disp_interval = self.pixel*4
        mesh_size = self.right_boundary/20
        pad = round(1.8*mesh_size/disp_interval)
        row = round(coor_y/disp_interval)
        column = round(coor_x/disp_interval)
        row_min = row - pad
        row_max = row + pad
        column_min = column - pad
        column_max = column + pad

        row_min = max(row_min, 0)
        row_max = min(row_max, i_max)
        column_min = max(column_min, 0)
        column_max = min(column_max, j_max)
        return row_min, row_max, column_min, column_max

    def generate_dof_neighbor(self):
        neighbor = {}
        for i in range(self.V.dim()):
            coor_x, coor_y = self.dof_coordinate[i]
            row_min, row_max, column_min, column_max = \
                self.calculate_neighbor_index(coor_x, coor_y)
            neighbor[i] = [np.arange(row_min, row_max+1, 1, dtype=int),
                           np.arange(column_min, column_max+1, 1, dtype=int)]
        return neighbor

    def Loss(self, predict_func, dic_step):
        dic_u = self.data_unpad(self.load_mat(dic_step, key="plot_u_ref_formatted"))
        dic_v = self.data_unpad(self.load_mat(dic_step, key="plot_v_ref_formatted"))
        row, column = dic_u.shape
        total_loss = 0
        count = 0
        i = 0
        for r in range(row):
            for c in range(column):
                coor_x = c*4*self.pixel
                coor_y = r*4*self.pixel
                try:
                    pred_u, pred_v = predict_func(coor_x, coor_y)
                    if dic_u[r, c] != 0:
                        total_loss += (pred_u - dic_u[r, c])**2
                        count += 1
                    if dic_v[r, c] != 0:
                        total_loss += (pred_v - dic_v[r, c])**2
                        count += 1
                except:
                    pass
        average_loss = total_loss/count
        # print(average_loss)
        return average_loss

    def calculate_gradient(self, predict_func, dic_u, dic_v):
        row, column = dic_u.shape
        shape_func = Function(self.V)
        grad = np.zeros_like(predict_func.vector()[:])

        for ui in self.u_index:
            grad_ui = 0
            shape_func.vector()[:] = 0
            shape_func.vector()[ui] = 1
            for r in self.dof_neighbor[ui][0]:
                for c in self.dof_neighbor[ui][1]:
                    coor_x = c * 4 * self.pixel
                    coor_y = r * 4 * self.pixel
                    if dic_u[r, c] == 0:
                        continue
                    try:
                        pred_u, pred_v = predict_func(coor_x, coor_y)
                        shape_u, shape_v = shape_func(coor_x, coor_y)
                        grad_ui += 2 * (pred_u - dic_u[r, c]) * shape_u
                    except:
                        continue
            grad[ui] = grad_ui

        for vi in self.v_index:
            grad_vi = 0
            shape_func.vector()[:] = 0
            shape_func.vector()[vi] = 1
            for r in self.dof_neighbor[vi][0]:
                for c in self.dof_neighbor[vi][1]:
                    coor_x = c * 4 * self.pixel
                    coor_y = r * 4 * self.pixel
                    if dic_v[r, c] == 0:
                        continue
                    try:
                        pred_u, pred_v = predict_func(coor_x, coor_y)
                        shape_u, shape_v = shape_func(coor_x, coor_y)
                        grad_vi += 2 * (pred_v - dic_v[r, c]) * shape_v
                    except:
                        continue
            grad[vi] = grad_vi
        return grad

    def gradient_decent(self, predict_func, dic_step, max_step=100, lr=0.2):
        dic_u = self.data_unpad(self.load_mat(dic_step, key="plot_u_ref_formatted"))
        dic_v = self.data_unpad(self.load_mat(dic_step, key="plot_v_ref_formatted"))
        result_func = Function(self.V)

        loss_previous = self.Loss(predict_func, dic_step)
        convergence_count = 0
        loss_min = loss_previous
        min_step = 0
        result_func.vector()[:] = predict_func.vector()[:]

        log_file = f"{os.path.split(self.exp_data_file)[0]}/{dic_step}_GD_log.txt"
        print(f"**********step {dic_step}**********")
        self.write_log(f"Step Loss Save Time", filename=log_file, clear=True)
        self.write_log(f"{0} {loss_previous} {1} "
                       f"{time.time()-self.time_start}", filename=log_file)

        for i in range(max_step):
            grad = self.calculate_gradient(predict_func, dic_u, dic_v)
            predict_func.vector()[:] -= lr*grad
            loss_current = self.Loss(predict_func, dic_step)

            # record optimal results
            if loss_current < loss_min:
                loss_min = loss_current
                min_step = i+1
                result_func.vector()[:] = predict_func.vector()[:]
                self.write_log(f"{i+1} {loss_current} {1} "
                               f"{time.time()-self.time_start}", filename=log_file)
            else:
                self.write_log(f"{i + 1} {loss_current} {0} "
                               f"{time.time()-self.time_start}", filename=log_file)

            # convergence criterion
            if loss_current < 1e-8:
                self.write_log(f"convergence: loss < 1e-8", filename=log_file)
                self.write_log(f"result: step {min_step} loss {loss_min}", filename=log_file)
                break
            loss_decent = loss_previous - loss_current
            if loss_decent < 0:
                lr = 0.9*lr
            elif 0 <= loss_decent < 0.5e-8:
                convergence_count += 1
                if convergence_count >=3:
                    self.write_log(f"convergence: loss_decent < 0.5e-8", filename=log_file)
                    self.write_log(f"result: step {min_step} loss {loss_min}", filename=log_file)
                    break
            elif loss_decent >= 0.5e-8:
                convergence_count = 0
            loss_previous = loss_current

        if i+1 == max_step:
            self.write_log(f"maximum iteration reached", filename=log_file)
            self.write_log(f"result: step {min_step} loss {loss_min}", filename=log_file)

        return result_func


    def write_log(self, info="", filename=None, clear=False, print_out=True, end="\n"):
        if filename is None:
            file_path, _ = os.path.split(self.exp_data_file)
            filename = f"{file_path}/log.txt"
        with open(filename, "a") as f:
            if clear:
                f.truncate(0)
            f.write(f"{info}{end}")
        if print_out:
            print(info)


if __name__ == "__main__":
    kwargs_file = "../results/DIC1113CIRCLE_1426_CG2_Pre_EXPDispBC_plane_stress_test4_BFGS_skip5_sp_loss2_3/"
    kwargs = np.load(f"{kwargs_file}kwargs.npy", allow_pickle=True).item()

    case = NNHyperEXPDispBCPlaneStress(kwargs, save_kwargs=False)

    case.exp2func_kwargs["preprocess_file"] = "../DIC1113/CIRCLE_1426/dic_disp_gd.xdmf"
    test = Exp2Func_GD(case.V, case.exp2func_kwargs)

    result, load = test.exp2func_gd(case.exp_train_steps, save_result=False, bc_type=case.bc_type)
    print(test.Loss(result[-1][0], 423) )




