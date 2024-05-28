from dolfin import *
from dolfin_adjoint import *

from constitutive_net import ConstitutiveNet, generate_consti_net
from exp_data2funcGD import Exp2Func_GD
from general_problem import GeneralCase
from utils import generate_xdmf_result_file, Logger
import constitutive_ground_truth as consti_gt

import numpy as np
import os
import sys
import time
import traceback

parameters["form_compiler"]["quadrature_degree"] = 4
# set_log_level(LogLevel.CRITICAL)


class NNHyperPlaneStress(GeneralCase):
    def __init__(self, kwargs, save_kwargs=True):
        super(NNHyperPlaneStress, self).__init__(kwargs)
        self.kwargs = kwargs
        self.case_name = self.kwargs["case_name"]
        self.mesh_file = self.kwargs["mesh_file"]
        self.post_file = self.kwargs["post_file"]
        if not os.path.exists(self.post_file):
            os.makedirs(self.post_file)
        if save_kwargs:
            np.save(self.post_file + f"kwargs.npy", self.kwargs)  # save paras for post process
        self.element_type = self.kwargs["element_type"]
        self.element_degree = self.kwargs["element_degree"]
        self.geometry_tol = self.kwargs.get("geometry_tol", 1e-6)
        self.E = self.kwargs.get("E", 100)
        self.nu = self.kwargs.get("nu", 0.3)
        self.ground_truth_consti_tag = self.kwargs.get("ground_truth_consti")
        self.ground_truth_consti = consti_gt.get_constitutive(self.ground_truth_consti_tag)
        self.exp_tag = self.kwargs.get("exp_tag", False)
        if self.ground_truth_consti is not None:
            assert self.exp_tag == False
            self.nn_train_list = self.kwargs.get("ground_truth_load_list",
                                                          np.linspace(0, 0.2, 3))
            self.loss_step = self.kwargs.get("loss_step",
                                             np.arange(len(self.nn_train_list)))
            self.evaluate_load_list = self.kwargs.get("evaluate_load_list",
                                                      np.linspace(0, 0.9, 10))
        else:
            assert self.exp_tag == True
            # if the ground truth is read from experiment
            # the self.nn_train_list is based on the experiment step
            # and is currently determined during generating the experiment ground truth
            self.exp2func_kwargs = self.kwargs.get("exp2func_kwargs")
            self.exp_train_steps = self.kwargs.get("exp_train_steps")
            self.loss_step = self.kwargs.get("loss_step",
                                             np.arange(len(self.exp_train_steps)))
            self.nn_train_list = None
            self.exp_evaluate_steps = self.kwargs.get("exp_evaluate_steps")
            self.evaluate_load_list = None

        self.ground_truth = None
        self.loss = None
        self.ux_scale = self.kwargs.get("ux_scale_in_loss", 0.2)
        self.uy_scale = self.kwargs.get("uy_scale_in_loss", 0.1)
        self.force_in_loss = self.kwargs.get("force_in_loss", False)
        self.weight_decay_lambda = self.kwargs.get("weight_decay_lambda", 0)
        self.bc_type = self.kwargs.get("bc_type", "force")

        # initial the constitutive nn
        self.nn_consti_kwargs = self.kwargs["nn_consti_kwargs"]
        self.nn_consti = generate_consti_net(**self.nn_consti_kwargs)
        # the number of evaluate loss ... during training
        self.train_callback_count = 0

        self.optimisation_method = self.kwargs.get("optimisation_method", "CG")

        # mesh functionspace and exp data
        self.read_mesh()
        element_u = VectorElement(self.element_type, self.mesh.ufl_cell(), self.element_degree)
        element_lambda = FiniteElement("DG", self.mesh.ufl_cell(), 0) \
            if self.element_degree - 1 == 0 else \
            FiniteElement(self.element_type, self.mesh.ufl_cell(), self.element_degree - 1)
        element_F33 = FiniteElement("DG", self.mesh.ufl_cell(), 0) \
            if self.element_degree - 1 == 0 else \
            FiniteElement(self.element_type, self.mesh.ufl_cell(), self.element_degree - 1)
        TH = MixedElement([element_u, element_lambda, element_F33])
        self.W = FunctionSpace(self.mesh, TH)
        self.V = self.W.sub(0).collapse()
        if self.exp_tag:
            self.exp2func_gd = Exp2Func_GD(self.V, self.exp2func_kwargs)

        self.time_initial = time.time()

    def define_bcs(self):
        dof_coordinates = self.V.tabulate_dof_coordinates()
        x_min = min(dof_coordinates[:, 0])
        y_min = min(dof_coordinates[:, 1])
        # print(f"corner: ({x_min}, {y_min})")
        def corner(x, on_boundary):
            return bool(near(x[0], x_min, self.geometry_tol) and near(x[1], y_min, self.geometry_tol))
        bc_corner = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), corner, method='pointwise')
        bc_bottom = DirichletBC(self.W.sub(0).sub(1), Constant(0.0), self.facet_regions, 3)
        self.bcs = [bc_corner, bc_bottom]

    def define_problem(self, constitutive, only_form=False):
        self.w = Function(self.W)
        # can not use w.split() to generate potential energy and weak form
        self.u, self.lamda, self.F33 = split(self.w)
        self.u_func, self.lamda_func, self.F33_func = self.w.split()
        self.F33 = variable(self.F33)
        w_t = TestFunction(self.W)
        dw = TrialFunction(self.W)
        self.F = grad(self.u) + Identity(self.dim)
        self.F = variable(self.F)
        self.C = self.F.T * self.F
        C33 = self.F33**2
        I1_prime = tr(self.C)
        I2_prime = ( I1_prime**2 - tr(self.C * self.C) )/2
        J_prime = det(self.F)
        self.I_1 = I1_prime + C33
        self.I_2 = I2_prime + C33*I1_prime
        self.J = self.F33*J_prime
        self.psi = constitutive(self.I_1, self.I_2, self.J, E=self.E, nu=self.nu, C=self.C)
        self.pk = diff(self.psi, self.F)
        self.pk33 = diff(self.psi, self.F33)
        # define variables for solving linear system
        epsilon = sym(grad(self.u))
        epsilon33 = variable(self.F33 - 1)
        theta = tr(epsilon) + epsilon33
        psi_linear = consti_gt.linear_elastic(theta, epsilon, epsilon33, E=self.E, nu=self.nu)
        sigma33 = diff(psi_linear, epsilon33)
        potential_energy_linear = (psi_linear + self.lamda * sigma33) * self.dx
        self.Form_linear = derivative(potential_energy_linear, self.w, w_t)

        # nolinear system
        self.bc_load = Constant(0.)    # fy on the top boundary
        self.strain_energy = (self.psi + self.lamda*self.pk33)*self.dx
        self.external_work = self.bc_load * self.u[1] * self.ds(4)
        self.potential_energy = self.strain_energy - self.external_work
        self.Form = derivative(self.potential_energy, self.w, w_t)

        if not only_form:
            Jac = derivative(self.Form, self.w, dw)
            self.problem = NonlinearVariationalProblem(self.Form, self.w, self.bcs, Jac)
            self.solver = NonlinearVariationalSolver(self.problem)
            prm = self.solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = 1E-8
            prm["newton_solver"]["relative_tolerance"] = 1E-7
            # prm["newton_solver"]["maximum_iterations"] = 20
            prm["newton_solver"]["linear_solver"] = "mumps"

    def solve(self, constitutive, func_name, load_list,
              loss_function=None, save_result=True, save_loss=False):
        self.define_bcs()
        self.define_problem(constitutive)
        # solve linear system to generate appropriate initial value
        print("********* solving linear problem **********")
        solve(self.Form_linear == 0, self.w, self.bcs)
        print("********* linear problem completed **********")
        if save_result:
            result_file_w = HDF5File(MPI.comm_world, self.post_file+f"{func_name}_w.hdf5", "w")
            result_file_disp = generate_xdmf_result_file(self.post_file+f"{func_name}_disp.xdmf")
            result_file_lambda = generate_xdmf_result_file(self.post_file + f"{func_name}_lambda.xdmf")
            result_file_F33 = generate_xdmf_result_file(self.post_file + f"{func_name}_F33.xdmf")
        results = []    # (disp_field, _x on the right boundary)
        loss = 0
        psi_list = np.zeros_like(load_list)
        for i, load in enumerate(load_list):
            # print("load:", load, load*13.328)
            self.bc_load.assign(load)
            self.solver.solve()
            psi_list[i] = assemble(self.psi * self.dx)
            if save_result:
                result_file_w.write(self.w, f"{load}")
                result_file_disp.write_checkpoint(self.u_func, f"{func_name}_disp", load, append=(i != 0))
                result_file_lambda.write_checkpoint(self.lamda_func, f"{func_name}_lambda", load, append=(i != 0))
                result_file_F33.write_checkpoint(self.F33_func, f"{func_name}_F33", load, append=(i != 0))
            fem_assembled_right_force_x = \
                assemble(dot(self.pk, self.normal_vector)[0]*self.ds(2))
            # print("assemble load: ", fem_assembled_right_force_x)
            results.append((interpolate(self.u_func, self.V), fem_assembled_right_force_x))
            if loss_function is not None:
                loss += loss_function(i, save_loss=save_loss)
            # check shear locking
            # print(assemble(self.potential_energy))
        if loss_function is None:
            loss = None
        return results, loss, psi_list

    def generate_ground_truth(self):
        '''
        ground truth comes from FEM, or experiment
        '''
        if self.ground_truth_consti is not None:
            # generate ground truth from ground truth constitutive
            ground_truth_name = "fem_train_truth"
            with stop_annotating():
                self.ground_truth, _, _ = self.solve(self.ground_truth_consti, ground_truth_name,
                                                  self.nn_train_list)

                # DIC_step = 0.0392  # mm
                # noise_pixels = 2  # the number of pixels with bias
                # noise_ground_truth_file = \
                #     generate_xdmf_result_file(self.post_file +
                #                               f"{ground_truth_name}_disp_{noise_pixels}_pixels_deviation.xdmf")
                # i = 0
                # for disp, _ in self.ground_truth:
                #     disp.vector()[:] = disp.vector()[:] + \
                #                        noise_pixels * DIC_step * np.random.randn(len(disp.vector()[:]))
                #     noise_ground_truth_file.write_checkpoint(disp,
                #                                              f"{ground_truth_name}_disp_{noise_pixels}_pixels_deviation",
                #                                              self.nn_train_list[i],
                #                                              append=(i != 0))
                #     i += 1
                # # 5/10/20% variance in the relative error of force
                # # self.nn_train_list = self.nn_train_list + \
                # #                      self.nn_train_list * noise / 100 * np.random.randn(len(self.nn_train_list))


        elif self.exp_tag:
            # generate ground truth from experiment
            ground_truth_name = "exp_train_truth_disp"
            with stop_annotating():
                self.ground_truth, self.nn_train_list = \
                    self.exp2func_gd.exp2func_gd(self.exp_train_steps,
                                                 func_name=ground_truth_name,
                                                 save_result=True,
                                                 save_file=f"{self.post_file}{ground_truth_name}.xdmf",
                                                 bc_type=self.bc_type)
        # save self.nn_train_list for post process
        np.savetxt(f"{self.post_file}nn_train_load_list.txt", self.nn_train_list)

    def generate_loss(self, load_list=None,
                      func_name = "nn_init",
                      save_result=False,
                      save_loss=False):
        '''
        forward compute
        generate the computing map
        '''
        if not self.ground_truth:
            self.generate_ground_truth()
        if load_list is None:
            load_list = self.nn_train_list
        result, self.loss, psi_list = self.solve(self.nn_consti, func_name, load_list,
                                                 loss_function=self.loss_function,
                                                 save_result=save_result,
                                                 save_loss=save_loss)
        self.loss += assemble(self.nn_consti.weight_decay_term(decay_lambda=self.weight_decay_lambda) * self.dx)/assemble(1*self.dx)
        print("weight decay term: ",
              assemble(self.nn_consti.weight_decay_term(decay_lambda=self.weight_decay_lambda) * self.dx)/assemble(1*self.dx))
        print("total loss: ", self.loss)
        return result, psi_list

    # def loss_function(self, i):
    #     if not i in self.loss_step:
    #         return 0
    #     else:
    #         disp_square = assemble(inner(self.ground_truth[i][0], self.ground_truth[i][0])*self.dx(1))
    #         i_temp = i
    #         while near(disp_square, 0, eps=1e-8):
    #             print(f"disp square is 0 at {i_temp}")
    #             i_temp += 1
    #             disp_square = assemble(inner(self.ground_truth[i_temp][0], self.ground_truth[i_temp][0])*self.dx(1))
    #         disp_normalize = disp_square
    #
    #         force_normalize = abs(self.ground_truth[i][1])
    #         i_temp = i
    #         while near(force_normalize, 0, eps=1e-4):
    #             print(f"force normalize is 0 at {i_temp}")
    #             i_temp += 1
    #             force_normalize = abs(self.ground_truth[i_temp][1])
    #
    #         error = self.u_func - self.ground_truth[i][0]
    #         label_area = assemble(1*self.dx(1))
    #         label_length = assemble(1 * self.ds(2))
    #
    #         disp_loss_weight = 1
    #         force_loss_weight = disp_loss_weight / 4 if self.force_in_loss else 0
    #         disp_loss_weight /= (disp_loss_weight + force_loss_weight)
    #         force_loss_weight = 1 - disp_loss_weight
    #         # force_loss_weight = 1 / label_length**2
    #         # disp_loss_weight = 1 - force_loss_weight
    #         if i == 0:
    #             print("loss weights in disp and force:",
    #                   disp_loss_weight, force_loss_weight)
    #
    #         loss = disp_loss_weight * assemble(inner(error, error) * self.dx(1))/disp_normalize + \
    #                force_loss_weight * (assemble((dot(self.pk, self.normal_vector)[0] -
    #                                               self.ground_truth[i][1]/label_length)*self.ds(2))/force_normalize)**2
    #         disp_loss = assemble(inner(error, error) * self.dx(1))/disp_normalize
    #         force_loss = (assemble((dot(self.pk, self.normal_vector)[0] -
    #                                               self.ground_truth[i][1]/label_length)*self.ds(2))/force_normalize)**2
    #         # print(f"weighted scaled loss of step {i} (disp loss, force loss): ",
    #         #       disp_loss_weight * disp_loss, force_loss_weight * force_loss)
    #         print(f"unweighted scaled loss of step {i} (disp loss, force loss): ",
    #               disp_loss, force_loss)
    #         return loss

    def loss_function(self, i, save_loss=False):
        # loss 1 disp:force = 4:1
        # loss 2 disp:force = 1:1
        # loss 4 disp:force = 1:1, normalized by ux_scale
        if not i in self.loss_step:
            return 0
        else:
            error = self.u_func - self.ground_truth[i][0]
            label_area = assemble(1*self.dx(1))
            label_length = assemble(1 * self.ds(2))

            disp_normalize = max((self.ground_truth[-1][0].vector()[:]) ** 2) * label_area
            # disp_normalize = (self.ux_scale ** 2) * label_area
            force_normalize = self.ground_truth[-1][1]

            disp_loss_weight = 1
            force_loss_weight = disp_loss_weight / 1 if self.force_in_loss else 0
            disp_loss_weight /= (disp_loss_weight + force_loss_weight)
            force_loss_weight = 1 - disp_loss_weight
            # force_loss_weight = 1 / label_length**2
            # disp_loss_weight = 1 - force_loss_weight
            if i == 0:
                print("loss weights in disp and force:",
                      disp_loss_weight, force_loss_weight)

            loss = disp_loss_weight * assemble(inner(error, error) * self.dx(1))/disp_normalize + \
                   force_loss_weight * (assemble((dot(self.pk, self.normal_vector)[0] -
                                                  self.ground_truth[i][1]/label_length)*self.ds(2))/force_normalize)**2
            disp_loss = assemble(inner(error, error) * self.dx(1))/disp_normalize
            force_loss = (assemble((dot(self.pk, self.normal_vector)[0] -
                                                  self.ground_truth[i][1]/label_length)*self.ds(2))/force_normalize)**2
            # print(f"weighted scaled loss of step {i} (disp loss, force loss): ",
            #       disp_loss_weight * disp_loss, force_loss_weight * force_loss)
            print(f"unweighted scaled loss of step {i} (disp loss, force loss): ",
                  disp_loss, force_loss)
            if save_loss:
                info = ""
                if i == 0:
                    info += f"Step DisplacementLoss ForceLoss\n"
                info += f"{i:.0f} {disp_loss} {force_loss}"
                self.write_log(info, filename=f"{self.post_file}unweighted_loss_record.txt", clear=(i == 0))
            return loss

    # def loss_function(self, i):
    #     # loss 3: divide u_x and u_y
    #     if not i in self.loss_step:
    #         return 0
    #     else:
    #         ux_error = self.u_func[0] - self.ground_truth[i][0][0]
    #         uy_error = self.u_func[1] - self.ground_truth[i][0][1]
    #         label_area = assemble(1 * self.dx(1))
    #         label_length = assemble(1 * self.ds(2))
    #         ux_normalize = (self.ux_scale ** 2) * label_area
    #         uy_normalize = (self.uy_scale ** 2) * label_area
    #         force_normalize = self.ground_truth[-1][1]
    #
    #         disp_loss_weight = 1
    #         force_loss_weight = disp_loss_weight / 1 if self.force_in_loss else 0
    #         disp_loss_weight /= (disp_loss_weight + force_loss_weight)
    #         force_loss_weight = 1 - disp_loss_weight
    #         if i == 0:
    #             print("loss weights in disp and force:",
    #                   disp_loss_weight, force_loss_weight)
    #
    #         loss = disp_loss_weight * (assemble(ux_error**2*self.dx(1))/ux_normalize +
    #                                     assemble(uy_error**2*self.dx(1))/uy_normalize) + \
    #                force_loss_weight * (assemble((dot(self.pk, self.normal_vector)[0] -
    #                                               self.ground_truth[i][1] / label_length) * self.ds(2)) /
    #                                     force_normalize) ** 2
    #
    #         disp_loss = assemble(ux_error**2*self.dx(1))/ux_normalize + \
    #                     assemble(uy_error**2*self.dx(1))/uy_normalize
    #         force_loss = (assemble((dot(self.pk, self.normal_vector)[0] -
    #                                 self.ground_truth[i][1] / label_length) * self.ds(2)) /
    #                       force_normalize) ** 2
    #         print(f"unweighted scaled loss of step {i} (disp loss, force loss): ",
    #               disp_loss, force_loss)
    #     return loss

    def trian_consti_nn(self):
        if self.loss is None:
            self.generate_loss()
        self.J_hat = ReducedFunctional(self.loss, self.nn_consti.design_variable_ctrls(),
                                  eval_cb_post=self.callback_of_J_hat)
        self.ctrl_state = {"ctrl_values": []}

        try:
            self.opt_consti_nn_weight = minimize(self.J_hat, method=self.optimisation_method, tol=1e-10,
                                                 options={"maxiter": 100, "gtol": 1e-10})
        except:
            filename = self.post_file + f"training_loss_log.txt"
            self.write_log(traceback.format_exc(), filename=filename)
            self.opt_consti_nn_weight = self.ctrl_state["ctrl_values"]

        self.nn_consti.set_design_variable(self.opt_consti_nn_weight)
        self.nn_consti.save()
        self.nn_consti.get_design_variables()

    def callback_of_J_hat(self, loss, *args):
        '''
        the function is used to record loss value during the optimization process
        '''
        filename = self.post_file+f"training_loss_log.txt"
        info = ""
        if self.train_callback_count == 0:
            # title
            info += "Callback Loss Time(s)\n"
        current_time = time.time()
        info += f"{self.train_callback_count:.0f} {loss} {current_time-self.time_initial}"
        self.write_log(info, filename=filename,
                       clear=(self.train_callback_count == 0))
        # backup result
        r = []
        controls = self.J_hat.controls
        for ctrl in controls:
            r.append(ctrl.tape_value()._ad_create_checkpoint())
        self.ctrl_state["ctrl_values"] = r
        # controls = self.J_hat.controls
        # for ctrl in controls[:1]:
        #     # get current iteration results
        #     print(ctrl.tape_value()._ad_create_checkpoint().values())
        self.train_callback_count += 1

    def evaluate_consti_nn(self):
        if self.ground_truth_consti is not None:
            # If the ground truth is generated by FEM calculation,
            # generate corresponding FEM result for contrast
            evaluate_truth_name = "fem_evaluate_truth"
            self.solve(self.ground_truth_consti, evaluate_truth_name, self.evaluate_load_list)
        elif self.exp_tag:
            evaluate_truth_name = "exp_evaluate_truth_disp"
            _, self.evaluate_load_list = \
                self.exp2func_gd.exp2func_gd(self.exp_evaluate_steps,
                                             func_name=evaluate_truth_name,
                                             save_result=True,
                                             save_file=f"{self.post_file}{evaluate_truth_name}.xdmf",
                                             bc_type=self.bc_type)
        # save self.evaluate_load_list for post process
        np.savetxt(f"{self.post_file}nn_evaluate_load_list.txt", self.evaluate_load_list)

        evaluate_consti_nn_name = "evaluate_consti_nn"
        self.solve(self.nn_consti, evaluate_consti_nn_name, self.evaluate_load_list)

    def employment_consti_nn(self, employment_step, save_data=True, save_loss=True):
        # generate employment loss
        # generate force-disp curve
        if self.ground_truth_consti is not None:
            # generate employment truth from ground truth constitutive
            employment_list = employment_step
            employment_truth_name = "fem_employment_truth"
            with stop_annotating():
                self.ground_truth, _, psi_truth = self.solve(self.ground_truth_consti, employment_truth_name,
                                                  employment_step, save_result=save_data)
            if save_data:
                np.savetxt(f"{self.post_file}employment_truth_{self.bc_type}_psi_data.txt",
                           np.concatenate((employment_list.reshape(-1, 1),
                                           psi_truth.reshape(-1, 1)), axis=1))
        elif self.exp_tag:
            # generate employment truth from experiment
            employment_truth_name = "exp_employment_truth_disp"
            with stop_annotating():
                self.ground_truth, employment_list = \
                    self.exp2func_gd.exp2func_gd(employment_step,
                                                 func_name=employment_truth_name,
                                                 save_result=save_data,
                                                 save_file=f"{self.post_file}{employment_truth_name}.xdmf",
                                                 bc_type=self.bc_type)
        # print employment loss and generate the boundary force
        employment_result, employment_psi = self.generate_loss(employment_list,
                                                               func_name="nn_employment",
                                                               save_result=save_data,
                                                               save_loss=save_loss)
        employment_load_force_data = np.zeros((len(employment_list), 2))
        truth_load_force_data = np.zeros((len(employment_list), 2))

        employment_disp_error_file = generate_xdmf_result_file(self.post_file + f"employment_disp_error.xdmf")
        # employment_relative_disp_error_file = \
        #     generate_xdmf_result_file(self.post_file + f"employment_relative_disp_error.xdmf")

        for i in range( len(employment_list) ):
            employment_load_force_data[i, :] = np.array([employment_list[i],
                                                         employment_result[i][-1]])
            truth_load_force_data[i, :] = np.array([employment_list[i],
                                                    self.ground_truth[i][-1]])
            if save_data:
                employment_disp_error_file.write_checkpoint(project(employment_result[i][0]-self.ground_truth[i][0],
                                                                    self.V),
                                                            "employment_disp_error",
                                                            employment_list[i],
                                                            append=(i != 0))
                # disp_error = employment_result[i][0]-self.ground_truth[i][0]
                # disp_truth = self.ground_truth[i][0]
                # relative_displacement_error = sqrt(dot(disp_error, disp_error))/(sqrt(dot(disp_truth, disp_truth)) + 1e-1)
                # employment_relative_disp_error_file.write_checkpoint(project(relative_displacement_error,
                #                                                              self.V.sub(0).collapse()),
                #                                                      "employment_relative_disp_error",
                #                                                      employment_list[i],
                #                                                      append=(i != 0))

        if save_data:
            np.savetxt(f"{self.post_file}employment_truth_{self.bc_type}_force_data.txt",
                       truth_load_force_data)
            np.savetxt(f"{self.post_file}employment_nn_{self.bc_type}_force_data.txt",
                       employment_load_force_data)
            np.savetxt(f"{self.post_file}employment_nn_{self.bc_type}_psi_data.txt",
                       np.concatenate((employment_list.reshape(-1, 1),
                                       employment_psi.reshape(-1, 1)), axis=1))
        self.ground_truth = None
        self.loss = None
        return truth_load_force_data, employment_load_force_data

    def post_pk_stress(self, constitutive, file_name, load_list,
                       pk_stress_name="pk_stress", save_file_name=None):
        self.define_problem(constitutive, only_form=True)
        pk_func_space = TensorFunctionSpace(self.mesh, "DG", 0) if self.element_degree -1 == 0 \
            else TensorFunctionSpace(self.mesh, self.element_type, self.element_degree - 1)
        file_read = HDF5File(MPI.comm_world, file_name, "r")
        if not save_file_name:
            save_file_name = self.post_file+f"{pk_stress_name}.xdmf"
        file_save = generate_xdmf_result_file(save_file_name)
        for i, load in enumerate(load_list):
            file_read.read(self.w, f"{load}")
            file_save.write_checkpoint(project(self.pk, pk_func_space), pk_stress_name, load, append=(i != 0))
            # print(load, assemble(dot(self.pk, self.normal_vector)[1]*self.ds(4)))

    # def plot_green_strain(self, file_name, load_list,
    #                    green_strain_name="green_strain", save_file_name=None):
    #     w_gs = Function(self.W)
    #     u_gs, _, _ = split(w_gs)
    #     F = grad(u_gs) + Identity(self.dim)
    #     C = F.T * F
    #     gs_func_space = TensorFunctionSpace(self.mesh, "DG", 0) if self.element_degree - 1 == 0 \
    #         else TensorFunctionSpace(self.mesh, self.element_type, self.element_degree - 1)
    #     gs_exp = (C - Identity(self.dim)) / 2
    #     file_read = HDF5File(MPI.comm_world, file_name, "r")
    #     if not save_file_name:
    #         save_file_name = self.post_file+f"{green_strain_name}.xdmf"
    #     file_save = generate_xdmf_result_file(save_file_name)
    #     for i, load in enumerate(load_list):
    #         file_read.read(w_gs, f"{load}")
    #         file_save.write_checkpoint(project(gs_exp, gs_func_space), green_strain_name, load, append=(i != 0))

    def plot_green_strain(self, file_name_truth, file_name_nn, load_list,
                          green_strain_name="green_strain", fun_name_truth=None):
        gs_func_space = TensorFunctionSpace(self.mesh, "DG", 0) if self.element_degree - 1 == 0 \
            else TensorFunctionSpace(self.mesh, self.element_type, self.element_degree - 1)
        # ground truth
        if self.ground_truth_consti is not None:
            w_gs_truth = Function(self.W)
            u_gs_truth, _, _ = split(w_gs_truth)
            F_truth = grad(u_gs_truth) + Identity(self.dim)
            C_truth = F_truth.T * F_truth
            gs_truth_exp = (C_truth - Identity(self.dim)) / 2
            file_read_truth = HDF5File(MPI.comm_world, file_name_truth, "r")
        elif self.exp_tag:
            u_gs_truth = Function(self.V)
            F_truth = grad(u_gs_truth) + Identity(self.dim)
            C_truth = F_truth.T * F_truth
            gs_truth_exp = (C_truth - Identity(self.dim)) / 2
            file_read_truth = XDMFFile(file_name_truth)
        # nn results
        w_gs_nn = Function(self.W)
        u_gs_nn, _, _ = split(w_gs_nn)
        F_nn = grad(u_gs_nn) + Identity(self.dim)
        C_nn = F_nn.T * F_nn
        gs_nn_exp = (C_nn - Identity(self.dim)) / 2
        file_read_nn = HDF5File(MPI.comm_world, file_name_nn, "r")

        gs_error_exp = gs_nn_exp - gs_truth_exp

        save_file_truth = self.post_file + f"{green_strain_name}_truth.xdmf"
        save_file_nn = self.post_file + f"{green_strain_name}_nn.xdmf"
        save_file_error = self.post_file + f"{green_strain_name}_error.xdmf"
        file_save_truth = generate_xdmf_result_file(save_file_truth)
        file_save_nn = generate_xdmf_result_file(save_file_nn)
        file_save_error = generate_xdmf_result_file(save_file_error)

        for i, load in enumerate(load_list):
            if self.ground_truth_consti is not None:
                file_read_truth.read(w_gs_truth, f"{load}")
            elif self.exp_tag:
                file_read_truth.read_checkpoint(u_gs_truth, fun_name_truth, i)
            file_read_nn.read(w_gs_nn, f"{load}")

            file_save_truth.write_checkpoint(project(gs_truth_exp, gs_func_space),
                                             f"{green_strain_name}_truth", load, append=(i != 0))
            file_save_nn.write_checkpoint(project(gs_nn_exp, gs_func_space),
                                          f"{green_strain_name}_nn", load, append=(i != 0))
            file_save_error.write_checkpoint(project(gs_error_exp, gs_func_space),
                                             f"{green_strain_name}_error", load, append=(i != 0))

    def plot_almansi_strain(self, file_name, load_list,
                       almansi_strain_name="almansi_strain", save_file_name=None):
        w_as = Function(self.W)
        u_as, _, _ = split(w_as)
        F = grad(u_as) + Identity(self.dim)
        B = F*F.T
        as_func_space = TensorFunctionSpace(self.mesh, "DG", 0) if self.element_degree - 1 == 0 \
            else TensorFunctionSpace(self.mesh, self.element_type, self.element_degree - 1)
        as_exp = (Identity(self.dim)-inv(B)) / 2
        file_read = HDF5File(MPI.comm_world, file_name, "r")
        if not save_file_name:
            save_file_name = self.post_file+f"{almansi_strain_name}.xdmf"
        file_save = generate_xdmf_result_file(save_file_name)
        for i, load in enumerate(load_list):
            file_read.read(w_as, f"{load}")
            file_save.write_checkpoint(project(as_exp, as_func_space), almansi_strain_name, load, append=(i != 0))

    def principal_stretch(self, file_name, load_list):
        '''
        save the contour and values of the principle stretches
        '''
        w = Function(self.W)
        u, la, F33 = split(w)
        F = grad(u) + Identity(self.dim)
        C = F.T * F
        C33 = F33 ** 2
        I1_prime = tr(C)
        I2_prime = (I1_prime ** 2 - tr(C * C)) / 2
        principle_V = self.W.sub(2).collapse()

        lamda_1 = sqrt((I1_prime + sqrt(I1_prime**2-4*I2_prime))/2)
        lamda_2 = sqrt((I1_prime - sqrt(I1_prime ** 2 - 4 * I2_prime)) / 2)
        lamda_3 = F33    #not always the minimum

        lamda_1_file = generate_xdmf_result_file(self.post_file+f"lambda1.xdmf")
        lamda_2_file = generate_xdmf_result_file(self.post_file + f"lambda2.xdmf")
        lamda_1_value = {}
        lamda_2_value = {}
        lamda_3_value = {}

        file_read = HDF5File(MPI.comm_world, file_name, "r")
        for i, load in enumerate(load_list):
            file_read.read(w, f"{load}")
            lamda_1_func = project(lamda_1, principle_V)
            lamda_2_func = project(lamda_2, principle_V)
            lamda_3_func = project(lamda_3, principle_V)
            lamda_1_value[load] = lamda_1_func.vector()[:]
            lamda_2_value[load] = lamda_2_func.vector()[:]
            lamda_3_value[load] = lamda_3_func.vector()[:]
            lamda_1_file.write_checkpoint(lamda_1_func, "lambda1", load, append=(i != 0))
            lamda_2_file.write_checkpoint(lamda_2_func, "lambda2", load, append=(i != 0))
            print(f"{i} {load} complete")
        np.save(self.post_file + f"lambda1.npy", lamda_1_value)
        np.save(self.post_file + f"lambda2.npy", lamda_2_value)
        np.save(self.post_file + f"lambda3.npy", lamda_3_value)

    def error_analysis(self, truth_file, prediction_file, load_list):
        '''
        save the absolute error ||u-u*|| and the relative error
        '''
        w_t = Function(self.W)
        u_t, _, _ = split(w_t)
        file_read_t = HDF5File(MPI.comm_world, truth_file, "r")
        w_p = Function(self.W)
        u_p, _, _ = split(w_p)
        file_read_p = HDF5File(MPI.comm_world, prediction_file, "r")

        abs = {}
        rel = {}
        tru = {}
        for i, load in enumerate(load_list):
            file_read_t.read(w_t, f"{load}")
            file_read_p.read(w_p, f"{load}")
            u_e = u_p - u_t
            u_abs = sqrt(dot(u_e, u_e))
            u_rel = sqrt(dot(u_e, u_e))/(sqrt(dot(u_t, u_t))+1e-9)
            u_tru = sqrt(dot(u_t, u_t))
            u_abs_func = project(u_abs, self.V.sub(0).collapse())
            u_rel_func = project(u_rel, self.V.sub(0).collapse())
            u_tru_func = project(u_tru, self.V.sub(0).collapse())
            abs[load] = u_abs_func.vector()[:]
            rel[load] = u_rel_func.vector()[:]
            tru[load] = u_tru_func.vector()[:]
        np.save(self.post_file + f"absolute_error.npy", abs)
        np.save(self.post_file + f"relative_error.npy", rel)
        np.save(self.post_file + f"truth.npy", tru)


class NNHyperDispBCPlaneStress(NNHyperPlaneStress):

    def __init__(self, kwargs, save_kwargs=True):
        super(NNHyperDispBCPlaneStress, self).__init__(kwargs, save_kwargs)
        self.force_in_loss = self.kwargs.get("force_in_loss", True)
        self.bc_type = self.kwargs.get("bc_type", "disp")

    def define_bcs(self):
        dof_coordinates = self.V.tabulate_dof_coordinates()
        x_min = min(dof_coordinates[:, 0])
        y_min = min(dof_coordinates[:, 1])
        # print(f"corner: ({x_min}, {y_min})")
        def corner(x, on_boundary):
            return bool(near(x[0], x_min, self.geometry_tol) and near(x[1], y_min, self.geometry_tol))
        bc_corner = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), corner, method='pointwise')
        bc_left = DirichletBC(self.W.sub(0).sub(0), Constant(0.0), self.facet_regions, 1)
        self.bc_load = Constant(0.0)
        bc_right = DirichletBC(self.W.sub(0).sub(0), self.bc_load, self.facet_regions, 2)
        self.bcs = [bc_corner, bc_left, bc_right]

    def define_problem(self, constitutive, only_form=False):
        self.w = Function(self.W)
        # can not use w.split() to generate potential energy and weak form
        self.u, self.lamda, self.F33 = split(self.w)
        self.u_func, self.lamda_func, self.F33_func = self.w.split()
        self.F33 = variable(self.F33)
        w_t = TestFunction(self.W)
        dw = TrialFunction(self.W)
        self.F = grad(self.u) + Identity(self.dim)
        self.F = variable(self.F)
        self.C = self.F.T * self.F
        C33 = self.F33 ** 2
        I1_prime = tr(self.C)
        I2_prime = (I1_prime ** 2 - tr(self.C * self.C)) / 2
        J_prime = det(self.F)
        self.I_1 = I1_prime + C33
        self.I_2 = I2_prime + C33 * I1_prime
        self.J = self.F33 * J_prime
        self.psi = constitutive(self.I_1, self.I_2, self.J, E=self.E, nu=self.nu, C=self.C)
        self.pk = diff(self.psi, self.F)
        self.pk33 = diff(self.psi, self.F33)
        # define variables for solving linear system
        epsilon = sym(grad(self.u))
        epsilon33 = variable(self.F33 - 1)
        theta = tr(epsilon) + epsilon33
        psi_linear = consti_gt.linear_elastic(theta, epsilon, epsilon33, E=self.E, nu=self.nu)
        sigma33 = diff(psi_linear, epsilon33)
        potential_energy_linear = (psi_linear + self.lamda * sigma33) * self.dx
        self.Form_linear = derivative(potential_energy_linear, self.w, w_t)

        # nolinear system
        self.strain_energy = (self.psi + self.lamda * self.pk33) * self.dx
        self.external_work = 0
        self.potential_energy = self.strain_energy - self.external_work
        self.Form = derivative(self.potential_energy, self.w, w_t)

        if not only_form:
            Jac = derivative(self.Form, self.w, dw)
            self.problem = NonlinearVariationalProblem(self.Form, self.w, self.bcs, Jac)
            self.solver = NonlinearVariationalSolver(self.problem)
            prm = self.solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = 1E-8
            prm["newton_solver"]["relative_tolerance"] = 1E-7
            # prm["newton_solver"]["maximum_iterations"] = 20
            prm["newton_solver"]["linear_solver"] = "mumps"


class NNHyperEXPDispBCPlaneStress(NNHyperDispBCPlaneStress):

    def define_bcs(self):
        bc_left = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), self.facet_regions, 1)
        self.bc_load = Constant(0.0)
        bc_right_x = DirichletBC(self.W.sub(0).sub(0), self.bc_load, self.facet_regions, 2)
        bc_right_y = DirichletBC(self.W.sub(0).sub(1), Constant(0.0), self.facet_regions, 2)
        self.bcs = [bc_left, bc_right_x, bc_right_y]


class NNHyperForceBCPlaneStress(NNHyperPlaneStress):
    def define_bcs(self):
        dof_coordinates = self.V.tabulate_dof_coordinates()
        x_min = min(dof_coordinates[:, 0])
        y_min = min(dof_coordinates[:, 1])
        # print(f"corner: ({x_min}, {y_min})")
        def corner(x, on_boundary):
            return bool(near(x[0], x_min, self.geometry_tol) and near(x[1], y_min, self.geometry_tol))
        bc_corner = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), corner, method='pointwise')
        bc_left = DirichletBC(self.W.sub(0).sub(0), Constant(0.0), self.facet_regions, 1)
        self.bcs = [bc_corner, bc_left]

    def define_problem(self, constitutive, only_form=False):
        self.w = Function(self.W)
        # can not use w.split() to generate potential energy and weak form
        self.u, self.lamda, self.F33 = split(self.w)
        self.u_func, self.lamda_func, self.F33_func = self.w.split()
        self.F33 = variable(self.F33)
        w_t = TestFunction(self.W)
        dw = TrialFunction(self.W)
        self.F = grad(self.u) + Identity(self.dim)
        self.F = variable(self.F)
        self.C = self.F.T * self.F
        C33 = self.F33 ** 2
        I1_prime = tr(self.C)
        I2_prime = (I1_prime ** 2 - tr(self.C * self.C)) / 2
        J_prime = det(self.F)
        self.I_1 = I1_prime + C33
        self.I_2 = I2_prime + C33 * I1_prime
        self.J = self.F33 * J_prime
        self.psi = constitutive(self.I_1, self.I_2, self.J, E=self.E, nu=self.nu, C=self.C)
        self.pk = diff(self.psi, self.F)
        self.pk33 = diff(self.psi, self.F33)
        # define variables for solving linear system
        epsilon = sym(grad(self.u))
        epsilon33 = variable(self.F33 - 1)
        theta = tr(epsilon) + epsilon33
        psi_linear = consti_gt.linear_elastic(theta, epsilon, epsilon33, E=self.E, nu=self.nu)
        sigma33 = diff(psi_linear, epsilon33)
        potential_energy_linear = (psi_linear + self.lamda * sigma33) * self.dx
        self.Form_linear = derivative(potential_energy_linear, self.w, w_t)

        # nolinear system
        self.bc_load = Constant(0.)  # fy on the top boundary
        self.strain_energy = (self.psi + self.lamda * self.pk33) * self.dx
        self.external_work = self.bc_load * self.u[0] * self.ds(2)
        self.potential_energy = self.strain_energy - self.external_work
        self.Form = derivative(self.potential_energy, self.w, w_t)

        if not only_form:
            Jac = derivative(self.Form, self.w, dw)
            self.problem = NonlinearVariationalProblem(self.Form, self.w, self.bcs, Jac)
            self.solver = NonlinearVariationalSolver(self.problem)
            prm = self.solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = 1E-8
            prm["newton_solver"]["relative_tolerance"] = 1E-7
            # prm["newton_solver"]["maximum_iterations"] = 20
            prm["newton_solver"]["linear_solver"] = "mumps"


class NNHyperEXPForceBCPlaneStress(NNHyperForceBCPlaneStress):
    def define_bcs(self):
        bc_left = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), self.facet_regions, 1)
        self.bcs = [bc_left]


if __name__ == "__main__":
    case_name = "test"
    post_file = "../results/UnitSquare_DispBC_plane_stress_test1_parabolic_lu/"

    # if the net is not trained then consti_nn_state_file  is None
    # consti_nn_state_file = f"{post_file}net_state.pkl"
    consti_nn_state_file = None

    # the load list for generate ground truth and training
    ground_truth_load_list = np.linspace(0, 0.2, 11)
    # np.loadtxt("../results/MR2_DIC_Disp/DIC_train_load.txt")
    # the load list for evaluating the trained constitutive neural network
    evaluate_load_list = np.linspace(0, 1, 21)
    # np.loadtxt("../results/MR2_DIC_Disp/DIC_evaluate_load.txt")

    nn_consti_kwargs = {"net_state_file": consti_nn_state_file,
                        "consti_tag": "ConstitutiveNetStressFreeTest1",
                        "layers": [4, 3, 1],
                        "bias": [True, True],
                        "activation": "parabolic_lu",
                        "post_file": post_file,
                        "manual_tag": True
                        }
    exp2func_kwargs = {"exp_data_file": "../DIC0825/2023_08_25_nocircle/NO_CIRCLE/data.npy",
                  "ROI_coor": [0, 0, 13.84, 10.0],
                  "ROI_index_range": [122, 105, 169, 170],
                  "pixel": 0.05322,
                  "right_boundary": 15.22,
                  "load_file": "../DIC0825/2023_08_25_nocircle/NO_CIRCLE/camera_Force.xlsx",
                  "force_load_scale": 1e2,
                  "boundary_pad": 5,
                  "sampling_pad": 2}
    exp_train_steps = np.arange(50, 200, 16)
    exp_evaluate_steps = np.arange(50, 500, 46)
    kwargs = {"case_name": case_name,
              "mesh_file": "../mesh/unit_element_square",
              "post_file": post_file,
              "element_type": "CG",
              "element_degree": 1,
              "ground_truth_consti": "MR2",  # None or specified constitutive
              "ground_truth_load_list": ground_truth_load_list,
              "evaluate_load_list": evaluate_load_list,
              "nn_consti_kwargs": nn_consti_kwargs,
              "exp_tag": False,  # bool
              "exp_train_steps": exp_train_steps,
              "exp_evaluate_steps": exp_evaluate_steps,
              "exp2func_kwargs": exp2func_kwargs,
              "weight_decay_lambda": 0,
              "optimisation_method": "CG"
              }
    case = NNHyperDispBCPlaneStress(kwargs, save_kwargs=True)
    case.trian_consti_nn()
    case.evaluate_consti_nn()

    # pk stress
    # case.post_pk_stress(case.ground_truth_consti, case.post_file + "fem_evaluate_truth_w.hdf5",
    #                          load_list=evaluate_load_list,
    #                          pk_stress_name="pk_stress_evaluate_truth")
    # case.post_pk_stress(case.nn_consti, case.post_file + "evaluate_consti_nn_w.hdf5",
    #                          load_list=evaluate_load_list,
    #                          pk_stress_name="pk_stress_evaluate_consti_nn")

    # # strain
    # case.plot_green_strain(case.post_file + "evaluate_consti_nn_w.hdf5",
    #                   evaluate_load_list,
    #                   green_strain_name="green_strain_consti_nn")
    # case.plot_almansi_strain(case.post_file + "evaluate_consti_nn_w.hdf5",
    #                          evaluate_load_list,
    #                          almansi_strain_name="almansi_strain_consti_nn")