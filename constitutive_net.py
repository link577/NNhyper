from dolfin import *
from dolfin_adjoint import *
import ufl
import numpy as np
import os
from numpy.random import randn, random
import pickle

np.random.seed(577)


class ConstitutiveNet():
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            # Load weights.
            return cls.load(args[0])
        else:
            return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], str):
            self.post_file = kwargs.get("post_file", "./nn_default_post/")
        else:
            activation_tag = kwargs.get("activation", "ELU")
            self.layers = args[0]  # a list of integer
            self.post_file = kwargs.get("post_file", "./nn_default_post/")
            self.bias = kwargs.get("bias", [True]*(len(self.layers)-1))  # a list of bool
            self.init_method = kwargs.get("init_method", "uniform")
            self.manual_tag = kwargs.get("manual_tag")
            self.output_activation = kwargs.get("output_activation", True)
            self.initialize_design_variables()    # generate initial self.design_variables
            self.manual_design_variables = self.generate_manual_design_variable()
            # self.weights = var_2_weight(self.design_variables)
            self.activation_func = set_activation(activation_tag)
            self.ctrls = None

        if not os.path.exists(self.post_file):
            os.makedirs(self.post_file)

    def __call__(self, *args, **kwargs):
        return self.Net(args) + \
               self.generate_manual_term(args)

    def design_variable_ctrls(self):
        if self.ctrls is None:
            r = []
            for design_variable in self.design_variables:
                var_matrix = design_variable["design_weights"]
                for row in var_matrix:
                    for component in row:
                        r.append(Control(component))
                if "design_bias" in design_variable:
                    r.append(Control(design_variable["design_bias"]))
            for mdv in self.manual_design_variables:
                r.append(Control(mdv))
            self.ctrls = r
        return self.ctrls

    def set_design_variable(self, var):
        '''
        read optimal results
        '''
        i = 0
        for design_variable in self.design_variables:
            var_matrix = design_variable["design_weights"]
            for row in var_matrix:
                for component in row:
                    component.assign(var[i])
                    # what is your function ?
                    component.block_variable.save_output()
                    i += 1
            if "design_bias" in design_variable:
                design_variable["design_bias"].assign(var[i])
                # what is your function ?
                design_variable["design_bias"].block_variable.save_output()
                i += 1
        for mdv in self.manual_design_variables:
            mdv.assign(var[i])
            mdv.block_variable.save_output()
            i += 1

    def generate_manual_design_variable(self):
        # E = 100 , nu = 0.3 --> mu = 38.46
        manual_design_variable_3 = Constant(sqrt(8.46))
        manual_design_variables = [manual_design_variable_3]
        if self.manual_tag:
            return manual_design_variables
        else:
            return []

    def generate_manual_term(self, inputs):
        # I_1, I_2, J = inputs
        if self.manual_tag:
            manual_term = -(self.manual_design_variables[-1] ** 2 + Constant(1e-5)) * ln(inputs[-1])
            return manual_term
        else:
            return 0

    def save(self):
        with open(f"{self.post_file}net_state.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path):
        with open(f"{path}", "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        state = self.__dict__.copy()
        state_design_variables = []
        for design_variable in self.design_variables:
            design_weights_matrix = design_variable["design_weights"]
            value_matrix = []
            value_dict = {}
            for row in design_weights_matrix:
                value_row = []
                for component in row:
                    value_row.append(component.values())
                value_matrix.append(value_row)
            value_dict["design_weights"] = value_matrix
            if "design_bias" in design_variable:
                value_dict["design_bias"] = \
                    design_variable["design_bias"].values()
            state_design_variables.append(value_dict)
        state["design_variables"] = state_design_variables

        state_manual_design_variables = []
        for mdv in self.manual_design_variables:
            state_manual_design_variables.append(mdv.values())
        state["manual_design_variables"] = state_manual_design_variables
        state["ctrls"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        design_variables = []
        for design_variable in self.design_variables:
            variable_dict = {}
            variable_matrix = []
            design_weights_matrix = design_variable["design_weights"]
            for row in design_weights_matrix:
                design_weights_row = []
                for value in row:
                    if len(value == 1):
                        value = value[0]
                    design_weights_row.append(Constant(value))
                variable_matrix.append(design_weights_row)
            variable_dict["design_weights"] = variable_matrix
            if "design_bias" in design_variable:
                variable_dict["design_bias"] = \
                    Constant(design_variable["design_bias"])
            design_variables.append(variable_dict)
        self.design_variables = design_variables

        manual_design_variables = []
        for mdv in self.manual_design_variables:
            if len(mdv == 1):
                mdv = mdv[0]
            manual_design_variables.append(Constant(mdv))
        self.manual_design_variables = manual_design_variables

    def initialize_design_variables(self):
        '''
        design_variables = [dict,dict .....]
        dict = {design_weights, design_bias}
        design_weights : a m x n Constant matrix
        design_bias : a Constant of shape m
        '''
        init_method = self.init_method.lower()
        assert init_method in ["normal", "uniform"]
        # each layer
        self.design_variables = []
        for i in range(len(self.layers)-1):
            design_variable = {}
            design_variable_matrix = []
            dim_row = self.layers[i + 1]
            dim_column = self.layers[i]
            # each row
            for row in range(dim_row):
                design_variable_row = []
                for column in range(dim_column):
                    if init_method == "uniform":
                        value = random()
                    elif init_method == "normal":
                        value = np.sqrt(2 / self.layers[i]) * randn()
                    component = Constant(value)
                    design_variable_row.append(component)
                design_variable_matrix.append(design_variable_row)
            design_variable["design_weights"] = design_variable_matrix
            if self.bias[i]:
                b = Constant(np.zeros(self.layers[i + 1]))
                design_variable["design_bias"] = b
            self.design_variables.append(design_variable)
        return self.design_variables

    def Net(self, inputs):
        # if set weights to a class attribute here,
        # then corresponding modifications should be
        # made in the __setstate__ method
        weights = self.var_2_weight()
        # I1, I2, J = inputs
        # inputs = (I1, I2, J)
        results = ufl.as_vector(inputs)
        depth = len(weights)
        for i, weight in enumerate(weights):
            weight_matrix = as_matrix(weight["weights"])
            term = dot(weight_matrix, results)
            if "bias" in weight:
                term += weight["bias"]
            if i + 1 >= depth:
                results = term
            else:
                results = apply_activation(term, func=self.activation_func)
        if self.output_activation:
            results = apply_activation(results, func=self.activation_func)
        if results.ufl_shape[0] == 1:
            # can only integral results[0]*dx
            results = results[0]
        return results

    def var_2_weight(self):
        '''
        weight_ij = var_ij**2
        ensuring the convexity
        '''
        weights = []
        for design_variable in self.design_variables:
            weight = {}
            var_matrix = design_variable["design_weights"]
            weight_matrix = []
            for row in var_matrix:
                weight_row = []
                for component in row:
                    weight_row.append(component ** 2)
                weight_matrix.append(weight_row)
            weight["weights"] = weight_matrix
            if "design_bias" in design_variable:
                weight["bias"] = design_variable["design_bias"]
            weights.append(weight)
        return weights

    def get_design_variables(self):
        """
        print out the values of design variables
        """
        filename = self.post_file + "design_variable_values.txt"
        for i, design_variable in enumerate(self.design_variables):
            var_matrix = design_variable["design_weights"]
            self.write_log(f"variables in {i}st layer", filename=filename, print_out=False,
                           clear=(i == 0))
            for row in var_matrix:
                row_values = []
                for component in row:
                    value = component.values()[0]
                    row_values.append(value)
                self.write_log(f"{row_values}", filename=filename, print_out=False)
            if "design_bias" in design_variable:
                value = design_variable["design_bias"].values()
                self.write_log(f"bias in {i}st layer \n{value}", filename=filename, print_out=False)

        manual_values = []
        self.write_log(f"manual variables", filename=filename, print_out=False)
        for mdv in self.manual_design_variables:
            value = mdv.values()[0]
            manual_values.append(value)
        self.write_log(f"{manual_values}", filename=filename, print_out=False)

    def weight_decay_term(self, decay_lambda=0.1):
        """
        weight decay term in loss function
        decay_lambda = 0 : no weight decay
        """
        decay_lambda = Constant(decay_lambda)
        term = 0.
        for design_variable in self.design_variables:
            var_matrix = design_variable["design_weights"]
            for row in var_matrix:
                for component in row:
                    # weight^2
                    term += 0.5*decay_lambda*component**4
        for mdv in self.manual_design_variables:
            term += 0.5 * decay_lambda * mdv ** 4
        return term

    def fake_updata(self):
        for i, design_variable in enumerate(self.design_variables):
            var_matrix = design_variable["design_weights"]
            for row in var_matrix:
                for component in row:
                    component.assign(Constant(5))
                    # what is your function ?
                    component.block_variable.save_output()
            if "design_bias" in design_variable:
                pass
                # design_variable["design_bias"].assign(var[i])
                # # what is your function ?
                # design_variable["design_bias"].block_variable.save_output()
        for mdv in self.manual_design_variables:
            mdv.assign(Constant(5))
            mdv.block_variable.save_output()

    def write_log(self, info="", filename=None, clear=False, print_out=True, end="\n"):
        if filename is None:
            filename = self.post_file+f"log.txt"
        with open(filename, "a") as f:
            if clear:
                f.truncate(0)
            f.write(f"{info}{end}")
        if print_out:
            print(info)


# class ConstitutiveNetStressFreeFernandez2021(ConstitutiveNet):

#     def __call__(self, *args, **kwargs):
#         # Cauchy(I, theta) = 0
#         assert "C" in kwargs.keys()
#         C = kwargs["C"]
#         psi_net = self.Net(args) + self.generate_manual_term(args)
#         I_1_initial = Constant(2.)
#         I_2_initial = Constant(1.)
#         J_initial = Constant(1.)
#         inputs_initial = (I_1_initial, I_2_initial, J_initial)
#         psi_initial = self.Net(inputs_initial) + self.generate_manual_term(inputs_initial)
#         D_I_1 = diff(psi_initial, I_1_initial)
#         D_I_2 = diff(psi_initial, I_2_initial)
#         D_J = diff(psi_initial, J_initial)
#         Cauchy_at_I = ((D_I_1 + D_I_2) + D_J / 2) * Identity(2)  # 1/2 Cauchy at I exactly
#         psi = psi_net - inner(Cauchy_at_I, C)
#         return psi

    # def __call__(self, *args, **kwargs):
    #     # partial(W)/partial(I1,I2,J) = 0
    #     I_1, I_2, J = args
    #     assert "C" in kwargs.keys()
    #     C = kwargs["C"]
    #     psi_net = self.Net(args) + self.generate_manual_term(args)
    #     I_1_initial = Constant(2.)
    #     I_2_initial = Constant(1.)
    #     J_initial = Constant(1.)
    #     inputs_initial = (I_1_initial, I_2_initial, J_initial)
    #     psi_initial = self.Net(inputs_initial) + self.generate_manual_term(inputs_initial)
    #     D_I_1 = diff(psi_initial, I_1_initial)
    #     D_I_2 = diff(psi_initial, I_2_initial)
    #     D_J = diff(psi_initial, J_initial)
    #     psi = psi_net - D_I_1*I_1 - D_I_2* I_2 - D_J*J
    #           # dot(as_vector((D_I_1, D_I_2, D_J)), as_vector(args))
    #     return psi


class ConstitutiveNetPolyConvexStressFree(ConstitutiveNet):
    '''
    construct a polyconvex NN constitutive though I1 & J
    which satisfies stress free condition
    '''
    def generate_manual_term(self, inputs):
        _, _, J = inputs
        if self.manual_tag:
            manual_term = -(self.manual_design_variables[-1] ** 2 + Constant(1e-5)) * ln(J)\
                          + (self.manual_design_variables[-1] ** 2 + Constant(1e-5)) * J
            return manual_term
        else:
            return 0

    def initialize_design_variables(self):
        '''
        design_variables = [dict,dict .....]
        dict = {design_weights, design_bias}
        design_weights : a m x n Constant matrix
        design_bias : a Constant of shape m
        '''
        init_method = self.init_method.lower()
        assert init_method in ["normal", "uniform"]
        # each layer
        self.design_variables = []
        for i in range(len(self.layers)-1):
            design_variable = {}
            design_variable_matrix = []
            dim_row = self.layers[i + 1]
            dim_column = self.layers[i] if i != 0 else self.layers[i]-1
            # each row
            for row in range(dim_row):
                design_variable_row = []
                for column in range(dim_column):
                    if init_method == "uniform":
                        value = random()
                    elif init_method == "normal":
                        value = np.sqrt(2 / self.layers[i]) * randn()
                    component = Constant(value)
                    design_variable_row.append(component)
                design_variable_matrix.append(design_variable_row)
            design_variable["design_weights"] = design_variable_matrix
            if self.bias[i]:
                # DJ/Db = 0 at some inappropriate initial bia value (b = 0)
                b = Constant(0*np.ones(self.layers[i + 1]))
                design_variable["design_bias"] = b
            self.design_variables.append(design_variable)
        return self.design_variables

    def Net(self, inputs):
        # if set weights to a class attribute here,
        # then corresponding modifications should be
        # made in the __setstate__ method
        weights = self.var_2_weight()
        I1, I2, J = inputs
        inputs = (I1, -J)
        results = ufl.as_vector(inputs)
        depth = len(weights)
        for i, weight in enumerate(weights):
            weight_matrix = as_matrix(weight["weights"])
            term = dot(weight_matrix, results)
            if "bias" in weight:
                term += weight["bias"]
            if i + 1 >= depth:
                results = term
            else:
                results = apply_activation(term, func=self.activation_func)
        if self.output_activation:
            results = apply_activation(results, func=self.activation_func)
        if results.ufl_shape[0] == 1:
            # can only integral results[0]*dx
            results = results[0]
        return results

    def var_2_weight(self):
        '''
        weight_ij = var_ij**2
        ensuring the convexity
        '''
        weights = []
        for layer_index, design_variable in enumerate(self.design_variables):
            weight = {}
            var_matrix = design_variable["design_weights"]
            weight_matrix = []
            for row in var_matrix:
                weight_row = []
                for component in row:
                    weight_row.append(component ** 2)
                # construct 2*D_(I1,...) - D(-J) = 0
                # by control the weight of first layer
                if layer_index == 0:
                    weight_minus_J = 0
                    for component_i in weight_row:
                        weight_minus_J += 2*component_i
                    weight_row.append(weight_minus_J)
                weight_matrix.append(weight_row)
            weight["weights"] = weight_matrix
            if "design_bias" in design_variable:
                weight["bias"] = design_variable["design_bias"]
            weights.append(weight)
        return weights

    def weight_decay_term(self, decay_lambda=0.1):
        """
        weight decay term in loss function
        decay_lambda = 0 : no weight decay
        """
        decay_lambda = Constant(decay_lambda)
        term = 0.
        for layer_index, design_variable in enumerate(self.design_variables):
            var_matrix = design_variable["design_weights"]
            for row in var_matrix:
                for component in row:
                    # weight^2
                    term += 0.5*decay_lambda*component**4
                if layer_index == 0:
                    weight_last = 0
                    for component in row:
                        weight_last += 2*component**2
                    term += 0.5 * decay_lambda * weight_last ** 2
        for mdv in self.manual_design_variables:
            term += 0.5 * 2 * decay_lambda * mdv ** 4
        return term


# class ConstitutiveNetStressFreeTest(ConstitutiveNetPolyConvexStressFree):
#     def Net(self, inputs):
#         # if set weights to a class attribute here,
#         # then corresponding modifications should be
#         # made in the __setstate__ method
#         weights = self.var_2_weight()
#         I1, I2, J = inputs
#         inputs = (I1, I2, -J)
#         results = ufl.as_vector(inputs)
#         depth = len(weights)
#         for i, weight in enumerate(weights):
#             weight_matrix = as_matrix(weight["weights"])
#             term = dot(weight_matrix, results)
#             if "bias" in weight:
#                 term += weight["bias"]
#             if i + 1 >= depth:
#                 results = term
#             else:
#                 results = apply_activation(term, func=self.activation_func)
#         if self.output_activation:
#             results = apply_activation(results, func=self.activation_func)
#         if results.ufl_shape[0] == 1:
#             # can only integral results[0]*dx
#             results = results[0]
#         return results

#     def var_2_weight(self):
#         '''
#         weight_ij = var_ij**2
#         ensuring the convexity
#         '''
#         weights = []
#         for layer_index, design_variable in enumerate(self.design_variables):
#             weight = {}
#             var_matrix = design_variable["design_weights"]
#             weight_matrix = []
#             for row in var_matrix:
#                 weight_row = []
#                 for component in row:
#                     weight_row.append(component ** 2)
#                 # construct 2*D_(I1,...) - D(-J) = 0
#                 # by control the weight of first layer
#                 if layer_index == 0:
#                     weight_minus_J = 2 * weight_row[0] + 4 * weight_row[1]
#                     weight_row.append(weight_minus_J)
#                 weight_matrix.append(weight_row)
#             weight["weights"] = weight_matrix
#             if "design_bias" in design_variable:
#                 weight["bias"] = design_variable["design_bias"]
#             weights.append(weight)
#         return weights

#     # def weight_decay_term(self, decay_lambda=0.1):
#     #     pass


# class ConstitutiveNetStressFreeTestManualKlein(ConstitutiveNetStressFreeTest):
#     def generate_manual_term(self, inputs):
#         _, _, J = inputs
#         if self.manual_tag:
#             manual_term = (self.manual_design_variables[-1] ** 2 + Constant(1e-5)) * \
#                           (J + 1/J - 2)**2
#             return manual_term
#         else:
#             return 0


# class ConstitutiveNetStressFreeTest1(ConstitutiveNetPolyConvexStressFree):
#     def Net(self, inputs):
#         # if set weights to a class attribute here,
#         # then corresponding modifications should be
#         # made in the __setstate__ method
#         weights = self.var_2_weight()
#         I1, I2, J = inputs
#         inputs = (I1, I2, J, J**(-1/3))
#         results = ufl.as_vector(inputs)
#         depth = len(weights)
#         for i, weight in enumerate(weights):
#             weight_matrix = as_matrix(weight["weights"])
#             term = dot(weight_matrix, results)
#             if "bias" in weight:
#                 term += weight["bias"]
#             if i + 1 >= depth:
#                 results = term
#             else:
#                 results = apply_activation(term, func=self.activation_func)
#         if self.output_activation:
#             results = apply_activation(results, func=self.activation_func)
#         if results.ufl_shape[0] == 1:
#             # can only integral results[0]*dx
#             results = results[0]
#         return results

#     def var_2_weight(self):
#         '''
#         weight_ij = var_ij**2
#         ensuring the convexity
#         '''
#         weights = []
#         for layer_index, design_variable in enumerate(self.design_variables):
#             weight = {}
#             var_matrix = design_variable["design_weights"]
#             weight_matrix = []
#             for row in var_matrix:
#                 weight_row = []
#                 for component in row:
#                     weight_row.append(component ** 2)
#                 # construct 2*D_(I1,...) - D(-J) = 0
#                 # by control the weight of first layer
#                 if layer_index == 0:
#                     weight_minus_J = 6 * weight_row[0] + \
#                                      12 * weight_row[1] + \
#                                      3 * weight_row[2]
#                     weight_row.append(weight_minus_J)
#                 weight_matrix.append(weight_row)
#             weight["weights"] = weight_matrix
#             if "design_bias" in design_variable:
#                 weight["bias"] = design_variable["design_bias"]
#             weights.append(weight)
#         return weights


# class ConstitutiveNetStressFreeTest1ManualKlein(ConstitutiveNetStressFreeTest1):
#     def generate_manual_term(self, inputs):
#         _, _, J = inputs
#         if self.manual_tag:
#             manual_term = (self.manual_design_variables[-1] ** 2 + Constant(1e-5)) * \
#                           (J + 1/J - 2)**2
#             return manual_term
#         else:
#             return 0


# class ConstitutiveNetStressFreeTest2(ConstitutiveNetPolyConvexStressFree):
#     def Net(self, inputs):
#         # if set weights to a class attribute here,
#         # then corresponding modifications should be
#         # made in the __setstate__ method
#         weights = self.var_2_weight()
#         I1, I2, J = inputs
#         inputs = (I1, I2, J, J**(-2/3))
#         results = ufl.as_vector(inputs)
#         depth = len(weights)
#         for i, weight in enumerate(weights):
#             weight_matrix = as_matrix(weight["weights"])
#             term = dot(weight_matrix, results)
#             if "bias" in weight:
#                 term += weight["bias"]
#             if i + 1 >= depth:
#                 results = term
#             else:
#                 results = apply_activation(term, func=self.activation_func)
#         if self.output_activation:
#             results = apply_activation(results, func=self.activation_func)
#         if results.ufl_shape[0] == 1:
#             # can only integral results[0]*dx
#             results = results[0]
#         return results

#     def var_2_weight(self):
#         '''
#         weight_ij = var_ij**2
#         ensuring the convexity
#         '''
#         weights = []
#         for layer_index, design_variable in enumerate(self.design_variables):
#             weight = {}
#             var_matrix = design_variable["design_weights"]
#             weight_matrix = []
#             for row in var_matrix:
#                 weight_row = []
#                 for component in row:
#                     weight_row.append(component ** 2)
#                 # construct 2*D_(I1,...) - D(-J) = 0
#                 # by control the weight of first layer
#                 if layer_index == 0:
#                     weight_minus_J = 3 * weight_row[0] + \
#                                      6 * weight_row[1] + \
#                                      3 / 2 * weight_row[2]
#                     weight_row.append(weight_minus_J)
#                 weight_matrix.append(weight_row)
#             weight["weights"] = weight_matrix
#             if "design_bias" in design_variable:
#                 weight["bias"] = design_variable["design_bias"]
#             weights.append(weight)
#         return weights


# class ConstitutiveNetStressFreeTest3(ConstitutiveNetPolyConvexStressFree):
#     def Net(self, inputs):
#         # if set weights to a class attribute here,
#         # then corresponding modifications should be
#         # made in the __setstate__ method
#         weights = self.var_2_weight()
#         I1, I2, J = inputs
#         inputs = (I1, I2, J, J ** (-1))
#         results = ufl.as_vector(inputs)
#         depth = len(weights)
#         for i, weight in enumerate(weights):
#             weight_matrix = as_matrix(weight["weights"])
#             term = dot(weight_matrix, results)
#             if "bias" in weight:
#                 term += weight["bias"]
#             if i + 1 >= depth:
#                 results = term
#             else:
#                 results = apply_activation(term, func=self.activation_func)
#         if self.output_activation:
#             results = apply_activation(results, func=self.activation_func)
#         if results.ufl_shape[0] == 1:
#             # can only integral results[0]*dx
#             results = results[0]
#         return results

#     def var_2_weight(self):
#         '''
#         weight_ij = var_ij**2
#         ensuring the convexity
#         '''
#         weights = []
#         for layer_index, design_variable in enumerate(self.design_variables):
#             weight = {}
#             var_matrix = design_variable["design_weights"]
#             weight_matrix = []
#             for row in var_matrix:
#                 weight_row = []
#                 for component in row:
#                     weight_row.append(component ** 2)
#                 # construct 2*D_(I1,...) - D(-J) = 0
#                 # by control the weight of first layer
#                 if layer_index == 0:
#                     weight_minus_J = 2 * weight_row[0] + \
#                                      4 * weight_row[1] + \
#                                      1 * weight_row[2]
#                     weight_row.append(weight_minus_J)
#                 weight_matrix.append(weight_row)
#             weight["weights"] = weight_matrix
#             if "design_bias" in design_variable:
#                 weight["bias"] = design_variable["design_bias"]
#             weights.append(weight)
#         return weights


# class ConstitutiveNetNeoHookean(ConstitutiveNet):

#     def generate_manual_design_variable(self):
#         # E = 100 , nu = 0.3 --> mu = 38.46, lambda = 57.69
#         manual_design_variable_mu = Constant(sqrt(10))
#         manual_design_variable_lambda = Constant(sqrt(10))
#         manual_design_variables = [manual_design_variable_mu,
#                                    manual_design_variable_lambda]
#         if self.manual_tag:
#             return manual_design_variables
#         else:
#             return []

#     def generate_manual_term(self, inputs):
#         I_1, I_2, J = inputs
#         mu, lamda = self.manual_design_variables
#         mu = mu**2
#         lamda = lamda**2
#         if self.manual_tag:
#             manual_term = mu/2*(I_1 - 3 - 2*ln(J)) + lamda/2*(J-1)**2
#             return manual_term
#         else:
#             return 0

#     def Net(self, inputs):
#         return 0


class ConstitutiveNetStressFreeTest4(ConstitutiveNetPolyConvexStressFree):
    def Net(self, inputs):
        # if set weights to a class attribute here,
        # then corresponding modifications should be
        # made in the __setstate__ method
        weights = self.var_2_weight()
        I1, I2, J = inputs
        inputs = (I1, I2, J**2, -J)
        results = ufl.as_vector(inputs)
        depth = len(weights)
        for i, weight in enumerate(weights):
            weight_matrix = as_matrix(weight["weights"])
            term = dot(weight_matrix, results)
            if "bias" in weight:
                term += weight["bias"]
            if i + 1 >= depth:
                results = term
            else:
                results = apply_activation(term, func=self.activation_func)
        if self.output_activation:
            results = apply_activation(results, func=self.activation_func)
        if results.ufl_shape[0] == 1:
            # can only integral results[0]*dx
            results = results[0]
        return results

    def var_2_weight(self):
        '''
        weight_ij = var_ij**2
        ensuring the convexity
        '''
        weights = []
        for layer_index, design_variable in enumerate(self.design_variables):
            weight = {}
            var_matrix = design_variable["design_weights"]
            weight_matrix = []
            for row in var_matrix:
                weight_row = []
                for component in row:
                    weight_row.append(component ** 2)
                # construct 2*D_(I1,...) - D(-J) = 0
                # by control the weight of first layer
                if layer_index == 0:
                    weight_minus_J = 2 * weight_row[0] + \
                                     4 * weight_row[1] + \
                                     2 * weight_row[2]
                    weight_row.append(weight_minus_J)
                weight_matrix.append(weight_row)
            weight["weights"] = weight_matrix
            if "design_bias" in design_variable:
                weight["bias"] = design_variable["design_bias"]
            weights.append(weight)
        return weights


# def generate_design_variables(layers, bias, init_method="normal"):
#     '''
#     design_variables = [dict,dict .....]
#     dict = {design_weights, design_bias}
#     design_weights : a m x n Constant matrix
#     design_bias : a Constant of shape m
#     '''
#     init_method = init_method.lower()
#     assert init_method in ["normal", "uniform"]
#     # each layer
#     design_variables = []
#     for i in range(len(layers)-1):
#         design_variable = {}
#         design_variable_matrix = []
#         dim_row = layers[i + 1]
#         dim_column = layers[i]
#         # each row
#         for row in range(dim_row):
#             design_variable_row = []
#             for column in range(dim_column):
#                 if init_method == "uniform":
#                     value = random()
#                 elif init_method == "normal":
#                     value = np.sqrt(2 / layers[i]) * randn()
#                 component = Constant(value)
#                 design_variable_row.append(component)
#             design_variable_matrix.append(design_variable_row)
#         design_variable["design_weights"] = design_variable_matrix
#         if bias[i]:
#             b = Constant(np.zeros(layers[i + 1]))
#             design_variable["design_bias"] = b
#         design_variables.append(design_variable)
#     return design_variables


# def var_2_weight(var):
#     '''
#     weight_ij = var_ij**2
#     ensuring the convexity
#     '''
#     weights = []
#     for design_variable in var:
#         weight = {}
#         var_matrix = design_variable["design_weights"]
#         weight_matrix = []
#         for row in var_matrix:
#             weight_row = []
#             for component in row:
#                 weight_row.append(component**2)
#             weight_matrix.append(weight_row)
#         weight["weights"] = weight_matrix
#         if "design_bias" in design_variable:
#             weight["bias"] = design_variable["design_bias"]
#         weights.append(weight)
#     return weights


def set_activation(tag):
    act_str = ["sigmoid", "relu", "ELU", "identity",
               "exp_minus_one", "softplus", "parabolic_lu",
               "parabolic_lu_1"]
    act_fun = [sigmoid, relu, ELU, identity,
               exp_minus_one, softplus, parabolic_lu,
               parabolic_lu_1]
    assert tag in act_str
    # TODO rewrite this
#     assert tag == "ELU"
    for i, s in enumerate(act_str):
        if s == tag:
            act = act_fun[i]
            break
    return act




# def Net(inputs, var, act, output_activation=False):
#     weights = var_2_weight(var)
#     I1, I2, J = inputs
#     inputs = (I1, I2, J)
#     results = ufl.as_vector(inputs)
#     depth = len(weights)
#     for i, weight in enumerate(weights):
#         weight_matrix = as_matrix(weight["weights"])
#         term = dot(weight_matrix, results)
#         if "bias" in weight:
#             term += weight["bias"]
#         if i+1 >= depth:
#             results = term
#         else:
#             results = apply_activation(term, func=act)
#     if output_activation:
#         results = apply_activation(results, func=act)
#     if results.ufl_shape[0] == 1:
#         # can only integral results[0]*dx
#         results = results[0]
#     return results


# some activation functions
def sigmoid(x):
    return 1 / (1 + ufl.exp(-x))


def identity(x):
    return conditional(ufl.gt(x, -1e8), x, (x+1e8)**2 + x)


def relu(x):
    return ufl.Max(0, x)


def ELU(x, alpha=1):
    return conditional(ufl.gt(x, 0), x, alpha * (ufl.exp(x) - 1))


def exp_minus_one(x):
    return ufl.exp(x) - 1


def softplus(x):
    return ufl.ln(1 + ufl.exp(x))


def parabolic_lu(x, a=0.1, alpha=1):
    return conditional(ufl.gt(x, 0), a*x**2+x, alpha * (ufl.exp(x) - 1))


def parabolic_lu_1(x, a=1, alpha=1):
    return conditional(ufl.gt(x, 0), a*x**2+x, alpha * (ufl.exp(x) - 1))


def apply_activation(vec, func):
    """Applies the activation function `func` element-wise to the UFL expression `vec`.
    """
    v = [func(vec[i]) for i in range(vec.ufl_shape[0])]
    return ufl.as_vector(v)


def generate_consti_net(**kwargs):
    tag = kwargs["consti_tag"]
    assert tag in ["ConstitutiveNet",
                   "ConstitutiveNetPolyConvexStressFree",
                   "ConstitutiveNetStressFreeTest4"]
    net_state_file = kwargs.get("net_state_file")
    if isinstance(net_state_file, str):
        if tag == "ConstitutiveNet":
            return ConstitutiveNet(net_state_file, **kwargs)
        if tag == "ConstitutiveNetPolyConvexStressFree":
            return ConstitutiveNetPolyConvexStressFree(net_state_file, **kwargs)
        if tag == "ConstitutiveNetStressFreeTest4":
            return ConstitutiveNetStressFreeTest4(net_state_file, **kwargs)
    else:
        if tag == "ConstitutiveNet":
            return ConstitutiveNet(kwargs["layers"], **kwargs)
        if tag == "ConstitutiveNetPolyConvexStressFree":
            return ConstitutiveNetPolyConvexStressFree(kwargs["layers"], **kwargs)
        if tag == "ConstitutiveNetStressFreeTest4":
            return ConstitutiveNetStressFreeTest4(kwargs["layers"], **kwargs)
