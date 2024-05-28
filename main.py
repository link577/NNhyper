from nnhyper_plane_stress import NNHyperDispBCPlaneStress, \
	NNHyperEXPDispBCPlaneStress, NNHyperForceBCPlaneStress, \
   NNHyperEXPForceBCPlaneStress
import numpy as np

# 10% trianing strain

case_name = "test"
post_file = "../results/DIC0131_3_GD_unit_thickness_CG2_Pre_EXPDispBC_plane_stress_test4_BFGS_skip5_sp_loss2_3/"

consti_nn_state_file = "../pretrained_model/NH_test4_sp_431.pkl"
# consti_nn_state_file = None

# ground_truth_load_list = np.linspace(0, 10, 6) / 13.328
# evaluate_load_list = np.linspace(0, 100, 26) / 13.328
ground_truth_load_list = np.linspace(0, 5, 21)
evaluate_load_list = np.linspace(0, 5, 21)

nn_consti_kwargs = {"net_state_file": consti_nn_state_file,
                  "consti_tag": "ConstitutiveNetStressFreeTest4",
                  "layers": [4, 3, 1],
                  "bias": [True, True],
                  "activation": "softplus", # "parabolic_lu",
                  "post_file": post_file,
                  "manual_tag": True
                  }

# # data0926 circle
# exp2func_kwargs = {"exp_data_file": "../DIC0926/CIRCLE/data.npy",
#             "ROI_coor": [0, 0, 26.54, 16.38],
#             "ROI_index_range": [14, 59, 64, 140],
#             "pixel": 0.08191,
#             "right_boundary": 20.40,
#             "load_file": "../DIC0926/CIRCLE/camera_Force.xlsx",
#             "force_load_scale": 1e2,
#             "boundary_pad": 5,
#             "sampling_pad": 2}

# # data0926 no_circle
# exp2func_kwargs = {"exp_data_file": "../DIC0926/NO_CIRCLE/data.npy",
#             "ROI_coor": [0, 0, 20.16, 11.02],
#             "ROI_index_range": [35, 83, 70, 147],
#             "pixel": 0.07874,
#             "right_boundary": 19.61,
#             "load_file": "../DIC0926/NO_CIRCLE/camera_Force.xlsx",
#             "force_load_scale": 1e2,
#             "boundary_pad": 5,
#             "sampling_pad": 2}

# data0825 circle
# exp2func_kwargs = {"exp_data_file": "../DIC0825/CIRCLE/data.npy",
#             "ROI_coor": [0, 0, 20.34, 15.41],
#             "ROI_index_range": [117, 81, 192, 180],
#             "pixel": 0.05136,
#             "right_boundary": 19.16,
#             "load_file": "../DIC0825/CIRCLE/camera_Force.xlsx",
#             "force_load_scale": 1e2,
#             "boundary_pad": 5,
#             "sampling_pad": 2}

# # data1113 circle_1426
# exp2func_kwargs = {"exp_data_file": "../DIC1113/CIRCLE_1426/data.npy",
#                   "ROI_coor": [0, 0, 25.36, 12.54],
#                   "ROI_index_range": [52, 47, 98, 140],
#                   "pixel": 0.06817,
#                   "right_boundary": 14.52,
#                   "load_file": "../DIC1113/CIRCLE_1426/camera_Force.xlsx",
#                   "force_load_scale": 1e2/0.2912,
#                   "boundary_pad": 5,
#                   "sampling_pad": 2}
# exp2func_kwargs["preprocess_file"] = "../DIC1113/CIRCLE_1426/dic_disp_gd.xdmf"

# # data1113 black
# exp2func_kwargs = {"exp_data_file": "../DIC1113/BLACK/data.npy",
#                   "ROI_coor": [0, 0, 23.72, 11.86],
#                   "ROI_index_range": [34, 59, 73, 137],
#                   "pixel": 0.076016,
#                   "right_boundary": 14.37,
#                   "load_file": "../DIC1113/BLACK/camera_Force.xlsx",
#                   "force_load_scale": 1e2/0.3869,
#                   "boundary_pad": 5,
#                   "sampling_pad": 2}
# exp2func_kwargs["preprocess_file"] = "../DIC1113/BLACK/dic_disp_gd.xdmf"

# # data0131 1_1
# exp2func_kwargs = {"exp_data_file": "../DIC0131/1/data.npy",
#                    "ROI_coor": [0, 0, 31.11, 14.34],
#                    "ROI_index_range": [106, 174, 177, 328],
#                    "pixel": 0.0505,
#                    "right_boundary": 19.29,
#                    "load_file": "../DIC0131/1/camera_Force.xlsx",
#                    "force_load_scale": 1e2 / 0.1204,
#                    "boundary_pad": 5,
#                    "sampling_pad": 2}
# exp_train_steps = np.linspace(17, 1012, 26, dtype = int)
# exp_evaluate_steps = np.linspace(17, 1012, 26, dtype = int)
# exp2func_kwargs["preprocess_file"] = "../DIC0131/1/dic_disp_gd.xdmf"

# # data0131 1_2
# exp2func_kwargs = {"exp_data_file": "../DIC0131/2/data.npy",
#                    "ROI_coor": [0, 0, 26.77, 15.62],
#                    "ROI_index_range": [24, 65, 101, 197],
#                    "pixel": 0.0507,
#                    "right_boundary": 19.42,
#                    "load_file": "../DIC0131/2/camera_Force.xlsx",
#                    "force_load_scale": 1e2 / 0.1204,
#                    "boundary_pad": 5,
#                    "sampling_pad": 2}
# exp_train_steps = np.linspace(29, 912, 26, dtype=int)
# exp_evaluate_steps = np.linspace(29, 912, 26, dtype = int)
# exp2func_kwargs["preprocess_file"] = "../DIC0131/2/dic_disp_gd.xdmf"

# data0131 1_3
exp2func_kwargs = {"exp_data_file": "../DIC0131/3/data.npy",
                   "ROI_coor": [0, 0, 31.16, 15.58],
                   "ROI_index_range": [22, 67, 101, 225],
                   "pixel": 0.0493,
                   "right_boundary": 19.77,
                   "load_file": "../DIC0131/3/camera_Force.xlsx",
                   "force_load_scale": 1e2 / 0.1204,
                   "boundary_pad": 5,
                   "sampling_pad": 2}
exp_train_steps = np.linspace(34, 990, 26, dtype=int)
exp_evaluate_steps = np.linspace(34, 990, 26, dtype = int)
exp2func_kwargs["preprocess_file"] = "../DIC0131/3/dic_disp_gd.xdmf"



# exp_train_steps = np.linspace(11, 423, 26, dtype = int)
# exp_evaluate_steps = np.linspace(11, 423, 26, dtype = int)

# loss_step = np.arange(len(ground_truth_load_list))
# loss_step = np.arange(len(exp_train_steps))[-5:]
loss_step = np.array([6,7,8,15,16,17,23,24,25])
# loss_step = np.array([1,2,3,10,11,12,18,19,20])
# loss_step = np.array([2,4,10,12,18,20])

kwargs = {"case_name": case_name,
         "mesh_file": "../mesh/dic0131_3_square_subdomain_intact",
         "post_file": post_file,	
         "element_type": "CG",
         "element_degree": 2,
         "ground_truth_consti": None,  # None or specified constitutive
         "ground_truth_load_list": ground_truth_load_list,
         "loss_step": loss_step,
         "evaluate_load_list": evaluate_load_list,
         "nn_consti_kwargs": nn_consti_kwargs,
         "exp_tag": True,  # bool
         "exp_train_steps": exp_train_steps,
         "exp_evaluate_steps": exp_evaluate_steps,
         "exp2func_kwargs": exp2func_kwargs,
         "weight_decay_lambda": 0,
         "optimisation_method": "BFGS"
         }

case = NNHyperEXPDispBCPlaneStress(kwargs, save_kwargs=True)
case.trian_consti_nn()
case.evaluate_consti_nn()







