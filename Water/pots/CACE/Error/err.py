# source ~/venvs/cace310/bin/activate

import numpy as np
import sys
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging

import sys
# sys.path.append('./cace/')
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

import cace
from cace.representations.cace_representation import Cace
from cace.calculators import CACECalculator

from ase import units

from ase.io import read, write


POT = "../t1/best_model_t1.pth"   #  potential file
TEST = "../../test_data.xyz"         # xyz with reference energy/forces


water_conf = read('test_data.xyz', '0')
cace_nnp = torch.load(POT, map_location=torch.device('cpu'), weights_only=False)
preprocessor = cace.modules.Preprocess()
cace_nnp.input_modules = nn.ModuleList([preprocessor])
cace_nnp.output_modules[1].calc_stress = True
cace_nnp.model_outputs.append('stress')

if os.path.exists('avge0.pkl'):
    with open('avge0.pkl', 'rb') as f:
        avge0 = pickle.load(f)
else:
    xyz = read('train_data.xyz',':')
    avge0 = cace.tools.compute_average_E0s(xyz)
    with open('avge0.pkl', 'wb') as f:
        pickle.dump(avge0, f)
calculator = CACECalculator(model_path=cace_nnp, 
                            device='cpu', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            compute_stress=True,
                           atomic_energies=avge0)
water_conf.set_calculator(calculator)



atoms_list = read(TEST, index=":")
e_err = []
f_err = []

for at in atoms_list:
    at.set_calculator(calculator)
    E_pred = at.get_potential_energy() / len(at)  
    F_pred = at.get_forces()
    E_true = at.info.get("energy")
    if E_true is None:
        raise KeyError("Missing atoms.info['energy'] in test.xyz")
    E_true /= len(at)
    F_true = at.arrays.get("forces")
    if F_true is None:
        F_true = at.arrays.get("force")
    if F_true is None:
        raise KeyError("Missing forces in test.xyz (need 'forces' or 'force')")
    e_err.append(E_pred - E_true)         # eV/atom
    f_err.append(F_pred - F_true)         # eV/Å

e_err = np.array(e_err)
f_err = np.vstack(f_err) 
# --- Compute MAE and RMSE ---
mae_E = np.mean(np.abs(e_err))
rmse_E = np.sqrt(np.mean(e_err**2))

mae_F = np.mean(np.abs(f_err))
rmse_F = np.sqrt(np.mean(f_err**2))

print(f"Energy MAE:  {mae_E:.6f} eV/atom")
print(f"Energy RMSE: {rmse_E:.6f} eV/atom")
print(f"Force MAE:   {mae_F:.6f} eV/Å")
print(f"Force RMSE:  {rmse_F:.6f} eV/Å")


e_err = np.array(e_err)                       
f_err = np.array(f_err)                       

# RMSEs
energy_rmse = np.sqrt(np.mean(e_err**2))                           # eV (per structure)
energy_rmse_per_atom = np.sqrt(np.mean((e_err / np.array(
    [len(at) for at in atoms_list]))**2))                           # eV/atom
force_rmse_component = np.sqrt(np.mean((f_err.reshape(-1,3))**2))   # eV/Å (per-component)
force_rmse_vector = np.sqrt(np.mean(np.sum(f_err**2, axis=2)))      # eV/Å (vector RMSE per atom)

print(f"Energy RMSE (per structure): {energy_rmse:.6f} eV")
print(f"Energy RMSE (per atom):      {energy_rmse_per_atom:.6f} eV/atom")
print(f"Force RMSE (component):      {force_rmse_component:.6f} eV/Å")
print(f"Force RMSE (vector):         {force_rmse_vector:.6f} eV/Å")
