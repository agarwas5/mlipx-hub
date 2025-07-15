import mlipx
import dataclasses
from ase.calculators.calculator import Calculator
import sys
from ase.units import Bohr


MODELS = {}

# Example MLIP
MODELS["mace_agnesi"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="MACECalculator",
    device="auto",
    kwargs={"model_paths": "/mnt/nfs/ml_training/pretrained_mace_models/mace_agnesi_medium.model", "dtype": "float64"},
)

MODELS["mace_mpa0"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "/mnt/nfs/ml_training/pretrained_mace_models/mace-mpa-0-medium.model", "dispersion":False, "dispersion_cutoff":40*Bohr, "default_dtype": "float64"},
)
MODELS["mace_matpes-omat"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="cpu",
    kwargs={"model": "/mnt/nfs/ml_training/pretrained_mace_models/MACE-matpes-pbe-omat-ft.model", "dtype": "float64"},
)
'''
MODELS["mace_dodh"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "/home/ubuntu/srishti_test/dodh_model_direct_learn/Cu-C-O-H_v10_64x2e_MLPirreps64x0e_MACE_0_swa.model", "dispersion":False, "dispersion_cutoff":40*Bohr, "default_dtype": "float64"},
)

MODELS["mace_dodh_vdw"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "/home/ubuntu/srishti_test/dodh_model_direct_learn/Cu-C-O-H_v10_64x2e_MLPirreps64x0e_MACE_0_swa.model", "dispersion":True, "dispersion_cutoff":20*Bohr, "default_dtype": "float64"},
)
'''
MODELS["sevennet"] = mlipx.GenericASECalculator(
    module="sevenn.sevennet_calculator",
    class_name="SevenNetCalculator",
    device="auto",
    kwargs={"model": "7net-mf-ompa", "modal": 'mpa'},
)

MODELS["mattersim"] = mlipx.GenericASECalculator(
    module="mattersim.forcefield",
    class_name="MatterSimCalculator",
)

MODELS["ocp"] = mlipx.GenericASECalculator(
    module="fairchem.core",
    class_name="OCPCalculator",
    kwargs={
        "checkpoint_path": "/mnt/nfs/ml_training/pretrained_ocp_models/eqV2_31M_omat_mp_salex.pt",
        "seed" : 0
        },
)

MODELS["grace"] = mlipx.GenericASECalculator(
    module="tensorpotential.calculator",
    class_name="grace_fm",
    kwargs={
        "model":"GRACE-2L-OAM",
        }
)

@dataclasses.dataclass
class OrbCalc:
    name: str
    device: str | None = None

    def get_calculator(self, **kwargs):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        method = getattr(pretrained, self.name)
        if self.device is None:
            orbff = method(**kwargs)
            calc = ORBCalculator(orbff, **kwargs)
        elif self.device == "auto":
            orbff = method(device="auto", **kwargs)
            calc = ORBCalculator(orbff, device="auto", **kwargs)
        else:
            orbff = method(device=self.device, **kwargs)
            calc = ORBCalculator(orbff, device=self.device, **kwargs)
        return calc

MODELS["orb_v2"] = OrbCalc(name="orb_v2")


# OPTIONAL
# ========
# If you have custom property names you can use the UpdatedFramesCalc
# to set the energy, force and isolated_energies keys mlipx expects.
#REFERENCE = mlipx.UpdateFramesCalc(
#        results_mapping={"energy": "DFT_energy", "forces": "DFT_forces"},
#    info_mapping={mlipx.abc.ASEKeys.isolated_energies.value: "isolated_free_energies"},
#)
