import zntrack
from models import MODELS

import mlipx

DATAPATH = "neb.xyz"

project = zntrack.Project()

with project.group("initialize"):
    data = mlipx.LoadDataFile(path=DATAPATH)
#    trajectory = mlipx.NEBinterpolate(data=data.frames, n_images=5, mic=True)

for model_name, model in MODELS.items():
    with project.group(model_name):
        neb = mlipx.NEBs(
            data=data.frames,
            model=model,
            interpolate=True,
            mic=True,
            constraints=True,
            constraint_z=13,
            relax=True,
            fmax=0.06,
        )

project.build()
