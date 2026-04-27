import torch
import nibabel as nib
import numpy as np

from .model import encoderOnlyComplex


device = "cuda"


model = encoderOnlyComplex().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()


def run_registration(moving_path, fixed_path):

    moving = nib.load(moving_path).get_fdata()
    fixed = nib.load(fixed_path).get_fdata()

    moving = torch.tensor(moving).unsqueeze(0).unsqueeze(0).float().to(device)
    fixed = torch.tensor(fixed).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():

        flow = model(moving, fixed, registration=True)

    # warped = warped.cpu().numpy()[0,0]
    flow = flow.cpu().numpy()[0]

    return flow, moving.cpu().numpy()[0,0], fixed.cpu().numpy()[0,0]