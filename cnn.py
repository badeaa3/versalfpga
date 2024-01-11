# notes
# 1. To get the xmodel first run with quant_mode="config" then again with "test"

# vitis ai imports
import pytorch_nndct
from pytorch_nndct.apis import torch_quantizer
from pytorch_nndct.apis import Inspector

# python 
import argparse
import datetime
import numpy as np
import os

# torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*6*10, 128)  # Adjust based on your input size # 64 * 3 * 5
        self.fc2 = nn.Linear(128, 3)  # Adjust output size based on your task

    def forward(self, x):
        # process the image
        x = x.view(-1, 1, 13, 21)  # Add channel dimension for grayscale image
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.shape[0], -1) #-1, 64 * 3 * 5)  # Adjust based on your input size
        # feed forward
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = x.flatten()
        return x

if __name__ == "__main__":
            
    # quant_mode = 'calib'
    quant_mode = 'test'

    # create an instance model
    model = SimpleCNN() 

    # create test inputs
    batch_size = 1 if quant_mode == "test" else 10
    x = torch.ones(batch_size,13,21)
    y = torch.ones(batch_size,1)

    # Inspect the model.
    input_shape = x[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inspector = Inspector("DPUCVDX8G_ISA3_C32B6")
    inspector.inspect(model, (input_shape), device=device, image_format='png')

    # nndct_macs, nndct_params = summary.model_complexity(model, input, return_flops=False, readable=False, print_model_analysis=True)
    quant_dir = "./quant_dir"
    quantizer = torch_quantizer(quant_mode, model, (input_shape), output_dir = quant_dir, device=device)
    quant_model = quantizer.quant_model
    
    # quick test evaluate
    # quant_model.eval()
    # with torch.no_grad():
    p = quant_model(x)

    # "calib" step.
    if quant_mode == "calib":
        print("calib")
        quantizer.export_quant_config()
    elif quant_mode == "test":
        print("test")
        # "deploy" step.The Xilinx scripts don't do all these things at once.
        quantizer.export_xmodel(quant_dir, deploy_check=True)
        quantizer.export_torch_script(output_dir=quant_dir)
        quantizer.export_onnx_model(output_dir=quant_dir)
