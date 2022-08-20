import cv2
import os
import numpy as np
import tensorflow as tf
import re
import torch
from torchvision import transforms
from PIL import Image


class FaceEmbedder:
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model)
        self.model.to(self.device)
        self.model.eval()
        self.transformer = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        super(FaceEmbedder, self).__init__()
        # img = Image.open(r"C:\Users\alexy\Desktop\0_fn_0_aligned.jpg").convert('RGB')
        # img = self.transformer(img).to(self.device)
        # self.model(torch.unsqueeze(img, 0))



    def preprocess(self, images):
        return
    # align
    # crop?
    # bgr to rgb
