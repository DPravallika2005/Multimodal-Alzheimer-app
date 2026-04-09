import torch
import torch.nn as nn
import joblib
from torchvision import models
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionMLP(nn.Module):

    def __init__(self, input_dim=1536, hidden1=256, hidden2=128, dropout1=0.3, dropout2=0.2):
        super(FusionMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_models():

    # RESNET
    resnet=models.resnet18(weights=None)
    resnet.fc=nn.Identity()
    resnet.load_state_dict(torch.load("models/best_resnet18_oasis.pth",map_location=device),strict=False)
    resnet=resnet.to(device)
    resnet.eval()

    # DENSENET
    densenet=models.densenet121(weights=None)
    densenet.classifier=nn.Identity()
    densenet.load_state_dict(torch.load("models/best_densenet121_oasis.pth",map_location=device),strict=False)
    densenet=densenet.to(device)
    densenet.eval()

    # MRI MLP
    mri_model=FusionMLP()
    mri_model.load_state_dict(torch.load("models/fusion_mlp_best.pth",map_location=device))
    mri_model=mri_model.to(device)
    mri_model.eval()

    # Cognitive
    cognitive_model=joblib.load("models/xgb_rid_groupkfold_model.pkl")

    # Speech
    tokenizer=AutoTokenizer.from_pretrained("models/Alzheimer_BERT_Model")
    speech_model=AutoModelForSequenceClassification.from_pretrained("models/Alzheimer_BERT_Model")
    speech_model=speech_model.to(device)
    speech_model.eval()

    return resnet,densenet,mri_model,cognitive_model,tokenizer,speech_model