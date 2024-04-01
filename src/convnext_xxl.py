import open_clip
import torch

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained=False, device=torch.device('cuda'))
