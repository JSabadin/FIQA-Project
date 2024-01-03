
import numpy as np

import torch
from torchvision.transforms import transforms

def load_model(model_name: str):

    if model_name == "arcface_o":

        from face_recognition.arcface.model import load_arcface as arcfaceO_model

        model = arcfaceO_model()
        trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    elif model_name == "curricularface":

        from face_recognition.curricularface.model_irse import load_curricularface as curricularface_model

        model = curricularface_model()
        trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    elif model_name == "cosface":
    
        from face_recognition.cosface.iresnet import load_cosface as cosface_model

        model = cosface_model()

        trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    elif model_name == "adaface":

        from face_recognition.adaface.model import load_adaface as adaface_model

        model = adaface_model()

        trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Lambda(lambda x: x[(2, 0, 1), :, :])
        ])
    elif model_name == "transface":

        from face_recognition.transface.model import load_transface as transface_model

        model = transface_model()

        trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: np.transpose(np.array(x), (2,0,1))),
            transforms.Lambda(lambda x: torch.from_numpy(x).float()),
            transforms.Lambda(lambda x: x.div_(255).sub_(0.5).div_(0.5))
        ])
    else:
        raise NotImplemented

    return model, trans