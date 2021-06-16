import torch

def Transformation(Sample, Transformationfunction):
    Batchdims = Sample.dim()-1
    if Transformationfunction == "Sigmoid":
        Sample = torch.exp(Sample)
        BatchNorm = torch.sum(Sample, axis=Batchdims, keepdims=True)
        Sample = Sample / BatchNorm
    return Sample
