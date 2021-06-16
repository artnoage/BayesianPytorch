import torch
import numpy as np
import random
from Sampling import *
from Objective import*
import time
from TransformationsAndPenalties import *

if torch.cuda.is_available():
    device=torch.device("cuda")


#Provide the Images/Archetypes
Dim=5
A1=torch.cat((torch.ones(Dim,dtype=torch.float64), torch.zeros(Dim**2-Dim)), 0)
A1= A1 / torch.sum(A1)
A2=torch.cat((torch.zeros(Dim**2-Dim), torch.ones(Dim)), 0)
A2= A2 / torch.sum(A2)
Archetypes = torch.stack([A1, A2], dim=0).to(device)
NumberOfArchetypes = len(Archetypes)
NumberOfAtoms= len(Archetypes[0])
PlanSize=NumberOfAtoms**2


Barycenter =  torch.ones(Dim**2).to(device)
FlattedKernels  = (torch.ones(NumberOfAtoms).to(device)/NumberOfAtoms).repeat(NumberOfAtoms*NumberOfArchetypes).view(NumberOfArchetypes,-1)



def WassersteinDistancesEstimator(Loglikelihood,InitialFlattedKernels,SampleSize,KernelSamplingIterations,BarycenterSample):
    WassersteinDistancesEstimation=0
    Index = np.arange(0, NumberOfAtoms)
    for ProfileNumber in range(NumberOfArchetypes):
        FlattedKernel=InitialFlattedKernels[ProfileNumber].clone()
        for IterationNumber in range(KernelSamplingIterations):
            RowSample = SampleGeneration(1.2, SampleSize * NumberOfAtoms, device)
            RowSample = RowSample.view(SampleSize, NumberOfAtoms)
            RowNumber = random.choices(Index,Transformation(BarycenterSample,"Sigmoid"), k=1)[0]
            KernelPreWeights,Min = Loglikelihood.Cost(RowSample, BarycenterSample, FlattedKernel, ProfileNumber, RowNumber)
            KernelWeights = torch.exp(-80*KernelPreWeights)
            WeightedSampleSum = torch.sum(RowSample.T * KernelWeights, axis=1)
            WeightSum = torch.sum(KernelWeights)
            FlattedKernel[RowNumber*NumberOfAtoms:(RowNumber+1)*NumberOfAtoms] = FlattedKernel[RowNumber*NumberOfAtoms:(RowNumber+1)*NumberOfAtoms] +WeightedSampleSum / WeightSum

        WassersteinDistancesEstimation=WassersteinDistancesEstimation+Min
    print(WassersteinDistancesEstimation)
    return WassersteinDistancesEstimation


def BarycenterEstimator(InitialBarycenter,Archetypes,BarycenterSampleSize,KernelSampleSize,KernelSamplingIterations):
    Loglikelihood = KernelCost((Archetypes, "Sigmoid", KernelSampleSize, device))
    Barycenter=InitialBarycenter
    BarycenterWeightedSampleSum=0
    BarycenterWeightSum=0
    for BarycenterSampleIndex in range(BarycenterSampleSize):
        BarycenterSample = SampleGeneration(0.8,  NumberOfAtoms, device)
        CalibratedBarycenterSample = Barycenter+BarycenterSample
        BarycenterSamplePreweight=WassersteinDistancesEstimator(Loglikelihood, FlattedKernels,KernelSampleSize,KernelSamplingIterations,CalibratedBarycenterSample)
        BarycenterWeight=torch.exp(-80*BarycenterSamplePreweight)
        BarycenterWeightedSampleSum=BarycenterWeight*BarycenterSample+BarycenterWeightedSampleSum
        BarycenterWeightSum=BarycenterWeight+BarycenterWeightSum
        print(BarycenterSampleIndex)
    Barycenter=Barycenter+BarycenterWeightedSampleSum/BarycenterWeightSum

    return Barycenter

for i in range(10):
    A=time.time()
    Barycenter=BarycenterEstimator(InitialBarycenter=Barycenter,Archetypes=Archetypes,BarycenterSampleSize=100,KernelSampleSize=1000,KernelSamplingIterations=500)
    print("end")
    print(Transformation(Barycenter,"Sigmoid"))
    print(time.time()-A)