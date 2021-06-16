import torch
from numba import *
import numpy as np
from Sampling import *
from Objective import*
import time
from TransformationsAndPenalties import *

InitialTime=time.time()
if torch.cuda.is_available():
    device=torch.device("cuda")

SampleSize=10000
Dim=8
#Provide the Images/Archetypes
A1=torch.cat((torch.ones(Dim, dtype=torch.float64), torch.zeros(Dim**2-Dim)), 0)
A1= A1 / torch.sum(A1)
A2=torch.cat((torch.zeros(Dim**2-Dim), torch.ones(Dim)), 0)
A2= A2 / torch.sum(A2)
Archetypes = torch.stack([A1, A2], dim=0)
NumberOfAtoms= len(Archetypes[0])
PlanSize=NumberOfAtoms**2


MeanMatrix= torch.ones(NumberOfAtoms,dtype=torch.double).to(device)/NumberOfAtoms
CovMatrix =torch.eye(NumberOfAtoms,dtype=torch.double).to(device)


#MeanMatrix =torch.load("MeanMatrix.pt")
#CovMatrix =torch.load("CovMatrix.pt")
Loglikelihood=KernelCost((Archetypes,"Sigmoid",SampleSize,device))
Loglikelihood2=KernelCost((Archetypes,"Sigmoid",1,device))

FlattedKernel=MeanMatrix.repeat(NumberOfAtoms)
Sample = SampleGeneration(1, SampleSize * NumberOfAtoms)
Sample = Sample.view(SampleSize, NumberOfAtoms)


def MeanMatrixCalculator(FlattedKernel,SampleSize,SamplingIterations,uniformlength):
    StartTime=time.time()
    for i in range(SamplingIterations):
        BroadcastedSample=torch.cat([Sample,torch.zeros(SampleSize*NumberOfAtoms*(NumberOfAtoms-1)).view(SampleSize,-1).to(device)],dim=1)
        Input=BroadcastedSample+FlattedKernel
        PreWeights, _= Loglikelihood.Cost(Input)
        Weights = torch.exp(- 200*PreWeights)
        Product = torch.sum(Sample.T * Weights, axis=1)
        WeightSum = torch.sum(Weights)
        FlattedKernel[:NumberOfAtoms] =FlattedKernel[:NumberOfAtoms] + Product / WeightSum



        for j in range(1,NumberOfAtoms-1):
            #Sample = SampleGeneration(1, SampleSize * NumberOfAtoms)
            #Sample = Sample.view(SampleSize, NumberOfAtoms)
            BroadcastedSample = torch.cat(
                [torch.zeros(SampleSize * j* NumberOfAtoms).view(SampleSize, -1).to(device),
                 Sample, torch.zeros(SampleSize * (NumberOfAtoms-j-1) *NumberOfAtoms ).view(SampleSize, -1).to(device)],
                dim=1)

            Input = BroadcastedSample + FlattedKernel
            PreWeights, _ = Loglikelihood.Cost(Input)
            Weights = torch.exp(- 200*PreWeights)
            Product = torch.sum(Sample.T * Weights, axis=1)
            WeightSum = torch.sum(Weights)
            FlattedKernel[j*NumberOfAtoms:(j+1)*NumberOfAtoms] = FlattedKernel[j*NumberOfAtoms:(j+1)*NumberOfAtoms] +Product / WeightSum


        BroadcastedSample = torch.cat(
            [torch.zeros(SampleSize * NumberOfAtoms * (NumberOfAtoms - 1)).view(SampleSize, -1).to(device),Sample],
            dim=1)
        #Sample = SampleGeneration(1, SampleSize * NumberOfAtoms)
        #Sample = Sample.view(SampleSize, NumberOfAtoms)
        Input = BroadcastedSample + FlattedKernel
        PreWeights, _ = Loglikelihood.Cost(Input)
        Weights = torch.exp(- 200*PreWeights)
        Product = torch.sum(Sample.T * Weights, axis=1)
        WeightSum = torch.sum(Weights)
        FlattedKernel[(NumberOfAtoms-1)*NumberOfAtoms:] = FlattedKernel[(NumberOfAtoms-1)*NumberOfAtoms:]  +Product / WeightSum



    return FlattedKernel




A=time.time()
FlattedKernel=MeanMatrixCalculator(FlattedKernel,SampleSize,500,0.1)
print(Loglikelihood2.Cost(FlattedKernel.unsqueeze(0)))
print(time.time()-A)

FlattedKernel=MeanMatrix.repeat(NumberOfAtoms)
Sample = SampleGeneration(0.7, SampleSize * NumberOfAtoms)
Sample = Sample.view(SampleSize, NumberOfAtoms)
FlattedKernel=MeanMatrixCalculator(FlattedKernel,SampleSize,500,0.3)
print(Loglikelihood2.Cost(FlattedKernel.unsqueeze(0)))

FlattedKernel=MeanMatrix.repeat(NumberOfAtoms)
Sample = SampleGeneration(0.5, SampleSize * NumberOfAtoms)
Sample = Sample.view(SampleSize, NumberOfAtoms)
FlattedKernel=MeanMatrixCalculator(FlattedKernel,SampleSize,500,0.5)
print(Loglikelihood2.Cost(FlattedKernel.unsqueeze(0)))

#torch.save(MeanMatrix,'MeanMatrix.pt')
#torch.save(CovMatrix,'CovMatrix.pt')
