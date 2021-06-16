import torch
from numba import *
import itertools
import numpy as np
from TransformationsAndPenalties import *

class PlanCost:

    def __init__(self, args):
        self.args=args
        self.Archetypes=self.args[0]
        self.PartitionLength= np.int(np.sqrt((len(self.Archetypes[0]))))
        self.NumberOfAtoms = self.PartitionLength ** 2

        #Creating the cost Matrix
        self.Partition = torch.linspace(0, 1, self.PartitionLength)
        self.couples = np.array(np.meshgrid(self.Partition, self.Partition)).T.reshape(-1, 2)
        self.x=np.array(list(itertools.product(self.couples, repeat=2)))
        self.x = torch.from_numpy(self.x)
        self.a = self.x[:, 0]
        self.b = self.x[:, 1]

        self.CostMatrix = torch.linalg.norm(self.a - self.b, axis=1) ** 2
        self.CostMatrix = self.CostMatrix + torch.zeros((self.args[2],len(self.CostMatrix)))
        self.Profile0   = self.Archetypes[0] + torch.zeros((self.args[2],len(self.Archetypes[0])))
        self.Profile1   = self.Archetypes[1] + torch.zeros((self.args[2],len(self.Archetypes[1])))
        self.CostMatrix = self.CostMatrix.to(args[3])
        self.Profile0   = self.Profile0.to(args[3])
        self.Profile1   = self.Profile1.to(args[3])
    def Cost(self,Batch):

        #We apply transformation
        Batch=Transformation(Batch,self.args[1])
        #Calculating the transportaion cost
        TransportationCost = torch.sum(self.CostMatrix* Batch, axis=1)

        #Calculating the Margins of each plan.
        Plan=Batch.view(-1,  self.NumberOfAtoms, self.NumberOfAtoms)

        ArchetypeMargin = torch.sum(Plan.transpose(1,2), axis=2)
        BarycenterMargin  = torch.sum(Plan, axis=2)

        FirstPenalty      =  torch.linalg.norm( BarycenterMargin - self.Profile0, axis=1)
        SecondPenalty     =  torch.linalg.norm( ArchetypeMargin - self.Profile1, axis=1)

        TotalCost = TransportationCost + 5*FirstPenalty + 5*SecondPenalty
        return TotalCost, ArchetypeMargin, BarycenterMargin

class PlanCost:

    def __init__(self, args):
        self.args=args
        self.Archetypes=self.args[0]
        self.PartitionLength= np.int(np.sqrt((len(self.Archetypes[0]))))
        self.NumberOfAtoms = self.PartitionLength ** 2

        #Creating the cost Matrix
        self.Partition = torch.linspace(0, 1, self.PartitionLength)
        self.couples = np.array(np.meshgrid(self.Partition, self.Partition)).T.reshape(-1, 2)
        self.x=np.array(list(itertools.product(self.couples, repeat=2)))
        self.x = torch.from_numpy(self.x)
        self.a = self.x[:, 0]
        self.b = self.x[:, 1]

        self.CostMatrix = torch.linalg.norm(self.a - self.b, axis=1) ** 2
        self.CostMatrix = self.CostMatrix + torch.zeros((self.args[2],len(self.CostMatrix)))
        self.Profile0   = self.Archetypes[0] + torch.zeros((self.args[2],len(self.Archetypes[0])))
        self.Profile1   = self.Archetypes[1] + torch.zeros((self.args[2],len(self.Archetypes[1])))
        self.CostMatrix = self.CostMatrix.view(self.args[2],self.NumberOfAtoms,self.NumberOfAtoms).to(args[3])
        self.Profile0   = self.Profile0.to(args[3])
        self.Profile1   = self.Profile1.to(args[3])
        self.Barycenter=self.Profile0.repeat(1,self.NumberOfAtoms).view(self.args[2],self.NumberOfAtoms,self.NumberOfAtoms).transpose(1,2)

    def Cost(self,Batch):

        #We creating the kernel
        Kernel = Batch.view(-1, self.NumberOfAtoms, self.NumberOfAtoms)
        Kernel = Transformation(Kernel,"Sigmoid")
        #print(Kernel)
        Plan=(self.Barycenter*Kernel)
        #Calculating the transportaion cost
        TransportationCost = torch.sum(torch.sum(self.CostMatrix* Plan, axis=2),axis=1)

        #Calculating the Margins of each plan.


        ArchetypeMargin = torch.sum(Plan.transpose(1,2), axis=2)

        #print(ArchetypeMargin)
        SecondPenalty     =  torch.linalg.norm(ArchetypeMargin - self.Profile1, axis=1)**2

        TotalCost =  TransportationCost + 5*SecondPenalty
        #print(TotalCost.min())

        return TotalCost, Kernel

class KernelCost:

    def __init__(self, args):
        self.args=args
        self.Archetypes=self.args[0].to(args[3])
        self.PartitionLength= np.int(np.sqrt((len(self.Archetypes[0]))))
        self.NumberOfAtoms = self.PartitionLength ** 2

        #Creating the cost Matrix
        self.Partition = torch.linspace(0, 1, self.PartitionLength)
        self.couples = np.array(np.meshgrid(self.Partition, self.Partition)).T.reshape(-1, 2)
        self.x=np.array(list(itertools.product(self.couples, repeat=2)))
        self.x = torch.from_numpy(self.x)
        self.a = self.x[:, 0]
        self.b = self.x[:, 1]

        self.CostMatrix = (torch.linalg.norm(self.a - self.b, axis=1) ** 2).to(self.args[3])
        self.Archetypes   = self.Archetypes + torch.zeros((self.args[2],len(self.args[0]),len(self.Archetypes[0]))).to(self.args[3])



    def Cost(self, Batch, Barycenter, Kernel ,ProfileNumber, RowNumber):
        Barycenter=Transformation(Barycenter,"Sigmoid")
        CostRow= self.CostMatrix[RowNumber*self.NumberOfAtoms:(RowNumber+1)*self.NumberOfAtoms]+torch.zeros((self.args[2], self.NumberOfAtoms)).to(self.args[3])

        #We creating the kernel

        CalibratedBatch = Batch+Kernel[RowNumber*self.NumberOfAtoms:(RowNumber+1)*self.NumberOfAtoms]
        TransformedBatch = Transformation(CalibratedBatch,"Sigmoid")

        Row= Barycenter[RowNumber]*TransformedBatch
        TransportationCost = torch.sum(CostRow * Row, axis=1)



        TransformedKernel=Transformation(Kernel.view(self.NumberOfAtoms,self.NumberOfAtoms),"Sigmoid")
        Plan = (Barycenter.unsqueeze(0).transpose(0,1)* TransformedKernel)
        ModifiedPlan=Plan.detach().clone()
        ModifiedPlan[RowNumber]=torch.zeros(self.NumberOfAtoms).to(self.args[3])

        ArchetypeMargin =   torch.sum(ModifiedPlan, axis=0)
        ArchetypeMargin =    Row+ArchetypeMargin
        SecondPenalty     =  torch.linalg.norm(ArchetypeMargin - self.Archetypes[:,ProfileNumber,:], axis=1)**2


        TotalCost =  TransportationCost + 50*SecondPenalty -2
        return TotalCost, torch.sum(torch.sum(self.CostMatrix.view(-1,self.NumberOfAtoms,self.NumberOfAtoms)[0] * Plan, axis=1), axis=0)



