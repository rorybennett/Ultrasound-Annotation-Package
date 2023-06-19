import torch
from torch import nn

from IPV.Code import parameters as pms


class Quadruplet(nn.Module):

    def __init__(self, num_of_points):
        super(Quadruplet, self).__init__()

        self.num_of_points = num_of_points

        num_of_classes = [len(l) for p in range(self.num_of_points) for l in pms.tasks_classes]

        self.net1 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False).cuda().half()
        self.net1.fc = torch.nn.Linear(512, 128)

        self.net2 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False).cuda().half()
        self.net2.fc = torch.nn.Linear(512, 128)

        self.net3 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False).cuda().half()
        self.net3.fc = torch.nn.Linear(512, 128)

        self.net4 = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False).cuda().half()
        self.net4.fc = torch.nn.Linear(512, 128)

        self.l1 = torch.nn.Linear(512, num_of_classes[0])
        self.l2 = torch.nn.Linear(512, num_of_classes[1])
        self.l3 = torch.nn.Linear(512, num_of_classes[0])
        self.l4 = torch.nn.Linear(512, num_of_classes[1])
        if num_of_points == 4:
            self.l5 = torch.nn.Linear(512, num_of_classes[0])
            self.l6 = torch.nn.Linear(512, num_of_classes[1])
            self.l7 = torch.nn.Linear(512, num_of_classes[0])
            self.l8 = torch.nn.Linear(512, num_of_classes[1])

    def forward(self, x):
        net1_out = self.net1(x[:, 0])
        net2_out = self.net2(x[:, 1])
        net3_out = self.net3(x[:, 2])
        net4_out = self.net4(x[:, 3])

        net_out = torch.cat((net1_out, net2_out, net3_out, net4_out), 1)

        x1 = self.l1(net_out)
        x2 = self.l2(net_out)
        x3 = self.l3(net_out)
        x4 = self.l4(net_out)
        if self.num_of_points == 4:
            x5 = self.l5(net_out)
            x6 = self.l6(net_out)
            x7 = self.l7(net_out)
            x8 = self.l8(net_out)

        if self.num_of_points == 2:
            return x1, x2, x3, x4
        else:
            return x1, x2, x3, x4, x5, x6, x7, x8
