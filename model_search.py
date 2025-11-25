import numpy as np
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
import collections
import os, csv
'''
# State-of-the-art Mixed Operation from P-DARTS
class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p, prefix="", suffix=""):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C, stride, False, prefix=prefix+'_'+str(i), suffix=suffix+'_'+str(i))
                if 'pool' in primitive:
                    op = nn.Sequential(
                        collections.OrderedDict([
                            ("mixop_"+str(i), op),
                            ("mixop_bn_"+str(i), nn.BatchNorm2d(C, affine=False)),
                        ])
                    )
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(
                        collections.OrderedDict([
                            ("mixop_"+str(i), op),
                            ("mixop_dropout_"+str(i), nn.Dropout(self.p)),
                        ])
                    )
                self.m_ops.append(op)

    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))
'''

# Refined Edge Topology
class MixedOp(nn.Module):
    # My Proposed Final
    def __init__(self, C, stride, switch, p, prefix="", suffix=""):
        super(MixedOp, self).__init__()
        self.active_indices = [i for i, active in enumerate(switch) if active]
        self.k = len(self.active_indices)
        self.C_per_op = C // self.k
        self.p = p

        # Point-wise compression: compress input from C → C/k
        self.compress = ReLUConvBN(C, self.C_per_op, 1, 1, 0, affine=False)
        # Define active operations
        self.ops = nn.ModuleList()
        for idx in self.active_indices:
            primitive = PRIMITIVES[idx]
            op = OPS[primitive](self.C_per_op, stride, False)

            if 'pool' in primitive:
                op = nn.Sequential(
                    collections.OrderedDict([
                        (f"mixop_{idx}", op),
                        (f"mixop_bn_{idx}", nn.BatchNorm2d(self.C_per_op, affine=False)),
                    ])
                )
            if isinstance(op, Identity) and p > 0:
                op = nn.Sequential(
                    collections.OrderedDict([
                        (f"mixop_{idx}", op),
                        (f"mixop_dropout_{idx}", nn.Dropout(p)),
                    ])
                )
            self.ops.append(op)

    def update_p(self):
        for op in self.ops:
            if isinstance(op, nn.Sequential):
                for module in op:
                    if isinstance(module, nn.Dropout):
                        module.p = self.p

    def forward(self, x, weights):
        x_compressed = self.compress(x)  # shape: [B, C_per_op, H, W]

        # Apply each op, weighted by corresponding alpha weight
        weighted_outputs = [
            w * op(x_compressed)
            for w, op in zip(weights, self.ops)
        ]

        return torch.cat(weighted_outputs, dim=1)  # shape: [B, C, H, W]

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p, prefix="", suffix=""):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False, prefix="cell_preprocess0_"+prefix, suffix=""+suffix)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False, prefix="cell_preprocess0_"+prefix, suffix=""+suffix)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False, prefix="cell_preprocess1_"+prefix, suffix=""+suffix)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p, prefix="", suffix="")
                self.cell_ops.append(op)
                switch_count = switch_count + 1

    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, switches_normal=[],
                 switches_reduce=[], p=0.0):
        super(Network, self).__init__()
        #self._C = C
        # Compute max number of active operations (True entries) across any edge
        max_true_ops = max(sum(s) for s in switches_normal)  # or switches_reduce

        if C % max_true_ops != 0:
            adjusted_C = C - (C % max_true_ops)
            print(f"[Info] Adjusting C from {C} → {adjusted_C} to match max active ops ({max_true_ops})")
            C = adjusted_C
        self._C = C  # use self._C throughout the rest of the model
        # force C to be divisible by number of operations

        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p
        self.switches_normal = switches_normal
        switch_ons = []
        for i in range(len(switches_normal)):
            ons = 0
            for j in range(len(switches_normal[i])):
                if switches_normal[i][j]:
                    ons = ons + 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]
        #self.switch_on = len(PRIMITIVES)

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce,
                            self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal,
                            self.p)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    # weights = torch.sigmoid(self.alphas_reduce)
                    weights = F.softmax(self.alphas_reduce , dim=0)
                else:
                    # weights = torch.sigmoid(self.alphas_reduce)
                    weights = F.softmax(self.alphas_reduce , dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    # weights = torch.sigmoid(self.alphas_normal)
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    # weights = torch.sigmoid(self.alphas_normal)
                    weights = F.softmax(self.alphas_normal , dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        # logits = self.classifier(out)
        # logits = logits.view(logits.size(0), -1)
        return logits

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = self.switch_on
        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        #self.alphas_normal = nn.Parameter(torch.full((k, num_ops), 0.5))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_ops)))
        #self.alphas_reduce = nn.Parameter(torch.full((k, num_ops), 0.5))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters