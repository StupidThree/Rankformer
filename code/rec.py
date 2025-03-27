import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import args


def sparse_sum(values, indices0, indices1, n):
    if indices0 is None:
        assert (len(indices1.shape) == 1 and values.shape[0] == indices1.shape[0])
    else:
        assert (len(indices0.shape) == 1 and len(indices1.shape) == 1)
        assert (indices0.shape[0] == indices1.shape[0])
    # assert (len(values.shape) <= 2)
    return torch.zeros([n]+list(values.shape)[1:], device=values.device, dtype=values.dtype).index_add(0, indices1, values if indices0 is None else values[indices0])


def rest_sum(values, indices0, indices1, n):
    return values.sum(0).unsqueeze(0)-sparse_sum(values, indices0, indices1, n)


class GCN(nn.Module):
    def __init__(self, dataset, alpha=1.0, beta=0.0):
        super(GCN, self).__init__()
        self.dataset = dataset
        self.alpha, self.beta = alpha, beta

    def forward(self, x, u, i):
        n, m = self.dataset.num_users, self.dataset.num_items
        du = sparse_sum(torch.ones_like(u), None, u, n).clamp(1)
        di = sparse_sum(torch.ones_like(i), None, i, m).clamp(1)
        w1 = (torch.ones_like(u)/du[u].pow(self.alpha)/di[i].pow(self.beta)).unsqueeze(-1)
        w2 = (torch.ones_like(u)/du[u].pow(self.beta)/di[i].pow(self.alpha)).unsqueeze(-1)
        xu, xi = torch.split(x, [n, m])
        zu = sparse_sum(xi[i]*w1, None, u, n)
        zi = sparse_sum(xu[u]*w2, None, i, m)
        return torch.concat([zu, zi], 0)


class Rankformer(nn.Module):
    def __init__(self, dataset, alpha):
        super(Rankformer, self).__init__()
        self.dataset = dataset
        self.my_parameters = []
        self.alpha = alpha

    def forward(self, x, u, i):
        n, m = self.dataset.num_users, self.dataset.num_items
        dui = sparse_sum(torch.ones_like(u), None, u, n)
        duj = m-dui
        dui, duj = dui.clamp(1).unsqueeze(1), duj.clamp(1).unsqueeze(1)
        xu, xi = torch.split(F.normalize(x), [n, m])
        vu, vi = torch.split(x, [n, m])
        xui = (xu[u]*xi[i]).sum(1).unsqueeze(1)
        sxi = sparse_sum(xi, i, u, n)
        sxj = xi.sum(0)-sxi
        svi = sparse_sum(vi, i, u, n)
        svj = vi.sum(0)-svi
        b_pos = (xu*sxi).sum(1).unsqueeze(1)/dui
        b_neg = (xu*sxj).sum(1).unsqueeze(1)/duj
        if args.del_benchmark:
            b_pos, b_neg = 0, 0
        xxi = xi.unsqueeze(1)*xi.unsqueeze(2)
        xvi = xi.unsqueeze(1)*vi.unsqueeze(2)
        du1 = (xu*sxi).sum(1).unsqueeze(1)/dui-b_neg+self.alpha
        du2 = -(xu*sxj).sum(1).unsqueeze(1)/duj+b_pos+self.alpha
        di1 = (xi*sparse_sum(xu/dui, u, i, m)).sum(1).unsqueeze(1)+sparse_sum((-b_neg+self.alpha)/dui, u, i, m)
        di2 = -(xi*rest_sum(xu/duj, u, i, m)).sum(1).unsqueeze(1)+rest_sum((b_pos+self.alpha)/duj, u, i, m)
        A = sparse_sum(xui*vi[i], None, u, n)
        zu1 = A/dui-svi*(b_neg-self.alpha)/dui
        zu2 = (torch.mm(xu, (xvi).sum(0))-A)/duj-svj*(b_pos+self.alpha)/duj
        zi1 = sparse_sum(xui*vu[u]/dui[u], None, i, m)-sparse_sum(vu*(b_neg-self.alpha)/dui, u, i, m)
        zi2 = torch.mm(xi, ((xu/duj).unsqueeze(2)*vu.unsqueeze(1)).sum(0))-sparse_sum(xui*(vu/duj)[u], None, i, m) \
            - rest_sum(vu*(b_pos+self.alpha)/duj, u, i, m)
        z1 = torch.concat([zu1, zi1], 0)
        z2 = torch.concat([zu2, zi2], 0)
        d1 = torch.concat([du1, di1], 0).clamp(args.rankformer_clamp_value)
        d2 = torch.concat([du2, di2], 0).clamp(args.rankformer_clamp_value)
        if args.del_neg:
            z2, d2 = 0, 0
        z, d = z1+z2, d1+d2
        if args.del_omega_norm:
            return z
        return z/d
