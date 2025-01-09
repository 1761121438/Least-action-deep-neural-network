from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
import math, torch, time, os
import torch.nn as nn
import numpy as np
import argparse
import random

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed(111)
torch.cuda.manual_seed_all(111)

parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--e', type=int, default=2000, help='Epochs')
parser.add_argument('--d', type=int, default=3, help='depth+1')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--nx', type=int, default=128, help='Sampling axis x')
parser.add_argument('--nt', type=int, default=128, help='Sampling axis t')
parser.add_argument('--T', type=float, default=1, help='T')
parser.add_argument('--xi', type=float, default=1e-5, help='Threshold')
parser.add_argument('--b1', type=int, default=100, help='penalty for boundary')
parser.add_argument('--b2', type=int, default=0, help='penalty for dE/dt=0')

args = parser.parse_args()


def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def errorFun(output, target, params):
    error = output - target
    error = math.sqrt(torch.mean(error * error))
    ref = math.sqrt(torch.mean(target * target))
    return error / (ref + params["minimal"])


def exact(X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    return np.sin(x + t)


def exact_t(X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    return np.cos(x + t)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Net(torch.nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()
        self.params = params
        self.device = device
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        nn.init.xavier_normal_(self.linearIn.weight)
        nn.init.constant_(self.linearIn.bias, 0)

        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.m = nn.Linear(self.params["width"], self.params["width"])
            nn.init.xavier_normal_(self.m.weight)
            nn.init.constant_(self.m.bias, 0)
            self.linear.append(self.m)

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])
        nn.init.xavier_normal_(self.linearOut.weight)
        nn.init.constant_(self.linearOut.bias, 0)

    def forward(self, X):
        x = torch.tanh(self.linearIn(X))
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        x = self.linearOut(x)

        tt = X[:, 0:1]
        xx = X[:, 1:2]
        TT = self.params["T"]
        x = torch.sin(tt * np.pi / TT) * x + torch.sin(TT - tt) * torch.sin(xx) / np.sin(TT) + torch.sin(
            tt) * torch.sin(xx + TT) / np.sin(TT)
        # x = tt * (TT - tt) * x + (TT - tt) * torch.sin(x) / TT + tt * torch.sin(xx + TT) / TT
        return x


def train(model, device, params, optimizer, scheduler):
    t = np.linspace(0, params["T"], params["nt"])
    x = np.linspace(-1 * np.pi, np.pi, params["nx"])
    T, X = np.meshgrid(t, x)
    data_train = np.concatenate([T.flatten()[:, None], X.flatten()[:, None]], axis=1)
    Data_train = torch.from_numpy(data_train).float().to(device)
    Data_train = Data_train.requires_grad_(True)

    t_b = params["T"] * np.random.rand(params["nt"], 1)
    data_l = np.concatenate([t_b, -1 * np.pi * np.ones((params["nt"], 1))], axis=1)
    data_r = np.concatenate([t_b, np.pi * np.ones((params["nt"], 1))], axis=1)

    Data_l = torch.from_numpy(data_l).float().to(device)
    Data_l = Data_l.requires_grad_(True)
    Data_r = torch.from_numpy(data_r).float().to(device)
    Data_r = Data_r.requires_grad_(True)

    t = np.linspace(0, params["T"], 400)
    x = np.linspace(-1 * np.pi, np.pi, 400)
    T, X = np.meshgrid(t, x)
    data_test = np.concatenate([T.flatten()[:, None], X.flatten()[:, None]], axis=1)
    u_test = exact(data_test)

    Data_test = torch.from_numpy(data_test).float().to(device)
    U_test = torch.from_numpy(u_test).float().to(device)

    start_time = time.time()
    total_start_time = time.time()

    Test = []
    Loss_s = []
    Loss_bc = []
    Loss_dE = []
    Num = []
    Energy = []
    Residual = []

    Step = []
    Time = []
    loss_epoch_list = []
    loss_std_list = []
    loss_mean_list = []

    loss_mean_last = 1
    for step in range(params["trainstep"]):
        U_pred = model(Data_train)
        U_pred_l = model(Data_l)
        U_pred_r = model(Data_r)

        U_t = GetGradients(U_pred, Data_train)[:, 0:1]
        U_x = GetGradients(U_pred, Data_train)[:, 1:2]

        U_tt = GetGradients(U_t, Data_train)[:, 0:1]
        U_xx = GetGradients(U_x, Data_train)[:, 1:2]
        U_xt = GetGradients(U_x, Data_train)[:, 0:1]

        # LAP
        L = 0.5 * (U_t ** 2 - U_x ** 2)
        S = torch.sum(L) * 2 * math.pi * params["T"] / (params["nx"] * params["nt"])
        loss_res = S

        # brdy
        loss_b = torch.mean(torch.square(U_pred_l - U_pred_r))

        # Residual
        Res = torch.mean(torch.square(U_tt - U_xx))

        # energy
        energy_x = 0.5 * (U_t ** 2 + U_x ** 2)
        energy_x_1 = energy_x.reshape(params["nx"], params["nt"])
        energy = torch.sum(energy_x_1, 0) * 2 * math.pi / params["nx"]

        # dE/dt
        DotEt_x = U_t * U_tt + U_x * U_xt
        DotEt_x_1 = DotEt_x.reshape(params["nx"], params["nt"])
        DotEt = torch.sum(DotEt_x_1, 0) * 2 * math.pi / params["nx"]
        loss_dE = torch.mean(torch.square(DotEt))

        model.zero_grad()
        loss = loss_res + params["beta1"] * loss_b + params["beta2"] * loss_dE

        loss_stop = loss_res.cpu().detach().numpy()
        loss_epoch_list.append(loss_stop)

        if step % 100 == 0 and step != 0:
            loss_std = np.std(np.array(loss_epoch_list[(step - 100):step]))
            loss_std_list.append(loss_std)
            loss_mean = np.mean(np.array(loss_epoch_list[(step - 100):step]))
            loss_mean_list.append(loss_mean)

            loss_mean_dis = np.sqrt(np.mean(np.square(loss_mean - loss_mean_last))) / np.sqrt(
                np.mean(np.square(loss_mean_last)))
            loss_mean_last = loss_mean

            if abs(loss_mean_dis) < params["xi"] or step == params["trainstep"] - 100:
                total_time = time.time() - total_start_time
                print('%% U no longer adapts, training stop')
                print('--------stop_step: %d' % step)
                print('--------final energy: %.3e' % loss_res)
                print("Training costs %s seconds." % (total_time))

                Step.append(step)
                Time.append(total_time)
                break

        if step % params["Writestep"] == 0:
            elapsed = time.time() - start_time

            U_pred_test = model(Data_test)

            test_error = errorFun(U_pred_test, U_test, params)
            print('Epoch: %d, Time: %.2f, loss_r: %.3e, Loss_bc: %.3e, Loss_dE: %.3e, Residual: %.3e, test: %.3e' %
                  (step, elapsed, loss_res, loss_b, loss_dE, Res, test_error))

            start_time = time.time()
            Loss_s.append(loss_res.cpu().detach().numpy())
            Loss_bc.append(loss_b.cpu().detach().numpy())
            Loss_dE.append(loss_dE.cpu().detach().numpy())
            Residual.append(Res.cpu().detach().numpy())
            Test.append(test_error)

            U_pred_test = model(Data_test)
            Num.append(np.squeeze(U_pred_test.cpu().detach().numpy()))
            Energy.append(energy.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()

    folder = './LAP_[0,{T}]_Depth{d}_Width{w}_nx{nx}_nt{nt}_beta1_{b1}_beta2_{b2}'.format(T=params["T"],
                                                                                          d=params["depth"] + 1,
                                                                                          w=params["width"],
                                                                                          nx=params["nx"],
                                                                                          nt=params["nt"],
                                                                                          b1=params["beta1"],
                                                                                          b2=params["beta2"])

    os.mkdir(folder)
    np.savetxt(folder + "/test.csv", Test)
    np.savetxt(folder + "/loss_res.csv", Loss_s)
    np.savetxt(folder + "/loss_b.csv", Loss_bc)
    np.savetxt(folder + "/loss_dE.csv", Loss_dE)
    np.savetxt(folder + "/Residual.csv", Residual)

    np.savetxt(folder + "/Numerical.txt", Num)
    np.savetxt(folder + "/Energy.txt", Energy)
    np.savetxt(folder + "/Step.csv", Step)
    np.savetxt(folder + "/Time.csv", Time)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    # stop condition
    params["xi"] = args.xi

    # architecture
    params["d"] = 2
    params["width"] = args.n
    params["depth"] = args.d
    params["dd"] = 1
    # sampling
    params["T"] = args.T
    params["nt"] = args.nt
    params["nx"] = args.nx
    # optimize
    params["beta1"] = args.b1
    params["beta2"] = args.b2
    params["lr"] = 0.001  # Learning rate
    params["step_size"] = 500  # lr decay
    params["gamma"] = 0.1  # lr decay rate
    params["trainstep"] = args.e
    params["Writestep"] = 100
    params["minimal"] = 10 ** (-14)

    startTime = time.time()
    model = Net(params, device).to(device)
    print("Generating network costs %s seconds." % (time.time() - startTime))
    print(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    train(model, device, params, optimizer, scheduler)
    print("The number of parameters is %s," % count_parameters(model))


if __name__ == "__main__":
    main()
