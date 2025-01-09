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
parser.add_argument('--e', type=int, default=5000, help='Epochs')
parser.add_argument('--d', type=int, default=3, help='depth+1')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--T', type=float, default=6, help='T')
parser.add_argument('--nt', type=int, default=1024, help='Sampling')
parser.add_argument('--pe', type=int, default=100, help='Penalty')
parser.add_argument('--xi', type=float, default=1e-5, help='Threshold')

args = parser.parse_args()


def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def exact(t):
    return np.cos(t)


def exact_t(t):
    return -1 * np.sin(t)


def errorFun(output, target, params):
    error = output - target
    error = math.sqrt(torch.mean(error * error))
    ref = math.sqrt(torch.mean(target * target))
    return error / (ref + params["minimal"])


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

    def forward(self, t):
        x = torch.tanh(self.linearIn(t))
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        x = self.linearOut(x)

        TT = self.params["T"]
        a = exact(0)
        b = exact(TT)
        x = t * (t - TT) * x + (TT - t) * a / TT + b * t / TT
        return x


def train(model, device, params, optimizer, scheduler):
    t = np.linspace(0, params["T"], params["nt"])[:, None]

    T_train = torch.from_numpy(t).float().to(device)
    T_train = T_train.requires_grad_(True)


    t_test = np.linspace(0, params["T"], params["ntest"])[:, None]
    x_test = exact(t_test)
    T_test = torch.from_numpy(t_test).float().to(device)
    X_test = torch.from_numpy(x_test).float().to(device)

    start_time = time.time()
    total_start_time = time.time()
    Loss = []
    Test = []
    Energy = []
    Res = []
    Step = []
    Time = []

    loss_epoch_list = []
    loss_std_list = []
    loss_mean_list = []

    loss_mean_last = 1

    for step in range(params["trainstep"]):
        X_pred = model(T_train)

        X_t = GetGradients(X_pred, T_train)[:, 0:1]
        L = X_t ** 2 / 2 - X_pred ** 2 / 2
        loss_res = torch.sum(L) * params["T"] / params["nt"]

        energy = X_t ** 2 / 2 + X_pred ** 2 / 2

        X_tt = GetGradients(X_t, T_train)[:, 0:1]
        dE = X_t * (X_tt + X_pred)
        loss_dE = torch.mean(torch.square(dE))

        res = torch.mean(torch.square(X_tt + X_pred))

        model.zero_grad()
        loss = loss_res + params["penalty"] * loss_dE

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

            X_pred_test = model(T_test)
            test_error = errorFun(X_pred_test, X_test, params)
            print(
                'Epoch: %d, Time: %.2f, S: %.3e, Test: %.3e' %
                (step, elapsed, loss_res, test_error))

            start_time = time.time()
            Loss.append(loss_res.cpu().detach().numpy())
            Test.append(test_error)
            Energy.append(np.squeeze(energy.cpu().detach().numpy()))
            Res.append(res.cpu().detach().numpy())

        loss.backward()
        optimizer.step()
        scheduler.step()

    X_pred_test = model(T_test)
    Num = X_pred_test.cpu().detach().numpy()

    folder = './LAP_[{i},{T}]_Depth{d}_Width{w}_nt{nt}_pe{pe}'.format(i=0,
                                                                      T=params["T"],
                                                                      d=params["depth"] + 1,
                                                                      w=params["width"],
                                                                      nt=params["nt"],
                                                                      pe=params["penalty"])

    os.mkdir(folder)
    np.savetxt(folder + "/loss.csv", Loss)
    np.savetxt(folder + "/test.csv", Test)
    np.savetxt(folder + "/Res.csv", Res)
    np.savetxt(folder + "/Numerical.txt", Num)
    np.savetxt(folder + "/Energy.txt", Energy)
    np.savetxt(folder + "/Step.csv", Step)
    np.savetxt(folder + "/Time.csv", Time)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    # stop condition
    params["xi"] = args.xi
    # T and sampling
    params["T"] = args.T
    params["nt"] = args.nt

    # architecture
    params["width"] = args.n
    params["depth"] = args.d

    # learning rate
    params["lr"] = 0.001
    params["step_size"] = 1000
    params["gamma"] = 0.5
    params["penalty"] = args.pe

    # fixed paramsters
    params["d"] = 1
    params["ntest"] = 2000
    params["dd"] = 1
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
