import os
import copy
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from .fedavg import FedAvg

import sys
sys.path.insert(0, '../')
from utils import compute_accuracy
import time
class FedOptim(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(FedOptim, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)


    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # FedAvgM, FedAdam, FedAdagrad
        parser.add_argument('--server_optimizer', type=str, default='gd',
                            help='the server optimizer. \
                            gd corresponds to FedAvgM and adam corresponds to FedAdam')
        parser.add_argument('--server_momentum', type=float, default=0.9,
                            help='the first order server momentum')
        parser.add_argument('--server_momentum_second', type=float, default=0.99,
                            help='the second order server momentum')
        parser.add_argument('--server_learning_rate', type=float, default=1.0,
                            help='Server learning rate of fedadam/fedyogi')
        parser.add_argument('--tau', type=float, default=0.001,
                            help='tau introduced in FedAdam paper. \
                            Essentially, this hyper-parameter provides \
                            numeric protection for second-order momentum')

        # feddecorr arguments
        parser.add_argument('--feddecorr', action='store_true',
                            help='whether to use FedDecorr')
        parser.add_argument('--feddecorr_coef', type=float, default=0.1,
                            help='coefficient of the FedDecorr loss')
        return parser.parse_args(extra_args)


    # function that executing the federated training
    import time  # 确保文件开头已导入

    def run_fed(self):
        round_times = []  # 用于存储每轮耗时

        for comm_round in range(self.args.n_comm_round):
            if getattr(self.args, 'measure_time', False):
                round_start = time.perf_counter()

            # 原有代码：记录日志、本地训练、聚合、评估等
            self.logger.info("in comm round:" + str(comm_round))

            # do local training on each party
            nets_this_round = self.local_training(comm_round)

            # conduct global aggregation
            self.global_aggregation(nets_this_round)

            # conduct server update (FedAvgM 特有部分)
            # ... (原有服务器更新逻辑) ...

            # compute acc
            self.global_net.cuda()
            train_acc, train_loss = compute_accuracy(self.global_net, self.global_train_dl)
            test_acc, test_loss = compute_accuracy(self.global_net, self.test_dl)
            self.global_net.to('cpu')

            # logging numbers (原有)
            self.logger.info('>> Global Model Train accuracy: %f' % train_acc)
            self.logger.info('>> Global Model Test accuracy: %f' % test_acc)
            self.logger.info('>> Global Model Train loss: %f' % train_loss)

            if (comm_round + 1) % self.args.print_interval == 0:
                print('round: ', str(comm_round))
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Train loss: %f' % train_loss)

            if (comm_round + 1) % self.args.save_interval == 0:
                torch.save(self.global_net.state_dict(),
                           os.path.join(self.args.ckptdir, self.args.approach,
                                        'globalmodel_' + self.args.log_file_name + '.pth'))
                torch.save(self.party2nets[0].state_dict(),
                           os.path.join(self.args.ckptdir, self.args.approach,
                                        'localmodel0_' + self.args.log_file_name + '.pth'))

            # 计时结束
            if getattr(self.args, 'measure_time', False):
                round_time = time.perf_counter() - round_start
                round_times.append(round_time)
                self.logger.info(f"通信轮次 {comm_round + 1} 耗时: {round_time:.4f} 秒")

        # 所有轮次结束后输出统计
        if getattr(self.args, 'measure_time', False) and len(round_times) > 0:
            avg_time = sum(round_times) / len(round_times)
            var = sum((t - avg_time) ** 2 for t in round_times) / len(round_times)
            std_time = var ** 0.5
            self.logger.info(f"平均每轮通信耗时: {avg_time:.4f} ± {std_time:.4f} 秒")
            print(f"平均每轮通信耗时: {avg_time:.4f} ± {std_time:.4f} 秒")
