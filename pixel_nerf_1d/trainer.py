import torch
import torch.nn as nn

import os

from PIL import Image


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from options import PixelNeRF1DOptions
from model import PixelNeRFMLP
from dataset import PairDataset

from accelerate.utils import set_seed

from tqdm import tqdm

class Trainer:
    def __init__(self, opts):
        self.opts = opts


        #config model
        self.model = PixelNeRFMLP(self.opts).to(self.opts.device)

        #config dataset
        self.dataset = PairDataset(self.opts)

        self.test_data = PairDataset(self.opts, set = 'test')

        #config optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.opts.lr)
        #config loss
        self.criterion = nn.MSELoss()


    def run(self):
        pbar = tqdm(range(self.opts.num_steps))
        data_idx = 0
        for step in pbar:
            self.model.train()
            data, data_pair = self.dataset.__getitem__(data_idx)
            data_idx = (data_idx + 1) % self.opts.num_funcs
            x = torch.linspace(0, 10, self.opts.num_points).unsqueeze(0).unsqueeze(0).to(self.opts.device)
            #normalize data
            x = x / 10.

            prior_input = data.unsqueeze(0).unsqueeze(0)

            y_pred = self.model(x, prior_input).squeeze(1)

            loss = self.criterion(y_pred, data_pair)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #test
            self.model.eval()
            with torch.no_grad():
                data_test, data_pair_test = self.test_data.__getitem__(0)
                x_test = torch.linspace(0, 10, self.opts.num_points).unsqueeze(0).unsqueeze(0).to(self.opts.device)
                x_test = x_test / 10.
                prior_input_test = data_test.unsqueeze(0).unsqueeze(0)
                #prior_input_test = torch.zeros_like(prior_input_test)
                y_pred_test = self.model(x_test, prior_input_test).squeeze(1)
                loss_test = self.criterion(y_pred_test, data_pair_test)

                #plot
                if step % 100 == 0:
                    plt.plot(x.squeeze().cpu().numpy(), data_test.cpu().numpy(), label = 'prior')
                    plt.plot(x.squeeze().cpu().numpy(), data_pair_test.cpu().numpy(), label = 'ground truth')
                    plt.plot(x.squeeze().cpu().numpy(), y_pred_test.squeeze().cpu().numpy(), label = 'prediction')
                    plt.title('Step: {}, Loss = {:.4f}'.format(step, loss_test.item()))
                    plt.legend()
                    #plt.savefig(os.path.join('step_{}.png'.format(step)))
                    plt.draw()
                    plt.pause(0.5)
                    plt.clf()
                    #plt.close()


            pbar.set_description('Step: {}, Loss = {:.4f}'.format(step, loss_test.item()))


            pbar.update(1)
            #time.sleep(0.1)




if __name__ == '__main__':
    opts = PixelNeRF1DOptions()

    set_seed(0)
    trainer= Trainer(opts)
    trainer.run()