
import torch
from collections import deque
import numpy as np
import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class trainer():

    def __init__(self, opts):
        self.opts = opts

    def train(self, D, D_solver, criterion, dataloader):

        steps = 0
        loss_deque = deque(maxlen=100)
        train_loss = []

        for epoch in range(self.opts.epoch):
            running_loss = 0
            correct = 0
            total = 0

            for x, y in dataloader:
                if len(x) != self.opts.batch:
                    continue
                D_solver.zero_grad()

                '''Real Images'''
                output = D(x.to(device))  # returns logit of real data.
                #print(output.shape)

                loss = criterion(output, y.to(device))
                loss.backward()
                D_solver.step()  # One step Descent into loss

                correct_, total_ = util.prediction_accuracy(output, y.to(device))

                correct += correct_
                total += total_

                loss_deque.append(loss.cpu().item())
                train_loss.append(np.mean(loss_deque))

                if steps % self.opts.print_every == 0:
                    print("Epoch: {}/{}...".format(epoch + 1, self.opts.epoch),
                          "LossL {:.4f}".format(running_loss / self.opts.print_every),
                          "Running Accuracy {:4f}".format(correct / np.float(total)))

                    running_loss = 0

        util.raw_score_plotter(train_loss)
        if self.opts.save_progress:
            print('\nSaving the model\n')
            torch.save(D.state_dict(), self.opts.model_path)
    def test(self, D, dataloader):

        correct = 0
        total = 0

        D.eval()

        for x, y in dataloader:

            '''Real Images'''
            output = D(x.to(device))  # returns logit of real data.
            #print(output.shape)

            correct_, total_ = util.prediction_accuracy(output, y.to(device))

            correct += correct_
            total += total_

        print("Test Accuracy {:4f}".format(correct / np.float(total)))

