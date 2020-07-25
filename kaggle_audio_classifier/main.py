#options
from options import options
from data_loader import Audio_Dataloader, read_data
import torch
from torchvision import datasets, transforms
from network import discriminator
import torch.optim as optim
from train import trainer
import torch.nn as nn


options = options()
opts = options.parse()

data_reader = read_data(opts)
train_filename, test_filename = data_reader.train_test_split()

train_data = Audio_Dataloader(train_filename, opts, data_reader.name_to_label_dict)
test_data = Audio_Dataloader(test_filename, opts, data_reader.name_to_label_dict)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers = opts.cpu_count)
testloader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch, shuffle=True, num_workers = opts.cpu_count)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d = discriminator(opts, data_reader.return_class_size()).to(device)

#optimizers
D_optim = optim.Adam(d.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
criterion = nn.CrossEntropyLoss()
train = trainer(opts)

#train the model
train.train(d, D_optim, criterion, trainloader)

#test the model
if opts.resume:
    print('\nLoading the model\n')
    d = torch.load(opts.model_path)
    d.eval()

train.test(d, testloader)

