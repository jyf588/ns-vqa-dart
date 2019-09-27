import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


PYTORCH_VER = torch.__version__


class AttributeNetwork():

    def __init__(self, opt):    
        if opt.concat_img:
            if opt.with_depth:
                self.input_channels = 8
            else:
                self.input_channels = 6
        else:
            self.input_channels = 3

        if opt.load_checkpoint_path:
            print('| loading checkpoint from %s' % opt.load_checkpoint_path)
            checkpoint = torch.load(opt.load_checkpoint_path)
            if self.input_channels != checkpoint['input_channels']:
                raise ValueError('Incorrect input channels for loaded model')
            self.output_dim = checkpoint['output_dim']
            self.net = _Net(self.output_dim, self.input_channels)
            self.net.load_state_dict(checkpoint['model_state'])
        else:
            print('| creating new model')
            output_dims = {
                'clevr': 18,    #TODO
                'clevr_dart': (9+3+9, 9+3)
            }
            self.output_dim = output_dims[opt.dataset]
            if opt.with_rot:
                self.output_dim = self.output_dim[0]
            else:
                self.output_dim = self.output_dim[1]
            self.net = _Net(self.output_dim, self.input_channels)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.learning_rate)

        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.net.cuda(opt.gpu_ids[0])

        self.input, self.label = None, None
                
    def set_input(self, x, y=None):
        self.input = self._to_var(x)
        if y is not None:
            self.label = self._to_var(y)

    def step(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss.backward()
        self.optimizer.step()

    def forward(self):
        self.pred = self.net(self.input)
        if self.label is not None:
            self.loss = self.criterion(self.pred, self.label)
            if self.output_dim == 21:   # TODO: this is with orientation
                self.old_loss = self.criterion(self.pred[:12], self.label[:12])
                self.rot_loss = self.criterion(self.pred[12:], self.label[12:])
            
    def get_loss(self):
        # print(PYTORCH_VER)
        if PYTORCH_VER.startswith('1.'):   # TODO
            return self.loss.data.item()
        else:
            return self.loss.data[0]

    def get_old_loss(self):
        if self.output_dim != 21:
            return None
        if PYTORCH_VER.startswith('1.'):   # TODO
            return self.old_loss.data.item()
        else:
            return self.old_loss.data[0]

    def get_rot_loss(self):
        if self.output_dim != 21:
            return None
        if PYTORCH_VER.startswith('1.'):   # TODO
            return self.rot_loss.data.item()
        else:
            return self.rot_loss.data[0]

    def get_pred(self):
        return self.pred.data.cpu().numpy()

    def eval_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def save_checkpoint(self, save_path):
        checkpoint = {
            'input_channels': self.input_channels,
            'output_dim': self.output_dim,
            'model_state': self.net.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if self.use_cuda:
            self.net.cuda(self.gpu_ids[0])

    def _to_var(self, x):
        if self.use_cuda:
            x = x.cuda()
        return Variable(x)


class _Net(nn.Module):

    def __init__(self, output_dim, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet34(pretrained=False)
        layers = list(resnet.children())
        
        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        import torch.nn.functional as F
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


def get_model(opt):
    model = AttributeNetwork(opt)
    return model