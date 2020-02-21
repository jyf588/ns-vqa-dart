import argparse
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import sys

sys.path.append("/home/michelle/workspace/ns-vqa-dart")
from bullet.profiler import Profiler


PYTORCH_VER = torch.__version__


class AttributeNetwork:
    def __init__(self, opt: argparse.Namespace):
        """
        Args:
            opt: Various options to the network.
        """
        self.profiler = Profiler()
        self.mode = None

        if opt.concat_img:
            if opt.with_depth:
                self.input_channels = 8
            else:
                self.input_channels = 6
        else:
            self.input_channels = 3

        if opt.load_checkpoint_path:
            print("| loading checkpoint from %s" % opt.load_checkpoint_path)
            checkpoint = torch.load(opt.load_checkpoint_path)
            if self.input_channels != checkpoint["input_channels"]:
                raise ValueError("Incorrect input channels for loaded model")
            self.output_dim = checkpoint["output_dim"]
            self.net = _Net(self.output_dim, self.input_channels)
            self.net.load_state_dict(checkpoint["model_state"])
        else:
            print("| creating new model")
            label2dim = {
                "attr": 7,
                "size": 2,
                "position": 3,
                "up_vector": 3,
                "height": 1,
            }
            if opt.dataset in ["dash", "clevr_dart"]:
                output_dim = 0
                if opt.pred_attr:
                    output_dim += label2dim["attr"]
                if opt.pred_size:
                    output_dim += label2dim["size"]
                if opt.pred_position:
                    output_dim += label2dim["position"]
                if opt.pred_up_vector:
                    output_dim += label2dim["up_vector"]
                assert output_dim > 0
                self.output_dim = output_dim
            elif opt.dataset == "clevr":
                self.output_dim = 18
            else:
                raise ValueError(f"Unsupported dataset: {opt.dataset}")
            self.net = _Net(self.output_dim, self.input_channels)

        if opt.fp16:
            self.half_net()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=opt.learning_rate
        )

        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.net.cuda(opt.gpu_ids[0])

        self.opt = opt
        self.input, self.label = None, None

    def set_input(self, x, y=None):
        """Stores x and y.

        Args:
            x: The input data to the network.
            y: The labels to train on.
        """
        self.input = self._to_var(x)
        if self.opt.fp16:
            self.input = self.input.half()

        if y is not None:
            self.label = self._to_var(y)

    def step(self):
        self.optimizer.zero_grad()
        if self.opt.fp16:
            self.half_net()
        self.forward()
        self.loss.backward()
        self.net.float()
        self.optimizer.step()

    def forward(self):
        self.pred = self.net(self.input)
        if self.label is not None:
            if self.opt.fp16:
                self.pred = self.pred.float()
            self.loss = self.criterion(self.pred, self.label)

    def get_loss(self):
        # print(PYTORCH_VER)
        if PYTORCH_VER.startswith("1."):  # TODO
            return self.loss.data.item()
        else:
            return self.loss.data[0]

    def get_pred(self):
        return self.pred.data.cpu().numpy()

    def eval_mode(self):
        if self.opt.fp16:
            self.half_net()
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def half_net(self):
        self.net.half()

        # https://medium.com/@dwightfoster03/fp16-in-pytorch-a042e9967f7e
        # for layer_i, layer in enumerate(self.net.modules()):
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.float()

    def save_checkpoint(self, save_path):
        checkpoint = {
            "input_channels": self.input_channels,
            "output_dim": self.output_dim,
            "model_state": self.net.cpu().state_dict(),
        }
        torch.save(checkpoint, save_path)
        if self.use_cuda:
            self.net.cuda(self.gpu_ids[0])

    def _to_var(self, x: torch.Tensor) -> Variable:
        """Converts a tensor into a variable.

        Args:
            x: A tensor.
        
        Returns:
            A variable.
        """
        if self.use_cuda:
            x = x.cuda()
        var = Variable(x)
        return var


class _Net(nn.Module):
    def __init__(self, output_dim, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet18(pretrained=False)
        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(
            0,
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
        )

        self.main = nn.Sequential(*layers)
        self.fc1 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # Reshape to (B, -1)
        output = self.fc1(x)  # Get the final outputs.
        return output


def get_model(opt):
    model = AttributeNetwork(opt)
    return model
