import cv2
import gym
import numpy as np

from collections import deque
from gym import spaces

import time

import gym
import gym.spaces
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import os
import datetime
import json

class DQN0(nn.Module):
    def __init__(self, num_actions):
        super(DQN0, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        # print("01----------------------------------------",out.shape)
        out = out.view(out.size(0), -1)
        # print("02----------------------------------------",out.shape)
        out = F.relu(self.fc1(out))
        # print("03----------------------------------------",out.shape)
        out = self.fc2(out)

        return out


class DQN1(nn.Module):
    def __init__(self, num_actions):
        super(DQN1, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        # print("10----------------------------------------",out.shape)

        out = F.relu(self.conv2(out))
        # print("11----------------------------------------",out.shape)

        out = F.relu(self.conv3(out))
        # print("12----------------------------------------",out.shape)

        out = out.view(out.size(0), -1)
        # print("13----------------------------------------",out.shape)
        out = F.relu(self.fc1(out))
        # print("14----------------------------------------",out.shape)
        out = self.fc2(out)

        return out
