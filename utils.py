import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import attridict
import gymnasium as gym
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def getEnvProperties(env):
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discreteActionBool = True
        actionSize = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discreteActionBool = False
        actionSize = env.action_space.shape[0]
    else:
        raise Exception
    return observationShape, discreteActionBool, actionSize


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


class Moments(nn.Module):
    def __init__( self, device, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()


def symlog(x):
    return torch.sign(x)*torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x)*torch.expm1(torch.abs(x))


class TwoHotDistribution:
    def __init__(self, logits, low = -10, high = 10):
        self.probabilities    = F.softmax(logits, dim=-1)
        self.logProbabilities = F.log_softmax(logits, dim=-1)
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)

    @property
    def mean(self):
        return symexp((self.probabilities * self.bins).sum(dim=-1))

    def log_prob(self, x):
        x = symlog(x)
        indexLow  = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        indexLow  = torch.maximum(indexLow, torch.zeros_like(indexLow))
        
        indexHigh = indexLow + 1
        indexHigh = torch.minimum(indexHigh, torch.full_like(indexHigh, len(self.bins) - 1))

        equal = indexLow == indexHigh
        distanceToLow  = torch.where(equal, 1, torch.abs(self.bins[indexLow]  - x))
        distanceToHigh = torch.where(equal, 1, torch.abs(self.bins[indexHigh] - x))
        total = distanceToLow + distanceToHigh
        weightLow  = distanceToHigh / total
        weightHigh = distanceToLow  / total

        target = F.one_hot(indexLow, len(self.bins)).squeeze(-2)*weightLow + F.one_hot(indexHigh, len(self.bins)).squeeze(-2)*weightHigh
        return (target*self.logProbabilities).sum(dim=-1)
    
