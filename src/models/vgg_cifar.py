import torch
from torch import nn
from torchvision.models import vgg16_bn


def _init_classifier(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_vgg16_bn_cifar(num_classes=10):
    model = vgg16_bn(weights=None)
    model.avgpool = nn.Identity()
    with torch.no_grad():
        features = model.features(torch.zeros(1, 3, 32, 32))
        feature_dim = int(features.flatten(1).shape[1])
    model.classifier = nn.Sequential(
        nn.Linear(feature_dim, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    model.classifier.apply(_init_classifier)
    return model
