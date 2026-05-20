from torch import nn
from torchvision.models import resnet18, resnet34

from src.models.vgg_cifar import get_vgg16_bn_cifar


def _adapt_resnet_for_cifar(model, num_classes):
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18_cifar(num_classes=10):
    return _adapt_resnet_for_cifar(resnet18(weights=None), num_classes)


def get_resnet34_cifar(num_classes=10):
    return _adapt_resnet_for_cifar(resnet34(weights=None), num_classes)


def get_model(name="resnet18", num_classes=10):
    name = str(name).lower()
    if name == "resnet18":
        return get_resnet18_cifar(num_classes=num_classes)
    if name == "resnet34":
        return get_resnet34_cifar(num_classes=num_classes)
    if name == "vgg16_bn":
        return get_vgg16_bn_cifar(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
