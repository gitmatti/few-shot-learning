import torch
from torch import nn
import torchvision.models as models

models.resnet18()


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class Identity(nn.Module):
    """Identity layer.

    # Arguments
        input: Input tensor
    """
    def forward(self, x):
        return x


class AdaptiveHead(nn.Module):
    r"""A collection of `torch.nn.Linear` with one active at a time, to be used
    as an adaptive head for a classification model.

    Args:
        in_features: in_features: size of each input sample
        out_features: list of sizes of output sample of each input sample
        bias: If set to ``False``, the heads will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`
          depends on the active head to be set by model.set_active(active).

    Attributes:
        fcs: list of `torch.nn.Linear`
        out_features: :math:`H_{out}` of the currently active head

    Examples::

        >>> model = AdaptiveHead(20, [5, 20])
        >>> input = torch.randn(128, 20)
        >>> output = model(input)
        >>> print(output.size())
        torch.Size([128, 5])
        >>> model.set_active(1)
        >>> output = model(input)
        >>> print(output.size())
        torch.Size([128, 20])
    """
    def __init__(self, in_features, out_features, bias=True):
        super(AdaptiveHead, self).__init__()

        self.in_features = in_features
        if not isinstance(out_features, list):
            self._out_features = [out_features]
        else:
            self._out_features = out_features

        for l, out_features in enumerate(self._out_features):
            setattr(self,
                    "fc{}".format(l),
                    nn.Linear(self.in_features, out_features, bias=bias))

        self._active = 0
        self._has_bias = bias

    def set_active(self, active):
        assert active < len(self._out_features)
        self._active = active

    @property
    def out_features(self):
        return self._out_features[self._active]
    
    @property
    def fc(self):
        return [getattr(self, "fc{}".format(l)) for l in self._out_features]
        
    def forward(self, inputs):
        fc = getattr(self, "fc{}".format(self._active))
        return fc(inputs)

    def extra_repr(self):
        out_features_str = "{}/" * len(self._out_features)
        out_features_str = out_features_str[:-1]
        out_features_str = out_features_str.format(*self._out_features)
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, out_features_str, self._has_bias is not None
        )


class AdaptiveHeadClassifier(nn.Module):
    r"""A classifier model with a (possibly pre-trained) model body and a
    flexible head to allow for fine-tuning and transfer learning.

    Args:
        out_features: list of sizes of output sample of each input sample
        architecture: one of torchvision's model architectures,
            e.g. ``'resnet50'``. Default ``'resnet18'``.
        pretrained: If set to ``True``, the model will be initialized
            with pre-trained weights. Default ``True``.
        freeze: If set to ``True``, the torchvision model will be used as a
            feature extractor with its weights frozen. The head will learn as
            usual. Default ``False``.

    Shape:
        - Input: depends on the model architecture. Typically :math:`(N, C, H, W)`
          where :math:`C, H, W` are the input channel, height and width.
        - Output: :math:`(N, H_{out})` where :math:`H_{out} = \text{out\_features}`
          depends on the active head, to be set by model.set_active(active)

    Attributes:
        model: one of torchvision's pre-defined models
        model_head: instance of `few-shot-learning.models.AdaptiveHead`
        out_features: :math:`H_{out}` of the currently active head

    Examples::

        >>> model = AdaptiveHeadClassifier([100, 5, 10])
        >>> input = torch.randn(128, 3, 28, 28)
        >>> output = model(input)
        >>> print(output.size())
        torch.Size([128, 100])
        >>> model.set_active(2)
        >>> output = model(input)
        >>> print(output.size())
        torch.Size([128, 10])
    """
    def __init__(self, out_features, architecture='resnet18', pretrained=True,
                 freeze=False):
        super(AdaptiveHeadClassifier, self).__init__()

        self._pretrained = pretrained
        self._architecture = architecture
        self._freeze = freeze

        self.model = models.__dict__[architecture](pretrained=pretrained)

        # freeze all lower layers of the network
        if self._freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        assert hasattr(self.model, "fc")
        assert hasattr(self.model.fc, "in_features")

        n_features = self.model.fc.in_features

        self.model_head = AdaptiveHead(n_features, out_features)
        self.model.fc = self.model_head

    def set_active(self, active):
        self.model_head.set_active(active)

    @property
    def out_features(self):
        return self.model_head.out_features

    def forward(self, inputs):
        return self.model(inputs)
