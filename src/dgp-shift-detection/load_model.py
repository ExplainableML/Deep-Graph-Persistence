from mlp import MLP
from argparse import Namespace

registered_models = dict()


def register_model(name):
    def decorator(model_loader):
        registered_models[name] = model_loader
        return model_loader

    return decorator


@register_model("mlp")
def load_mlp(in_size, num_classes: int, parameters: Namespace) -> MLP:
    return MLP(
        input_size=in_size,
        num_classes=num_classes,
        hidden_size=parameters.hidden,
        num_layers=parameters.layers,
        activation_function=parameters.activation,
        initialization=parameters.initialization,
        dropout=parameters.dropout,
    )


def load_model(name: str, in_size, num_classes: int, parameters: Namespace):
    if name not in registered_models:
        raise ValueError(f"Model {name} not registered")

    return registered_models[name](in_size, num_classes, parameters)
