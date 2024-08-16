from bonner.models import zoo
from bonner.models.hooks import Hook, GlobalMaxpool, GlobalAveragePool, RandomProjection

from lib.models import Model

def load_default_nodes(architecture: str) -> list[str]:
    match architecture:
        case "resnet18":
            return (
                ["relu"]
                + [f"layer{i}.{j}.relu" for i in range(1, 5) for j in range(2) ]
            )
        case "resnet50" | "resnext50_32x4d" | "wide_resnet50_2":
            return (
                ["relu"]
                + [f"layer1.{i}.relu_2" for i in range(3)]
                + [f"layer2.{i}.relu_2" for i in range(4)]
                + [f"layer3.{i}.relu_2" for i in range(6)]
                + [f"layer4.{i}.relu_2" for i in range(3)]
            )
        case "alexnet":
            return [f"features.{i}" for i in [1, 4, 7, 9, 11]]
        case "vgg16":
            return [
                f"features.{i}" 
                for i in [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
            ]
        case "densenet121":
            return ((
                ["features.relu0"]
                + [f"features.denseblock1.denselayer{i}.relu2" for i in range(1, 7)]
                + [f"features.denseblock2.denselayer{i}.relu2" for i in range(1, 13)]
                + [f"features.denseblock3.denselayer{i}.relu2" for i in range(1, 25)]
                + [f"features.denseblock4.denselayer{i}.relu2" for i in range(1, 17)]
                + [f"features.transition{i}.relu" for i in range(1, 4)]
            ))
        case "squeezenet1_1":
            return (
                ["features.1"]
                + [
                    f"features.{j}.{i}"
                    for i in ["squeeze_activation", "expand1x1_activation", "expand3x3_activation"]
                    for j in [3, 4, 6, 7, 9, 10, 11, 12]
                ]
            )
        case "shufflenet_v2_x1_0":
            return (
                [f"conv{i}.2" for i in [1, 5]]
                + [f"stage{i}.0.branch1.4" for i in range(2, 5)]
                + [f"stage{i}.0.branch2.{j}" for i in range(2, 5) for j in [2, 7]]
                + [f"stage{i}.{j}.branch2.{k}" for i in [2, 4] for j in range(1, 4) for k in [2, 7]]
                + [f"stage3.{i}.branch2.{j}" for i in range(1, 8) for j in [2, 7]]
            )
        case "convnext_tiny":
            return (
                [f"features.{i}.{j}.block.4" for i in [1, 3, 7] for j in range(0, 3, 2)]
                + [f"features.5.{i}.block.4" for i in range(0, 9, 2)]
            )
        case "swin_t":
            return (
                [f"features.{i}.{j}.{k}" for i in [1, 3, 7] for j in range(2) for k in ["norm1", "norm2", "mlp"]]
                + [f"features.5.{i}.{j}" for i in range(6) for j in ["norm1", "norm2", "mlp"]]
                + ["norm"]
            )
        case "maxvit_t":
            return [f"blocks.{i[0]}.layers.{i[1]}.layers.{j}_attention.{k}" for i in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (2, 4), (3, 0), (3, 1)] for j in["window", "grid"] for k in ["attn_layer.0", "mlp_layer.2"] ]
        case "cait_xxs24_224":
            return (
                ["patch_embed.norm"]
                + [f"blocks.{i}.{j}" for i in range(24) for j in ["norm1", "norm2", "mlp.act"] ]
                + ["norm"]
            )
        case "coat_lite_tiny":
            return (
                [f"serial_blocks{i}.{j}.norm{k}.layer_norm" for i in range(1, 5) for j in range(2) for k in [1, 2] ]
                + [f"serial_blocks{i}.{j}.mlp.act" for i in range(1, 5) for j in range(2)]
            )
        case "deit_tiny_patch16_224":
            return (
                ["patch_embed.norm"]
                + [f"blocks.{i}.{j}" for i in range(12) for j in ["norm1", "norm2", "mlp.act"] ]
                + ["norm"]
            )
        case "levit_128":
            return (
                [f"stem.act{i}" for i in range(1,4)]
                + [f"stages.{i}.blocks.{j}.{k}.act" for i in range(3) for j in range(4) for k in ["attn.proj", "mlp"]]
            )
        case "mixer_b16_224":
            return (
                [f"blocks.{i}.{j}" for i in range(1, 12, 2) for j in ["norm1", "norm2", "mlp_channels.act"]]
                + ["norm"]
            )
        case "resmlp_12_224":
            return (
                [f"blocks.{i}.{j}" for i in range(12) for j in ["norm1", "norm2", "mlp_channels.act"]]
                + ["norm"]
            )
        case "dla34":
            return (
                ["base_layer.2"]
                + [f"level{i}.2" for i in range(2)]
                + [f"level{i}.{j}.relu" for i in [2, 5] for j in ["tree1", "tree2", "root"]]
                + [f"level{i}.tree{j}.{k}.relu" for i in [3, 4] for j in [1, 2] for k in ["tree1", "tree2", "root"]]
            )


def _load_default_hooks(architecture: str, hooks: str, hook_kws: dict) -> Hook:
    match hooks:
        case "global_maxpool":
            match architecture:
                case (
                    "convnext_tiny"
                    | "swin_t"
                    | "maxvit_t"
                ):
                    return GlobalMaxpool(amax_dim=[-3, -2], **hook_kws)
                case _:
                    return GlobalMaxpool(**hook_kws)
        case "random_projection":
            return RandomProjection(**hook_kws)
        case None:
            return None
        

def load_default_hooks(
    architecture: str,
    nodes: list[str],
    hooks: str,
    hook_kws: dict,
) -> dict[str, Hook]:
    hook = _load_default_hooks(architecture, hooks, hook_kws)
    if hook:
        return {n: hook for n in nodes}
    else:
        return {}
  
    
def load_model(
    architecture: str,
    source: str,
    weights: str = None,
    nodes: list[str] = None,
    hooks: dict[str, Hook] = "global_maxpool",
    hooks_kws: dict = {},
    seed: int = 11,   
) -> Model:
    identifier = (
        f"{architecture.lower()}"
        f".weights={weights.lower()}"
    )
    match source:
        case "vissl":
            model, preprocess = zoo.load_vissl_model(
                architecture=architecture,
                weights=weights
            )
        case "torchvision":
            model, preprocess = zoo.load_pytorch_model(
                architecture=architecture,
                weights=weights,
                seed=seed,
            )
            if weights == "untrained":
                identifier += f".seed={seed}"
        case "model_zoo":
            model, preprocess = zoo.load_model_zoo_model(
                seed=seed,
            )
            identifier += f".seed={seed}"
        case "timm":
            model, preprocess = zoo.load_timm_model(
                architecture=architecture,
                pretrained=(weights is None or weights != "untrained"),
                seed=seed,
            )
            if weights == "untrained":
                identifier += f".seed={seed}"
        case _:
            raise ValueError(f"Source {source} not supported.")
    nodes = nodes if nodes else load_default_nodes(architecture.lower())
    hooks = hooks if isinstance(hooks, dict) else load_default_hooks(architecture.lower(), nodes, hooks, hooks_kws)
    return Model(
        model=model,
        preprocess=preprocess,
        identifier=identifier,
        nodes=nodes,
        hooks=hooks,
    )
    
def load_model_identifier(
    architecture: str,
    source: str,
    weights: str = None,
    seed: int = 11, 
    hooks: dict[str, Hook] = "global_maxpool",
    hooks_kws: dict = {},      
) -> Model:
    identifier = (
        f"{architecture.lower()}"
        f".weights={weights.lower()}"
    )
    match source:
        case "vissl":
            pass
        case "torchvision" | "timm":
            if weights == "untrained":
                identifier += f".seed={seed}"
        case "model_zoo":
            identifier += f".seed={seed}"
        case _:
            raise ValueError(f"Source {source} not supported.")
    return identifier
