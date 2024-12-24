import logging
logging.basicConfig(level=logging.INFO)
import numpy as np

from lib.models import PseudoModel, Model, load_model, load_model_identifier, load_default_hooks, load_default_nodes
from bonner.models.hooks import Hook



VISSL_TASKS = [
    "Jigsaw-ImageNet1K",
    "RotNet-ImageNet1K",
    "ClusterFit-16K-RotNet-ImageNet1K",
    "NPID++-ImageNet1K",
    "PIRL-ImageNet1K",
    "SimCLR-ImageNet1K",
    "SwAV-ImageNet1K",
    "DeepClusterV2-ImageNet1K",
]
TORCHVISION_ARCHS = [
    "resnet18",
    "resnet50",
    "resnext50_32x4d",
    "wide_resnet50_2",
    "alexnet",
    "vgg16",
    "densenet121",
    "squeezenet1_1",
    "shufflenet_v2_x1_0",
    "convnext_tiny",
    "swin_t",
    "maxvit_t",
]
TIMM_ARCHS = [
    "cait_xxs24_224",
    "coat_lite_tiny",
    "deit_tiny_patch16_224",
    "levit_128",
    "mixer_b16_224",
    "resmlp_12_224",
    "dla34",
]
VGGPLACES_WEIGHTS = [
    "Places365",
    "Hybrid1365",
]
IPCL_WEIGHTS = [
    "ref06_diet_imagenet",
    "ref07_diet_openimagesv6",
    "ref08_diet_places2",
    "ref09_diet_vggface2",
]


def load_model_by_identifier(yielder_name: str, identifier: str) -> Model:
    sid = None
    for i, v in enumerate(load_yielder(yielder_name, identifier_only=True)()):
        if v == identifier:
            sid = i
            break
    for m in yield_models(load_yielder(yielder_name), start_idx=sid, num_yield=1):
        return m


def yield_models(
    yielder: callable,
    num_yield: int = 9999,
    start_idx: int = 0,
):
    for idx, model in enumerate(yielder()):
        if idx < start_idx:
            continue
        yield model
        if idx == (start_idx + num_yield - 1):
            break


def _resnet50_imagenet1k_varied_tasks(
    loader: callable,
    hooks: (dict | str),
    hooks_kws: dict,
) -> callable:
    def resnet50_imagenet1k_varied_tasks():
        yield loader(
            architecture="ResNet50",
            source="torchvision",
            weights="IMAGENET1K_V1",
            hooks=hooks,
            hooks_kws=hooks_kws,
        )
        for task in VISSL_TASKS:
            yield loader(
                architecture="ResNet50",
                source="vissl",
                weights=task,
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
        # yield loader(
        #     architecture="ResNet50",
        #     source="barlowtwins",
        #     weights="barlowtwins-ImageNet1K",
        #     hooks=hooks,
        #     hooks_kws=hooks_kws,
        # )
    return resnet50_imagenet1k_varied_tasks


def _classification_imagenet1k_varied_architectures(
    loader: callable,
    hooks: (dict | str),
    hooks_kws: dict,
) -> callable:
    def classification_imagenet1k_varied_architectures():
        for arch in TORCHVISION_ARCHS:
            yield loader(
                architecture=arch,
                source="torchvision",
                weights="IMAGENET1K_V1",
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
        for arch in TIMM_ARCHS:
            yield loader(
                architecture=arch,
                source="timm",
                weights="IMAGENET1K_V1",
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
    return classification_imagenet1k_varied_architectures


def _untrained_varied_architectures(
    loader: callable,
    hooks: (dict | str),
    hooks_kws: dict,
) -> callable:
    def untrained_varied_architectures():
        seed = 0
        for arch in TORCHVISION_ARCHS:
            yield loader(
                architecture=arch,
                source="torchvision",
                weights="untrained",
                seed=seed,
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
            seed += 1
        for arch in TIMM_ARCHS:
            yield loader(
                architecture=arch,
                source="timm",
                weights="untrained",
                seed=seed,
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
            seed += 1
    return untrained_varied_architectures


def _untrained_resnet18_varied_seeds(
    loader: callable,
    hooks: (dict | str),
    hooks_kws: dict,
) -> callable:
    def untrained_resnet18_varied_seeds():
        for seed in range(1, 100, 5):
            yield loader(
                architecture="ResNet18",
                source="torchvision",
                weights="untrained",
                seed=seed,
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
    return untrained_resnet18_varied_seeds


def _resnet18_classification_imagenet1k_varied_seeds(
    loader: callable,
    hooks: (dict | str),
    hooks_kws: dict,
) -> callable:
    def resnet18_classification_imagenet1k_varied_seeds():
        for seed in range(1, 100, 5):
            yield loader(
                architecture="ResNet18",
                source="model_zoo",
                weights="tiny_imagenet",
                seed=seed,
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
    return resnet18_classification_imagenet1k_varied_seeds

def _varied_visual_diets(
    loader: callable,
    hooks: (dict | str),
    hooks_kws: dict,
) -> callable:
    def varied_visual_diets():
        for weights in IPCL_WEIGHTS:
            yield loader(
                architecture="alexnetgn",
                source="ipcl",
                weights=weights,
                hooks=hooks,
                hooks_kws=hooks_kws,
            )
    return varied_visual_diets
        

def _pseudo_models_topleft_outlier(identifier_only) -> callable:
    odd_ranks = [1, 0]
    if identifier_only:
        def pseudo_models_topleft_outlier():
            for odd_rank in odd_ranks:
                yield PseudoModel(
                    outlier_type="topleft",
                    odd_rank=odd_rank,
                ).identifier
    else:
        def pseudo_models_topleft_outlier():
            for odd_rank in odd_ranks:
                yield PseudoModel(
                    outlier_type="topleft",
                    odd_rank=odd_rank,
                )
    return pseudo_models_topleft_outlier


def _pseudo_models_bottomright_outlier(identifier_only) -> callable:
    noise_indices = np.arange(10)
    if identifier_only:
        def pseudo_models_bottomright_outlier():
            for noise_index in noise_indices:
                yield PseudoModel(
                    outlier_type="bottomright",
                    noise_index=noise_index,
                ).identifier
    else:
        def pseudo_models_bottomright_outlier():
            for noise_index in noise_indices:
                yield PseudoModel(
                    outlier_type="bottomright",
                    noise_index=noise_index,
                )
    return pseudo_models_bottomright_outlier
       

def load_yielder(
    yielder_name: str, 
    identifier_only: bool = False,
    hooks: (dict | str) = "global_maxpool",
    hooks_kws: dict = {},
) -> callable:
    loader = load_model_identifier if identifier_only else load_model
    match yielder_name:
        case "resnet50_imagenet1k_varied_tasks":
            return _resnet50_imagenet1k_varied_tasks(loader, hooks, hooks_kws)
        case "classification_imagenet1k_varied_architectures":
            return _classification_imagenet1k_varied_architectures(loader, hooks, hooks_kws)
        case "untrained_varied_architectures":
            return _untrained_varied_architectures(loader, hooks, hooks_kws)
        case "untrained_resnet18_varied_seeds":
            return _untrained_resnet18_varied_seeds(loader, hooks, hooks_kws)
        case "resnet18_classification_imagenet1k_varied_seeds":
            return _resnet18_classification_imagenet1k_varied_seeds(loader, hooks, hooks_kws)
        case "varied_visual_diets":
            return _varied_visual_diets(loader, hooks, hooks_kws)
        case "pseudo_models_topleft_outlier":
            return _pseudo_models_topleft_outlier(identifier_only)
        case "pseudo_models_bottomright_outlier":
            return _pseudo_models_bottomright_outlier(identifier_only)
        case _:
            raise ValueError(f"Unknown yielder: {yielder_name}")
        
    