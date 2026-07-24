import argparse
import os
import sys
import torch
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path_dict = {
    "ads-b": {
        "name": "ads-b",
        "linux": "~/Datasets/ADS-B",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ADS-B"),
        "pt_class": 90,
        "ft_class": 30
    },
    "ads-b-20": {
        "name": "ads-b",
        "linux": "~/Datasets/ADS-B",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ADS-B"),
        "pt_class": 90,
        "ft_class": 20
    },
    "ads-b-10": {
        "name": "ads-b",
        "linux": "~/Datasets/ADS-B",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ADS-B"),
        "pt_class": 90,
        "ft_class": 10
    },
    "lora": {
        "name": "lora",
        "linux": "~/Datasets/LoRa",
        "windows": "E:\\Datasets\\LoRa",
        "pt_class": 30,
        "ft_class": 10
    },
    "wifi": {
        "name": "wifi",
        "linux": "~/Datasets/WiFi_ft62",
        "windows": "E:\\Datasets\\WiFi_ft62",
        "pt_class": 10,
        "ft_class": 6
    },
    "manytx": {
        "name": "manytx",
        "linux": "~/Datasets/ManyTx",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ManyTx"),
        "pt_class": 90,
        "ft_class": 30
    },
    "single": {
        "name": "single",
        "linux": "~/Datasets/Single",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "Single"),
        "pt_class": 28,
        "ft_class": 20
    }
}
for ft_class in [30, 20, 10]:
    dataset_path_dict[f"ads-b{ft_class}"] = deepcopy(dataset_path_dict["ads-b"])
    dataset_path_dict[f"ads-b{ft_class}"]["ft_class"] = ft_class
    dataset_path_dict[f"ads-b-{ft_class}"] = deepcopy(dataset_path_dict["ads-b"])
    dataset_path_dict[f"ads-b-{ft_class}"]["ft_class"] = ft_class
for ft_class in [30, 20, 10]:
    dataset_path_dict[f"manytx{ft_class}"] = deepcopy(dataset_path_dict["manytx"])
    dataset_path_dict[f"manytx{ft_class}"]["ft_class"] = ft_class
    dataset_path_dict[f"manytx-{ft_class}"] = deepcopy(dataset_path_dict["manytx"])
    dataset_path_dict[f"manytx-{ft_class}"]["ft_class"] = ft_class
for ft_class in [20, 10]:
    dataset_path_dict[f"single{ft_class}"] = deepcopy(dataset_path_dict["single"])
    dataset_path_dict[f"single{ft_class}"]["ft_class"] = ft_class
    dataset_path_dict[f"single-{ft_class}"] = deepcopy(dataset_path_dict["single"])
    dataset_path_dict[f"single-{ft_class}"]["ft_class"] = ft_class

model_path_dict = {
    "ResNet18": os.path.join(PROJECT_ROOT, "models", "ResNet18Feature.py"),
    "CVTSLANet": os.path.join(PROJECT_ROOT, "models", "CVTSLANetFeature.py"),
    "CVTSLANet-Shallow": os.path.join(PROJECT_ROOT, "models", "SCVTSLANet.py"),
    "CVTSLANet-Deep": os.path.join(PROJECT_ROOT, "models", "DCVTSLANet.py"),
    "CVCM": os.path.join(PROJECT_ROOT, "models", "CVCMFeature.py"),
    "STC": os.path.join(PROJECT_ROOT, "models", "STCFeature.py"),
    "LinearClassifier": os.path.join(PROJECT_ROOT, "models", "LinearClassifier.py"),
    "NCE": os.path.join(PROJECT_ROOT, "models", "NCELoss.py"),
    "MML": os.path.join(PROJECT_ROOT, "models", "ManifoldMixupLoss.py"),
    "MTL": os.path.join(PROJECT_ROOT, "models", "AutomaticWeightedLoss.py"),
    "Rotator": os.path.join(PROJECT_ROOT, "models", "Rotator.py")
}


def _validate_choice(value, choices, name):
    if value not in choices:
        raise ValueError(f"Unsupported {name} '{value}'. Available values: {', '.join(sorted(choices))}")


# The input param 'feature_dim' must be a valid integer and a positive even number.
def feature_dim_type(value):
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

    if int_value <= 0 or int_value % 2 != 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive even number")

    return int_value


# The input param 'epoch_threshold' must be a valid float or integer.
def epoch_threshold_type(value):
    try:
        f_value = float(value)
        if f_value.is_integer():
            return int(f_value)
        return f_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold value: {value}")


def is_joint_interference_method(method_name):
    if method_name is None:
        return False
    method_name = str(method_name)
    return method_name == "NEW3" or method_name.lower().startswith("robustsei")


def TSLA_add_args(parser, seq_len=4800, patch_size=32, num_channels=2, emb_dim=256, depth=3, dropout_rate=0.3):
    parser.add_argument("--TSLA_len", type=int, default=seq_len)
    parser.add_argument("--TSLA_patch", type=int, default=patch_size)
    parser.add_argument("--TSLA_channels", type=int, default=num_channels)
    parser.add_argument("--TSLA_emb", type=feature_dim_type, default=emb_dim)
    parser.add_argument("--TSLA_depth", type=int, default=depth)
    parser.add_argument("--TSLA_dropout", type=float, default=dropout_rate)
    return parser


def TSLA_parse_args(opt):
    return {
        "seq_len": opt.TSLA_len,
        "patch_size": opt.TSLA_patch,
        "num_channels": opt.TSLA_channels,
        "emb_dim": opt.TSLA_emb,
        "depth": opt.TSLA_depth,
        "dropout_rate": opt.TSLA_dropout
    }


def pretrain_config(encoder_name="ResNet18", classifiar_name="Linear", dataset_name="ads-b", input_type="iq", rot_num=4, input_class=-1,
                    normalize_fn="power", batch_size=32, max_epoch=300, epoch_threshold=0.5, lr=0.001, lr_step=50, lr_gamma=0.1, momentum=0.9,
                    weight_decay=5e-4, feature_dim=2048, tsla_conf=None, mml_b=2.0, save_freq=50, RANDOM_SEED=2024, ablate="", extra_info="",
                    resume="", use_lfdb=True, lfdb_weight=1.0, awgn_enable=True, awgn_min=0.0, awgn_max=30.0,
                    con_weight=1.0, adv_weight=0.2, ch_weight=1.0, mask_weight=0.05, mask_ratio=0.5,
                    method_name="NEW3", inv_weight=0.2, int_weight=0.5, warmup_epochs=5,
                    snr_levels=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", type=str, default=encoder_name)
    parser.add_argument("--classifiar", "-c", type=str, default=classifiar_name)
    parser.add_argument("--dataset", "-d", type=str, default=dataset_name)
    parser.add_argument("--input_type", "-t", type=str, default=input_type)
    parser.add_argument("--rot_num", type=int, default=rot_num)
    parser.add_argument("--input_class", type=int, default=input_class)
    parser.add_argument("--normalize_fn", "-n", type=str, default=normalize_fn)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epoch", type=int, default=max_epoch)
    parser.add_argument("--threshold", type=epoch_threshold_type, default=epoch_threshold)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--lr_step", type=int, default=lr_step)
    parser.add_argument("--lr_gamma", type=float, default=lr_gamma)
    parser.add_argument("--momentum", type=float, default=momentum)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--feature_dim", type=feature_dim_type, default=feature_dim)
    parser.add_argument("--mml_b", type=float, default=mml_b)
    parser.add_argument("--save_freq", type=int, default=save_freq)
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ablate", "-a", type=str, nargs='+', default=ablate)
    parser.add_argument("--extra_info", type=str, default=extra_info)
    parser.add_argument("--resume", "-r", type=str, default=resume)
    parser.add_argument("--method_name", type=str, default=method_name)
    parser.add_argument("--use_lfdb", action="store_true", default=use_lfdb)
    parser.add_argument("--no_lfdb", action="store_false", dest="use_lfdb")
    parser.add_argument("--lfdb_weight", type=float, default=lfdb_weight)
    parser.add_argument("--awgn_enable", action="store_true", default=awgn_enable)
    parser.add_argument("--no_awgn", action="store_false", dest="awgn_enable")
    parser.add_argument("--awgn_min", type=float, default=awgn_min)
    parser.add_argument("--awgn_max", type=float, default=awgn_max)
    parser.add_argument("--con_weight", type=float, default=con_weight)
    parser.add_argument("--inv_weight", type=float, default=inv_weight)
    parser.add_argument("--adv_weight", type=float, default=adv_weight)
    parser.add_argument("--ch_weight", type=float, default=ch_weight)
    parser.add_argument("--int_weight", type=float, default=int_weight)
    parser.add_argument("--mask_weight", type=float, default=mask_weight)
    parser.add_argument("--mask_ratio", type=float, default=mask_ratio)
    parser.add_argument("--warmup_epochs", type=int, default=warmup_epochs)
    parser.add_argument("--snr_levels", type=float, nargs="+", default=snr_levels or [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    explicit_tsla_conf = tsla_conf is not None
    parser = TSLA_add_args(parser, **({} if tsla_conf is None else tsla_conf))
    opt = parser.parse_args()
    if opt.classifiar.lower() == "linear":
        opt.classifiar = "Linear"

    encoder_choices = {k for k in model_path_dict if not k.endswith("Classifier") and k not in {"NCE", "MML", "MTL", "Rotator"}}
    _validate_choice(opt.encoder, encoder_choices, "encoder")
    _validate_choice(opt.classifiar, {"Linear"}, "classifier")
    _validate_choice(opt.dataset, dataset_path_dict, "dataset")
    if opt.rot_num < 2:
        raise ValueError("rot_num must be at least 2")
    if opt.awgn_min > opt.awgn_max:
        raise ValueError("awgn_min cannot be greater than awgn_max")

    method_name = opt.method_name.strip()
    if method_name.upper() == "NEW3":
        method_name = "NEW3"
    if method_name.lower() in {"robustsei", "robust_sei"}:
        method_name = "RobustSEI"
    if is_joint_interference_method(method_name) and opt.TSLA_emb == 256 and not explicit_tsla_conf:
        opt.TSLA_emb = 128
    tsla_conf = TSLA_parse_args(opt)
    loss_item = ["id", "inv", "adv", "int", "mask"] if is_joint_interference_method(method_name) and opt.use_lfdb else ["rot_cls", "sei_cls", "mml"]
    if opt.use_lfdb and not is_joint_interference_method(method_name):
        loss_item.extend(["con", "adv", "ch", "mask"])
    ablate_item = ([opt.ablate] if not isinstance(opt.ablate, list) else opt.ablate) if opt.ablate else []
    exp_suffix = ""
    for item in ablate_item:
        if item in loss_item:
            loss_item.remove(item)
            exp_suffix += f"_{item}Ablate"
    method_suffix = f"_{method_name}" if method_name else ""
    exp = f"{opt.encoder}_{dataset_path_dict[opt.dataset]['name']}_{opt.input_type}_{opt.normalize_fn}Norm{method_suffix}{exp_suffix}"
    platform = "windows" if sys.platform.startswith("win") else "linux"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_item = tuple(loss_item)
    num_classes = dataset_path_dict[opt.dataset]["pt_class"] if opt.input_class == -1 else opt.input_class
    dataset_norm = opt.normalize_fn
    dataset_root = dataset_path_dict[opt.dataset][platform]
    dataset_root = dataset_root if opt.input_type == "iq" else os.path.join(dataset_root, opt.input_type)
    dataset_root = dataset_root if os.path.exists(os.path.expanduser(dataset_root)) else dataset_root.replace("~", "~/xulai")

    return {
        "exp_name": exp,
        "exp_type": f"Pretext_{method_name}" if method_name else "Pretext",
        "method_name": method_name,
        "random_seed": opt.random_seed,
        "platform": platform,
        "device": device,
        "start_epoch": 0,
        "epoch": opt.epoch,
        "threshold": min(int(opt.threshold * opt.epoch), opt.epoch) if isinstance(opt.threshold, float) else opt.threshold,
        "save_freq": opt.save_freq,
        "resume": opt.resume,
        "dataset": {
            "name": opt.dataset,
            "root": dataset_root,
            "type": opt.input_type,
            "normalize": dataset_norm,
            "batch_size": opt.batch_size,
            "ratio": 0.2,
            "num_classes": num_classes
        },
        "augmentation": {
            "awgn_enable": opt.awgn_enable,
            "awgn_snr_range": (opt.awgn_min, opt.awgn_max),
            "snr_levels": tuple(opt.snr_levels),
        },
        "encoder": {
            "name": opt.encoder,
            "root": model_path_dict[opt.encoder],
            "feature_dim": opt.feature_dim,
            "TSLA_config": tsla_conf
        },
        "rot_classifier": {
            "root": model_path_dict[f"{opt.classifiar}Classifier"],
            "in_dim": opt.feature_dim,
            "num_classes": opt.rot_num
        },
        "mixed_classifier": {
            "root": model_path_dict[f"{opt.classifiar}Classifier"],
            "in_dim": opt.feature_dim,
            "num_classes": num_classes
        },
        "optimizer": {
            "lr": opt.lr,
            "momentum": opt.momentum,
            "weight_decay": opt.weight_decay,
            "step_size": opt.lr_step,
            "gamma": opt.lr_gamma
        },
        "mml": {
            "root": model_path_dict["MML"],
            "beta": opt.mml_b
        },
        "mtl": {
            "root": model_path_dict["MTL"],
            "item": loss_item,
            "num": len(loss_item)
        },
        "lfdb": {
            "enabled": opt.use_lfdb,
            "weight": opt.lfdb_weight,
            "num_classes": num_classes,
            "snr_classes": len(opt.snr_levels) if is_joint_interference_method(method_name) else 3,
            "fading_classes": 3,
            "con_weight": opt.con_weight,
            "inv_weight": opt.inv_weight,
            "adv_weight": opt.adv_weight,
            "ch_weight": opt.ch_weight,
            "int_weight": opt.int_weight,
            "mask_weight": opt.mask_weight,
            "mask_ratio": opt.mask_ratio,
            "warmup_epochs": opt.warmup_epochs,
        },
        "extra_info": opt.extra_info
    }


def finetune_config(encoder_name="ResNet18", classifier_name="Linear", dataset_name="ads-b", input_type="iq", input_class=-1, normalize_fn="power",
                    train_batch_size=30, test_batch_size=30, shot=5, max_epoch=50, max_iteration=100, lr=0.001, weight_decay=0, feature_dim=2048,
                    tsla_conf=None, RANDOM_SEED=2024, pretrain_normalize_fn="same", pretrain_batch_size=32, pretrain_epoch=250, ablate="",
                    extra_info="", pretrain_date="", snr_enable=False, snr=0, use_dcfa=False, dcfa_aug_per_class=100,
                    dcfa_top_m=3, dcfa_alpha=1.0, dcfa_beta=0.2, dcfa_epsilon=1e-3, aux_dataset="ads-b",
                    use_lfdb_features=False, method_name="NEW3"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", type=str, default=encoder_name)
    parser.add_argument("--classifier", "-c", type=str, default=classifier_name)
    parser.add_argument("--dataset", "-d", type=str, default=dataset_name)
    parser.add_argument("--input_type", "-t", type=str, default=input_type)
    parser.add_argument("--input_class", type=int, default=input_class)
    parser.add_argument("--normalize_fn", "-n", type=str, default=normalize_fn)
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--test_batch_size", type=int, default=test_batch_size)
    parser.add_argument("--shot", "-s", type=int, nargs='+', default=shot)
    parser.add_argument("--epoch", type=int, default=max_epoch)
    parser.add_argument("--iteration", "-i", type=int, default=max_iteration)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--feature_dim", type=feature_dim_type, default=feature_dim)
    parser.add_argument("--random_seed", "-r", type=int, default=RANDOM_SEED)
    parser.add_argument("--pretrain_normalize_fn", type=str, default=normalize_fn if pretrain_normalize_fn == "same" else pretrain_normalize_fn)
    parser.add_argument("--pretrain_batch_size", type=int, default=pretrain_batch_size)
    parser.add_argument("--pretrain_epoch", type=int, default=pretrain_epoch)
    parser.add_argument("--pretrain_date", type=str, default=pretrain_date)
    parser.add_argument("--method_name", type=str, default=method_name)
    parser.add_argument("--ablate", "-a", type=str, nargs='+', default=ablate)
    parser.add_argument("--extra_info", type=str, default=extra_info)
    parser.add_argument("--snr_enable", action="store_true", default=snr_enable)
    parser.add_argument("--snr", type=int, nargs='+', default=snr)
    parser.add_argument("--use_dcfa", action="store_true", default=use_dcfa)
    parser.add_argument("--dcfa_aug_per_class", type=int, default=dcfa_aug_per_class)
    parser.add_argument("--dcfa_top_m", type=int, default=dcfa_top_m)
    parser.add_argument("--dcfa_alpha", type=float, default=dcfa_alpha)
    parser.add_argument("--dcfa_beta", type=float, default=dcfa_beta)
    parser.add_argument("--dcfa_epsilon", type=float, default=dcfa_epsilon)
    parser.add_argument("--aux_dataset", type=str, default=aux_dataset)
    parser.add_argument("--use_lfdb_features", action="store_true", default=use_lfdb_features)
    parser = TSLA_add_args(parser, **({} if tsla_conf is None else tsla_conf))
    opt = parser.parse_args()
    method_name = opt.method_name.strip()
    if method_name.upper() == "NEW3":
        method_name = "NEW3"
    if method_name.lower() in {"robustsei", "robust_sei"}:
        method_name = "RobustSEI"
    classifier_aliases = {
        "dl": "Linear",
        "linear": "Linear",
        "lr": "lr",
        "knn": "knn",
        "svm": "svm",
    }
    opt.classifier = classifier_aliases.get(opt.classifier.lower(), opt.classifier)
    encoder_choices = {k for k in model_path_dict if not k.endswith("Classifier") and k not in {"NCE", "MML", "MTL", "Rotator"}}
    _validate_choice(opt.encoder, encoder_choices, "encoder")
    _validate_choice(opt.classifier.lower(), {"linear", "lr", "knn", "svm"}, "classifier")
    _validate_choice(opt.dataset, dataset_path_dict, "dataset")
    _validate_choice(opt.aux_dataset, dataset_path_dict, "auxiliary dataset")

    tsla_conf = TSLA_parse_args(opt)

    loss_item = ["rot_cls", "sei_cls", "mml"]
    ablate_item = ([opt.ablate] if not isinstance(opt.ablate, list) else opt.ablate) if opt.ablate else []
    exp_suffix = ""
    for item in ablate_item:
        if item in loss_item:
            exp_suffix += f"_{item}Ablate"
    dataset_fullname = opt.dataset + "_" + opt.input_type + ("_snr" if opt.snr_enable else "")
    method_suffix = f"_{method_name}" if method_name else ""
    exp = f"{opt.encoder}_{dataset_fullname}_PT_{opt.pretrain_normalize_fn}Norm_FT_{opt.normalize_fn}Norm{method_suffix}{exp_suffix}"
    pretrain_exp = f"{opt.encoder}_{dataset_path_dict[opt.dataset]['name']}_{opt.input_type}_{opt.pretrain_normalize_fn}Norm{method_suffix}{exp_suffix}"
    pretrain_exp_type = f"Pretext_{method_name}_random_rot" if method_name else "Pretext_random_rot"
    platform = "windows" if sys.platform.startswith("win") else "linux"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = dataset_path_dict[opt.dataset]["ft_class"] if opt.input_class == -1 else opt.input_class
    dataset_norm = opt.normalize_fn
    dataset_root = dataset_path_dict[opt.dataset][platform]
    dataset_root = dataset_root if opt.input_type == "iq" else os.path.join(dataset_root, opt.input_type)
    dataset_root = dataset_root if os.path.exists(os.path.expanduser(dataset_root)) else dataset_root.replace("~", "~/xulai")

    return {
        "exp_name": exp,
        "exp_type": (f"Downstream_{method_name}" if method_name else "Downstream") if opt.pretrain_epoch else 'FineZero',
        "method_name": method_name,
        "random_seed": opt.random_seed,
        "platform": platform,
        "device": device,
        "iteration": opt.iteration,
        "epoch": opt.epoch,
        "dataset": {
            "name": opt.dataset,
            "root": dataset_root,
            "type": opt.input_type,
            "normalize": dataset_norm,
            "train_batch_size": opt.train_batch_size,
            "test_batch_size": opt.test_batch_size,
            "ratio": 0.2,
            "num_classes": num_classes,
            "shot": opt.shot if isinstance(opt.shot, list) else [opt.shot],
            "snr": (opt.snr if isinstance(opt.snr, list) else [opt.snr]) if opt.snr_enable else [None]
        },
        "encoder": {
            "name": opt.encoder,
            "root": model_path_dict[opt.encoder],
            "feature_dim": opt.feature_dim,
            "pretrain_path": os.path.join(
                PROJECT_ROOT,
                "runs",
                pretrain_exp_type,
                pretrain_exp,
                *([opt.pretrain_date] if opt.pretrain_date else []),
                "best_encoder.pth"
            ) if opt.pretrain_epoch else "",
            "lfdb_path": os.path.join(
                PROJECT_ROOT,
                "runs",
                pretrain_exp_type,
                pretrain_exp,
                *([opt.pretrain_date] if opt.pretrain_date else []),
                "best_lfdb.pth"
            ) if opt.pretrain_epoch else "",
            "TSLA_config": tsla_conf
        },
        "lfdb": {
            "enabled": opt.use_lfdb_features,
            "num_classes": dataset_path_dict[opt.dataset]["pt_class"],
            "snr_classes": 7 if is_joint_interference_method(method_name) else 3,
            "fading_classes": 3,
        },
        "classifier": {
            "name": opt.classifier.lower(),
            "root": model_path_dict[f"{opt.classifier}Classifier"] if opt.classifier.lower() not in ["lr", "knn", "svm"] else "",
            "in_dim": opt.feature_dim,
            "ratio": 0.2,
            "num_classes": num_classes
        },
        "optimizer": {
            "lr": opt.lr,
            "weight_decay": opt.weight_decay
        },
        "dcfa": {
            "enabled": opt.use_dcfa,
            "aux_root": dataset_path_dict[opt.aux_dataset][platform],
            "aux_num_classes": dataset_path_dict[opt.aux_dataset]["pt_class"],
            "aug_per_class": opt.dcfa_aug_per_class,
            "top_m": opt.dcfa_top_m,
            "alpha": opt.dcfa_alpha,
            "beta": opt.dcfa_beta,
            "epsilon": opt.dcfa_epsilon,
        },
        "extra_info": opt.extra_info
    }


if __name__ == "__main__":
    import json

    pretrain_conf = pretrain_config()
    finetune_conf = finetune_config()
    print("==> Pretrain Config:", json.dumps(pretrain_conf))
    print("==> Finetune Config:", json.dumps(finetune_conf))
