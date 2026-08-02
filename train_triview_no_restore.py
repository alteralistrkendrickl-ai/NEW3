import os

from pretext import pretext
from utils.config import PROJECT_ROOT, pretrain_config


V3_RUN_ROOT = os.path.join(
    PROJECT_ROOT,
    "runs",
    "Pretext_RobustSEI_CleanAnchorV3_random_rot",
    "MSFTFNet_manytx_iq_powerNorm_RobustSEI_CleanAnchorV3",
)


if __name__ == "__main__":
    pretext(pretrain_config(
        encoder_name="MSFTFNet",
        classifiar_name="Linear",
        dataset_name="manytx",
        input_type="iq",
        rot_num=8,
        feature_dim=1024,
        max_epoch=60,
        batch_size=128,
        lr=1e-4,
        weight_decay=1e-4,
        lr_step=40,
        save_freq=1,
        method_name="RobustSEI_CleanAnchorV4_TriViewNoRestore",
        clean_id_weight=1.0,
        noisy_id_weight=1.0,
        mask_weight=0.0,
        low_snr_start_epoch=10,
        very_low_snr_start_epoch=30,
        teacher_run_root=V3_RUN_ROOT,
        teacher_checkpoint="best",
        restoration_only=True,
        selective_encoder_finetune=True,
        encoder_adapt_lr=1e-5,
        grad_clip=5.0,
        tsla_conf={
            "seq_len": 256,
            "patch_size": 16,
            "num_channels": 2,
            "emb_dim": 128,
            "depth": 3,
            "dropout_rate": 0.3,
        },
    ))
