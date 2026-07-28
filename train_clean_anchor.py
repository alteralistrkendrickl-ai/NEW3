from pretext import pretext
from utils.config import pretrain_config


if __name__ == "__main__":
    pretext(pretrain_config(
        encoder_name="MSFTFNet",
        classifiar_name="Linear",
        dataset_name="manytx",
        input_type="iq",
        rot_num=8,
        feature_dim=1024,
        max_epoch=120,
        batch_size=128,
        weight_decay=1e-4,
        save_freq=1,
        method_name="RobustSEI_CleanAnchor",
        clean_id_weight=1.0,
        noisy_id_weight=0.5,
        clean_cons_weight=0.2,
        mask_weight=0.02,
        low_snr_start_epoch=20,
        very_low_snr_start_epoch=60,
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
