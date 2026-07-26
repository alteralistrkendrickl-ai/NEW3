from utils.config import pretrain_config
from pretext import pretext


if __name__ == "__main__":
    pretext(pretrain_config(
        encoder_name="CVTSLANet",
        classifiar_name="Linear",
        dataset_name="manytx",
        input_type="iq",
        rot_num=8,
        feature_dim=1024,
        max_epoch=120,
        batch_size=128,
        weight_decay=1e-4,
        save_freq=1,
        method_name="RobustSEI",
        con_weight=0.2,
        adv_weight=0.1,
        mask_weight=0.05,
        warmup_epochs=25,
        stage2_epochs=25,
        mask_min=0.10,
        mask_max=0.40,
        mask_tv_weight=0.1,
        tsla_conf={
            "seq_len": 256,
            "patch_size": 16,
            "num_channels": 2,
            "emb_dim": 128,
            "depth": 3,
            "dropout_rate": 0.3,
        },
    ))
