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
        adv_weight=0.001,
        inv_weight=0.001,
        int_weight=0.05,
        mask_weight=0.005,
        warmup_epochs=40,
        tsla_conf={
            "seq_len": 256,
            "patch_size": 16,
            "num_channels": 2,
            "emb_dim": 128,
            "depth": 3,
            "dropout_rate": 0.3,
        },
    ))
