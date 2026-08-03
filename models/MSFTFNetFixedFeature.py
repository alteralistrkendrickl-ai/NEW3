from models.MSFTFNetFeature import MSFTFNet


class MSFTFNetFixed(MSFTFNet):
    """MSFTFNet ablation with parameter-free time-frequency averaging."""

    def forward_stages(self, x):
        time_map = self.time_stem(x.float())
        freq_map = self.freq_stem(self._complex_spectrum(x))
        time_map, freq_map, reliability = self.tf_enhancer(time_map, freq_map)
        feature_map = 0.5 * (time_map + freq_map)
        tokens = feature_map.transpose(1, 2) + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return {
            "time_map": time_map,
            "freq_map": freq_map,
            "fused_map": feature_map,
            "feature_map": tokens.transpose(1, 2),
            "reliability": reliability,
        }


def create_model(feature_dim=1024, dtype="iq", **kwargs):
    if "iq" not in dtype:
        raise ValueError("MSFTFNet-Fixed is only used for IQ input.")
    return MSFTFNetFixed(
        seq_len=kwargs.get("seq_len", 256),
        patch_size=kwargs.get("patch_size", 16),
        num_channels=kwargs.get("num_channels", 2),
        emb_dim=kwargs.get("emb_dim", 128),
        depth=kwargs.get("depth", 3),
        num_classes=feature_dim,
        dropout_rate=kwargs.get("dropout_rate", 0.3),
    )
