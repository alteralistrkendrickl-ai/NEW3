import unittest

import torch

from models.MSFTFNetFixedFeature import MSFTFNetFixed
from models.MSFTFNetQCRouterFeature import MSFTFNetQCRouter
from utils.config import (
    use_pairview_fixed_mix_no_restore,
    use_quality_rank,
    use_quality_router,
)


class QCRouterModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.randn(3, 2, 64)

    def test_router_outputs_normalized_weights_and_diagnostics(self):
        model = MSFTFNetQCRouter(
            seq_len=64,
            patch_size=8,
            emb_dim=16,
            depth=1,
            num_classes=32,
            dropout_rate=0.0,
        )
        stages = model.forward_stages(self.inputs)
        self.assertEqual(stages["feature_map"].shape[0], 3)
        self.assertEqual(stages["quality"].shape, (3,))
        self.assertEqual(stages["route_weights"].shape, (3, 5))
        self.assertEqual(stages["route_entropy"].shape, (3,))
        torch.testing.assert_close(
            stages["route_weights"].sum(dim=1),
            torch.ones(3),
        )
        self.assertTrue(torch.all(stages["quality"] >= 0.0))
        self.assertTrue(torch.all(stages["quality"] <= 1.0))

    def test_router_receives_gradients(self):
        model = MSFTFNetQCRouter(
            seq_len=64,
            patch_size=8,
            emb_dim=16,
            depth=1,
            num_classes=32,
            dropout_rate=0.0,
        )
        stages = model.forward_stages(self.inputs)
        loss = stages["feature_map"].square().mean() + stages["quality"].mean()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.qc_router.parameters()
        ))

    def test_fixed_fusion_preserves_feature_map_contract(self):
        model = MSFTFNetFixed(
            seq_len=64,
            patch_size=8,
            emb_dim=16,
            depth=1,
            num_classes=32,
            dropout_rate=0.0,
        )
        stages = model.forward_stages(self.inputs)
        self.assertEqual(stages["feature_map"].shape[0], 3)
        self.assertEqual(stages["feature_map"].shape[1], 16)


class QCRouterConfigTest(unittest.TestCase):
    def test_method_flags_keep_fixed_pair_protocol(self):
        methods = (
            "RobustSEI_CleanAnchor_QCFixedAvg",
            "RobustSEI_CleanAnchor_QCCurrentGate",
            "RobustSEI_CleanAnchor_QCRouterNoRank",
            "RobustSEI_CleanAnchor_QCRouter",
        )
        for method in methods:
            self.assertTrue(use_pairview_fixed_mix_no_restore(method))

    def test_only_full_router_enables_quality_rank(self):
        self.assertTrue(use_quality_router(
            "RobustSEI_CleanAnchor_QCRouterNoRank"
        ))
        self.assertFalse(use_quality_rank(
            "RobustSEI_CleanAnchor_QCRouterNoRank"
        ))
        self.assertTrue(use_quality_rank(
            "RobustSEI_CleanAnchor_QCRouter"
        ))


if __name__ == "__main__":
    unittest.main()
