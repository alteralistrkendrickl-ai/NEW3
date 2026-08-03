from train_qc_common import run_qc_experiment


if __name__ == "__main__":
    run_qc_experiment(
        "MSFTFNet-QCRouter",
        "RobustSEI_CleanAnchor_QCRouter",
        use_quality_rank=True,
    )
