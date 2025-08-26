from qm8_dataset import QM8GraphDataset

dataset = QM8GraphDataset(
    root="dataset/QM8_E1-CAM",
    target_column="E1-CAM",
    use_global_features=True,
    use_3d_geo=True
)
dataset.print_dataset_info()
