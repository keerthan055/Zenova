"""Integration test for GenericDatasetPipeline."""
from zenova.data.pipeline import GenericDatasetPipeline


def test_generic_pipeline_execution(tmp_path):
    samples = [
        {"dialog_id": f"d_{i // 2}", "turn": i % 2, "text": f"Clean supportive turn #{i} with context", "label": f"strat_{i % 3}"}
        for i in range(30)
    ]

    pipeline = GenericDatasetPipeline(dataset_name="Test Pipeline Run", base_data_dir=str(tmp_path / "data"))
    metadata = pipeline.run(
        raw_samples=samples,
        source="Integration Test",
        official_url="https://zenova.test",
        license_name="MIT",
        citation="Test Citation (2026)",
        intended_task="test_task",
        features=["text", "dialog_id"],
        labels=["strat_0", "strat_1", "strat_2"],
        group_field="dialog_id",
        seed=42
    )

    assert metadata.dataset_name == "Test Pipeline Run"
    assert metadata.number_of_samples["processed"] == 30
    assert metadata.leakage_check_passed is True
    assert metadata.train_validation_test_split.train_count > 0
    assert "raw" in metadata.checksums
    assert "train" in metadata.checksums
