"""Demonstration script for ZENOVA Generic Dataset Pipeline using a synthetic wellbeing benchmark."""
import sys
from pathlib import Path

# Dynamic path resolution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random
from zenova.data.pipeline import GenericDatasetPipeline
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.demo_dataset")


def generate_synthetic_wellbeing_data(num_dialogues: int = 50, turns_per_dialogue: int = 4):
    """Generates controlled synthetic wellbeing dialogue turns across Hill's helping skills."""
    strategies = [
        "Question",
        "Restatement or Paraphrasing",
        "Reflection of feelings",
        "Affirmation and Reassurance",
        "Self-disclosure",
        "Providing Suggestions",
        "Information",
        "Others"
    ]
    emotions = ["anxiety", "sadness", "frustration", "exhaustion", "neutral"]

    sample_dialogue_topics = [
        "academic stress with upcoming final exams",
        "feeling isolated after moving to a new city",
        "work burnout and chronic sleep difficulties",
        "relationship tension with a close family member",
        "imposter syndrome during a technical internship"
    ]

    utterance_templates = {
        "Question": [
            "How long have you been experiencing this pressure?",
            "What part of this situation feels most overwhelming right now?",
            "Could you share a bit more about how that affected your sleep?"
        ],
        "Reflection of feelings": [
            "It sounds like you have been carrying a deep sense of heaviness.",
            "I hear how anxious and exhausted this whole week has left you feeling.",
            "That sounds really painful and discouraging to navigate on your own."
        ],
        "Affirmation and Reassurance": [
            "You have demonstrated tremendous strength by simply talking about this.",
            "It is completely understandable to feel shaken, and you are doing your best.",
            "Acknowledging these feelings takes genuine bravery."
        ],
        "Restatement or Paraphrasing": [
            "So what happened was that multiple deadlines collided all at once.",
            "In other words, you felt unheard during the team meeting.",
            "From what you are saying, the workload simply exceeded your bandwidth."
        ],
        "Providing Suggestions": [
            "Would you be open to trying a five-minute breathing break before continuing?",
            "Perhaps jotting down just one achievable task for tomorrow could help.",
            "When you feel up to it, a short walk away from screens might bring relief."
        ],
        "Information": [
            "Burnout often manifests as physical exhaustion and sudden lack of motivation.",
            "Sleep disruptions and emotional irritability frequently reinforce one another.",
            "Gradual pacing is a well-established method for managing cognitive overload."
        ],
        "Self-disclosure": [
            "I have spoken with many students who encountered that exact same challenge.",
            "Navigating new environments often brings up very similar self-doubts.",
            "Many people find the transition to a demanding role deeply disorienting."
        ],
        "Others": [
            "I am right here with you. Take all the time you need.",
            "Thank you for trusting me with these thoughts.",
            "We can take things one gentle step at a time."
        ]
    }

    random.seed(42)
    samples = []

    for d_id in range(1, num_dialogues + 1):
        topic = random.choice(sample_dialogue_topics)
        emotion = random.choice(emotions)

        for t_id in range(1, turns_per_dialogue + 1):
            strategy = random.choice(strategies)
            supporter_text = f"{random.choice(utterance_templates[strategy])} [Dialogue #{d_id:03d} on {topic}, phase {t_id}]"

            sample = {
                "dialog_id": f"synth_dlg_{d_id:03d}",
                "turn_id": t_id,
                "situation": topic,
                "emotion": emotion,
                "speaker": "supporter",
                "text": supporter_text,
                "label": strategy
            }
            samples.append(sample)

    return samples


def main():
    dataset_name = "Synthetic Wellbeing Benchmark"
    print(f"Generating synthetic dataset samples for {dataset_name}...")
    samples = generate_synthetic_wellbeing_data(num_dialogues=60, turns_per_dialogue=4)
    print(f"Generated {len(samples)} turns across 60 dialogues.")

    pipeline = GenericDatasetPipeline(dataset_name=dataset_name)

    metadata = pipeline.run(
        raw_samples=samples,
        source="ZENOVA Synthetic Benchmark Generator",
        official_url="https://zenova.internal/datasets/synthetic_wellbeing",
        license_name="MIT",
        citation="Zenova Research Team. (2026). Synthetic Wellbeing Benchmark Dataset for Controlled Pipeline Validation.",
        intended_task="support_strategy_planning",
        features=["text", "situation", "emotion", "dialog_id", "turn_id"],
        labels=[
            "Question",
            "Restatement or Paraphrasing",
            "Reflection of feelings",
            "Affirmation and Reassurance",
            "Self-disclosure",
            "Providing Suggestions",
            "Information",
            "Others"
        ],
        label_field="label",
        text_field="text",
        group_field="dialog_id",
        version="1.0.0",
        seed=42
    )

    print("\n=== Pipeline Execution Summary ===")
    print(f"Dataset: {metadata.dataset_name} (v{metadata.version})")
    print(f"Total Processed Turns: {metadata.number_of_samples['processed']}")
    print(f"Train Count: {metadata.train_validation_test_split.train_count}")
    print(f"Val Count: {metadata.train_validation_test_split.val_count}")
    print(f"Test Count: {metadata.train_validation_test_split.test_count}")
    print(f"Leakage Check Passed: {metadata.leakage_check_passed}")
    print(f"Datasheet Location: {metadata.local_storage_location['docs']}")
    print(f"Metadata Registered: {metadata.local_storage_location['metadata']}")


if __name__ == "__main__":
    main()
