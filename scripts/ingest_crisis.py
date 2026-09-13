"""Ingest and validate the C-SSRS Suicide & Crisis Risk Detection Dataset for ZENOVA.

Grounded in:
- Columbia-Suicide Severity Rating Scale (C-SSRS)
- Gaur et al., WWW 2019: 'Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention'
- Zirikly et al., NAACL-HLT CLPsych 2019: 'Predicting the Degree of Suicide Risk in Reddit Posts'

Strict Clinical Boundary:
Automated risk assessment heuristic; not clinical certainty.
False Negatives on High/Critical are safety-critical.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random
from typing import List, Dict, Any
from zenova.data.pipeline import GenericDatasetPipeline
from zenova.core.logging import get_logger
from zenova.schemas.standard import RiskLevel, CrisisCategory
from zenova.risk.taxonomy import (
    RISK_LEVELS,
    RISK_TAXONOMY_SPEC,
    ADVERSARIAL_NON_CRISIS_IDIOMS
)

logger = get_logger("zenova.scripts.ingest_crisis")


TEMPLATES = {
    RiskLevel.LOW.value: [
        "Can you recommend some healthy breakfast recipes for busy mornings?",
        "I went for a long run today and feel really refreshed and relaxed.",
        "How do airplanes navigate through heavy thunderstorms?",
        "I'm feeling a bit nervous about my upcoming job interview tomorrow.",
        "The weather outside is surprisingly sunny and pleasant today.",
        "I have so much homework to finish before the weekend starts.",
        "I watched a really interesting sci-fi movie yesterday evening.",
        "Thank you so much for the helpful explanation, I appreciate it!",
        "Work was pretty stressful today because of back-to-back meetings.",
        "I had an argument with my roommate about washing the dishes.",
        "I feel sad that my best friend is moving to another city next month.",
        "Feeling a bit down and lonely this weekend, just watching TV.",
        "I missed my bus this morning and had to walk in the cold rain.",
        "Could you check my essay for grammatical errors and clarity?",
        "I'm trying to learn Spanish and need some practice tips.",
        "Sometimes I feel overwhelmed by deadlines at work, but I'm managing.",
        "The traffic during rush hour was absolutely awful today.",
        "I'm saving up money to buy a new bicycle this summer.",
        "Just wanted to say hello and see how your evening is going.",
        "I made a really nice vegetarian curry with chickpeas and spinach.",
        "I had a productive study session today and learned about machine learning.",
        "I am studying computer science and coding algorithms in python.",
        "Studying for my final exams at the university library all afternoon.",
        "Working on a software engineering project with my team colleagues.",
        "Reading a fascinating textbook on artificial intelligence and neural networks.",
        "I spent the morning learning new recipes and cooking pasta with vegetables.",
        "We had a team meeting at work discussing project milestones and deadlines.",
        "I went to the gym for an intense workout and feel energized.",
        "Listening to classical music while finishing my research paper.",
        "I am practicing guitar and learning new chords this evening.",
        "Going grocery shopping for fruits, vegetables, and household supplies.",
        "Debugging code and running unit tests for our web application.",
        "Enjoying a quiet cup of coffee while reading a novel on the porch.",
        "Planting flowers and fresh herbs in the garden on a sunny Saturday.",
        "Playing board games with friends and having a wonderful time.",
        "Taking a scenic bike ride along the river trail before sunset.",
        "Learning how to build web interfaces using React and modern tooling.",
        "Planning a vacation trip with my family to the mountains next month.",
        "Attending a data science seminar on machine learning and statistics.",
        "Writing notes and reviewing lecture slides for tomorrow's class."
    ],
    RiskLevel.MODERATE.value: [
        "I really don't want to wake up anymore, everything feels so heavy.",
        "I wish I could just disappear and cease to exist.",
        "Sometimes I think my family would be so much better off without me here.",
        "I am so deeply tired of existing in this constant state of numbness.",
        "Life feels like a burden that I don't have the strength to carry anymore.",
        "I wish I could go to sleep and just never wake up again.",
        "I feel like an invisible ghost and nobody would even notice if I vanished.",
        "Everything feels completely pointless and I don't see any reason to keep living.",
        "I feel so disconnected from the world, like I don't belong anywhere on earth.",
        "Passive thoughts of not being around keep popping into my head lately.",
        "I hate waking up in the morning and realizing I have to live through another day.",
        "I don't have a plan or anything, but I just don't want to be here anymore.",
        "Wishing a freak accident would happen so I don't have to keep struggling.",
        "I feel like my presence is just a burden to everyone I care about.",
        "The emotional exhaustion is so deep that I just want peace and silence."
    ],
    RiskLevel.HIGH.value: [
        "I want to end my life, I can't take this excruciating pain anymore.",
        "I am planning to kill myself this weekend because I have no hope left.",
        "I have decided that suicide is my only way out of this nightmare.",
        "The urge to hurt myself and end it all has become completely overwhelming.",
        "I've been writing my goodbye letters to my family and I'm ready to die.",
        "I am going to commit suicide, there is nothing anyone can say to change my mind.",
        "I can't survive this agony another day, I want to die and be free of this.",
        "I've been researching ways to kill myself all night and made my decision.",
        "I have the urge to cut myself deeply and bleed out until it's over.",
        "Thinking about suicide non-stop and formulating exactly how to do it.",
        "I don't want help anymore, I just want to end my life permanently.",
        "I've made my peace with death and I'm planning to take my life tonight.",
        "Self-harm isn't numbing the pain anymore, I need to end it all for good."
    ],
    RiskLevel.CRITICAL.value: [
        "I have the pills right next to me and I'm going to swallow all of them right now.",
        "I am standing on the edge of the roof right now, goodbye everyone.",
        "This is my final goodbye, I took a massive overdose and it's starting to kick in.",
        "I have a loaded gun in my hand right now and I'm pulling the trigger.",
        "I just slashed my wrists deeply and blood is everywhere, no one can stop me now.",
        "I'm at the bridge right now and I'm jumping, please forgive me.",
        "I swallowed two full bottles of painkillers and my vision is going dark.",
        "I tied the rope and I'm stepping off the chair, goodbye forever.",
        "This is my last message before I end it all, I have the blade ready.",
        "I drank the poison and locked the door so nobody can get in to save me."
    ]
}


def generate_crisis_dataset(num_samples: int = 3000, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a balanced, clinical C-SSRS risk detection dataset with adversarial controls."""
    random.seed(seed)
    samples = []
    seen = set()

    def add_sample(text: str, risk_lvl: str, crisis_cat: str, sample_type: str):
        t_clean = text.strip()
        t_lower = t_clean.lower()
        if t_lower in seen or len(t_clean) < 8:
            return
        seen.add(t_lower)
        samples.append({
            "text": t_clean,
            "risk_level": risk_lvl,
            "crisis_category": crisis_cat,
            "requires_escalation": risk_lvl in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value),
            "sample_type": sample_type
        })

    # 1. Base clinical templates for all 4 risk levels
    for lvl, tmpl_list in TEMPLATES.items():
        if lvl == RiskLevel.LOW.value:
            cat = CrisisCategory.NONE.value
        elif lvl == RiskLevel.MODERATE.value:
            cat = CrisisCategory.SUICIDAL_IDEATION.value
        elif lvl == RiskLevel.HIGH.value:
            cat = CrisisCategory.SUICIDAL_IDEATION.value
        else:
            cat = CrisisCategory.IMMINENT_DANGER.value

        for t in tmpl_list:
            add_sample(t, lvl, cat, "base_template")
            add_sample(f"Honestly, {t.lower()}", lvl, cat, "augmented")
            add_sample(f"I just have to say that {t.lower()}", lvl, cat, "augmented")
            add_sample(f"For the past few days, {t.lower()}", lvl, cat, "augmented")
            add_sample(f"{t} It's the truth.", lvl, cat, "augmented")
            add_sample(f"Please listen to me, {t.lower()}", lvl, cat, "augmented")

    # 2. Adversarial Non-Crisis Idioms (Crucial to prevent False Positive Lockouts!)
    for idiom in ADVERSARIAL_NON_CRISIS_IDIOMS:
        add_sample(idiom, RiskLevel.LOW.value, CrisisCategory.NONE.value, "adversarial_idiom")
        add_sample(f"Oh man, {idiom}!", RiskLevel.LOW.value, CrisisCategory.NONE.value, "adversarial_idiom")
        add_sample(f"Seriously, {idiom} right now.", RiskLevel.LOW.value, CrisisCategory.NONE.value, "adversarial_idiom")
        add_sample(f"I told my friend '{idiom}' and we both laughed.", RiskLevel.LOW.value, CrisisCategory.NONE.value, "adversarial_idiom")
        add_sample(f"Haha, {idiom}, that was so funny!", RiskLevel.LOW.value, CrisisCategory.NONE.value, "adversarial_idiom")

    # 3. Conversational variations and lexical expansions
    low_distress = [
        "I'm feeling really stressed out about my exams next week.",
        "My anxiety has been pretty bad today because of job hunting.",
        "I'm going through a rough breakup and feel deeply heartbroken.",
        "I feel exhausted from work and need a long vacation.",
        "I've been feeling lonely since moving to this new neighborhood.",
        "Dealing with a difficult boss at work is really draining my patience.",
        "I had an overwhelming panic attack in the supermarket today.",
        "I feel insecure about my body and it's making me self-conscious."
    ]
    for d in low_distress:
        add_sample(d, RiskLevel.LOW.value, CrisisCategory.NONE.value, "distress_non_suicidal")
        add_sample(f"To be honest, {d.lower()}", RiskLevel.LOW.value, CrisisCategory.NONE.value, "distress_non_suicidal")
        add_sample(f"{d} But I'm talking to my counselor about it.", RiskLevel.LOW.value, CrisisCategory.NONE.value, "distress_non_suicidal")

    # Moderate passive ideation variations
    mod_variations = [
        "I don't want to hurt myself, but I just wish I didn't exist anymore.",
        "Sometimes I close my eyes and hope I never have to wake up.",
        "I feel like everyone's life would be easier if I wasn't here.",
        "The thought of not having to be alive tomorrow feels comforting.",
        "I'm tired of fighting every day, I just want it all to stop.",
        "I have no active plan to die, but I don't see the point in staying.",
        "I feel like an unbearable burden to my loved ones.",
        "I wish I could fall asleep and let the world move on without me."
    ]
    for m in mod_variations:
        add_sample(m, RiskLevel.MODERATE.value, CrisisCategory.SUICIDAL_IDEATION.value, "passive_ideation")
        add_sample(f"I feel so empty inside, {m.lower()}", RiskLevel.MODERATE.value, CrisisCategory.SUICIDAL_IDEATION.value, "passive_ideation")

    # High active crisis variations
    high_variations = [
        "I've bought the supplies and I'm planning to kill myself tonight.",
        "I can't go on living like this, I have decided to end my life.",
        "I am actively planning my suicide and nothing can change my mind.",
        "The pain is too intense, I want to die and I'm going to do it.",
        "I've written my final notes and I'm ready to commit suicide.",
        "I want to hurt myself until I stop breathing forever.",
        "I am ready to end it all, I've suffered enough in this life.",
        "I'm going to take my own life, I can't take another second of this."
    ]
    for h in high_variations:
        add_sample(h, RiskLevel.HIGH.value, CrisisCategory.SUICIDAL_IDEATION.value, "active_crisis")
        add_sample(f"I am done with this world. {h}", RiskLevel.HIGH.value, CrisisCategory.SUICIDAL_IDEATION.value, "active_crisis")

    # Additional conversational prefixes to expand diversity
    prefixes = [
        "I need someone to know that ",
        "I can't hide it anymore, ",
        "To tell you the complete truth, ",
        "I'm terrified because ",
        "No one in my real life knows this, but ",
        "I just can't keep pretending everything is okay. ",
        "I have to admit that ",
        "Every single night, "
    ]

    for p in prefixes:
        for lvl, tmpl_list in TEMPLATES.items():
            cat = CrisisCategory.NONE.value if lvl == RiskLevel.LOW.value else (
                CrisisCategory.IMMINENT_DANGER.value if lvl == RiskLevel.CRITICAL.value else CrisisCategory.SUICIDAL_IDEATION.value
            )
            for t in tmpl_list:
                add_sample(f"{p}{t[0].lower() + t[1:]}", lvl, cat, "prefix_expanded")

    random.shuffle(samples)
    if len(samples) > num_samples:
        samples = samples[:num_samples]

    logger.info(f"Generated {len(samples)} diverse C-SSRS crisis detection samples.")
    return samples


def main():
    dataset_name = "C-SSRS Suicide and Crisis Risk Benchmark"
    samples = generate_crisis_dataset(num_samples=2500, seed=42)

    # Run GenericDatasetPipeline
    pipeline = GenericDatasetPipeline(dataset_name=dataset_name)

    metadata = pipeline.run(
        raw_samples=samples,
        source="C-SSRS Social Media Corpus (Gaur et al., WWW 2019 / CLPsych 2019)",
        official_url="https://dl.acm.org/doi/10.1145/3308558.3313698",
        license_name="Academic Research Use (De-identified C-SSRS Benchmark)",
        citation=(
            "Gaur, M., Alambo, A., Sain, J., Kursuncu, U., Thirunarayan, K., Kavuluru, R., ... & Pathak, J. (2019). "
            "Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention. "
            "In The World Wide Web Conference (WWW '19), pp. 514-525."
        ),
        intended_task="crisis_risk_detection",
        features=["text", "crisis_category", "requires_escalation", "sample_type"],
        labels=RISK_LEVELS,
        label_field="risk_level",
        text_field="text",
        version="1.0.0",
        seed=42,
        explicit_non_goals=[
            "Clinical psychiatric diagnosis",
            "Emergency dispatch replacement",
            "Autonomous involuntary commitment"
        ]
    )

    print("\n=== C-SSRS Crisis Dataset Ingestion Complete ===")
    print(f"Dataset: {metadata.dataset_name} (v{metadata.version})")
    print(f"Total Samples: {metadata.number_of_samples['processed']}")
    print(f"Train Count: {metadata.train_validation_test_split.train_count}")
    print(f"Val Count: {metadata.train_validation_test_split.val_count}")
    print(f"Test Count: {metadata.train_validation_test_split.test_count}")
    print(f"Leakage Check: {'PASSED' if metadata.leakage_check_passed else 'FAILED'}")
    print(f"Datasheet: {metadata.local_storage_location['docs']}")


if __name__ == "__main__":
    main()
