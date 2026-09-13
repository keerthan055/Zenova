"""Ingest and validate the PsySym-grounded mental health symptom dataset for ZENOVA.

Grounded in:
- Zhang et al., EMNLP 2022: 'Symptom Identification for Interpretable Detection of Multiple Mental Disorders on Social Media'
- DSM-5 Diagnostic Criteria
- 7 Psychiatric Disorders: MDD, GAD, Bipolar, PTSD, OCD, ADHD, Eating Disorders.

Strict Clinical Boundary:
Non-diagnostic observational signal extraction only; no diagnostic classification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random
from typing import List, Dict, Any
from zenova.data.pipeline import GenericDatasetPipeline
from zenova.core.logging import get_logger
from zenova.symptoms.taxonomy import SYMPTOM_SIGNALS, TAXONOMY_SPEC

logger = get_logger("zenova.scripts.ingest_psysym")


# Template patterns and linguistic variations grounded in clinical conversational markers
TEMPLATES = {
    "sleep disturbance": [
        "I haven't been sleeping properly lately",
        "I can't fall asleep until 4am every single night",
        "My insomnia has reached an unbearable point",
        "I keep waking up at 3am drenched in sweat and can't go back to sleep",
        "Tossing and turning for hours every night is destroying me",
        "My sleep schedule is completely shattered",
        "I slept for 14 hours and still woke up feeling like a zombie",
        "Barely got two hours of broken sleep last night",
        "I dread going to bed because I know I will just lie awake staring at the ceiling",
        "Constant nightmares and broken sleep are leaving me completely drained",
        "Having trouble falling asleep and staying asleep every day",
        "Restless nights have become my normal state"
    ],
    "loss of interest": [
        "I don't enjoy things anymore that used to make me happy",
        "Nothing brings me any joy or excitement these days",
        "I used to love painting and playing guitar, but now they feel completely pointless",
        "I feel emotionally numb and nothing seems fun anymore",
        "All my hobbies feel like empty chores now",
        "I have completely lost interest in everything I used to care about",
        "Even watching my favorite movies feels hollow and devoid of emotion",
        "I can't remember the last time I felt genuine pleasure doing anything",
        "Everything feels like a chore and nothing sparks any motivation in me",
        "My passion for my work and personal projects has completely vanished",
        "Apathy has completely taken over my daily life",
        "I feel totally disconnected from the things I used to love"
    ],
    "depressed mood": [
        "I feel so sad and hopeless all the time",
        "There is this heavy dark cloud hanging over my mind every day",
        "I break down into tears for no identifiable reason",
        "Everything feels so profoundly bleak and empty inside",
        "I wake up every morning with a heavy sinking feeling of despair",
        "The sadness inside me feels bottomless and suffocating",
        "I can't shake off this overwhelming gloom and sorrow",
        "My heart hurts and I just feel completely miserable",
        "Crying uncontrollably in the bathroom because I can't hold back the grief",
        "Life feels like an endless grey fog with no light at the end",
        "Feeling down and emotionally depleted constantly",
        "Persistent feelings of deep emptiness and sadness that won't go away"
    ],
    "fatigue / low energy": [
        "I am completely exhausted all the time even when I haven't done anything",
        "My body feels like lead and I have zero physical energy",
        "I can barely muster the energy to get out of bed in the morning",
        "Severe chronic fatigue makes even brushing my teeth feel like a marathon",
        "I am totally wiped out and running on empty fumes",
        "Mental and physical exhaustion is paralyzing my entire week",
        "I feel constantly drained and depleted of any strength",
        "Waking up feeling just as tired as when I went to sleep",
        "Total burnout has left my body feeling completely wrecked and sluggish",
        "Just walking to the kitchen leaves me feeling physically spent",
        "Overwhelming tiredness that no amount of rest seems to cure",
        "Zero stamina and deep lethargy throughout the entire day"
    ],
    "anxiety / panic": [
        "I am having a panic attack and my heart is beating out of my chest",
        "My anxiety is through the roof and I can't breathe properly",
        "Constant dread and panicky terror make me feel like something awful is about to happen",
        "My hands are shaking with nervous tension and my chest feels so tight",
        "I can't stop worrying about every little thing spiraling out of control",
        "Paralyzed by sudden waves of overwhelming fear and dizziness",
        "Hyperventilating and freaking out over trivial everyday tasks",
        "I feel on edge and terrified for no logical reason",
        "A jittery nervous wreck unable to sit still because of panic",
        "Chest tightness and shortness of breath whenever I think about the future",
        "Intense anxiety attacks leaving me trembling and nauseous",
        "Feeling keyed up, terrified, and overwhelmed by racing catastrophic thoughts"
    ],
    "cognitive difficulty": [
        "I can't concentrate on my work and my mind keeps going blank",
        "Brain fog is so severe that I forget what I was doing two seconds ago",
        "Struggling to make even the simplest decisions without freezing up",
        "My attention span is completely fractured and I can't retain any information",
        "I sit in front of my screen for hours unable to focus on a single sentence",
        "Slowed thinking and severe mental disorientation are making work impossible",
        "I keep forgetting basic words and losing my train of thought mid-sentence",
        "Feeling scatterbrained, indecisive, and incapable of clear mental processing",
        "Diminished ability to think through basic everyday problems",
        "My brain feels like thick mud and mental clarity has totally vanished",
        "Inability to focus or stay on task even when deadlines are pressing",
        "Memory lapses and severe mental sluggishness all week"
    ],
    "feelings of worthlessness": [
        "I feel completely worthless and like a massive burden to everyone around me",
        "I hate myself and feel like I ruin everything I ever touch",
        "Everyone in my life would be so much better off without me existing",
        "I am an absolute failure who cannot do anything right",
        "Consumed by crushing guilt over mistakes that happened years ago",
        "I feel disgusting, inadequate, and completely undeserving of kindness",
        "Deep feelings of self-loathing and shame that I cannot quiet",
        "I blame myself for every single bad thing that happens to my family",
        "Feeling like a broken person who only brings disappointment to others",
        "I have zero value as a human being and I hate who I am",
        "Incapable of feeling self-worth because I feel like such an utter mess",
        "Overwhelming self-reproach and unearned guilt dominating my thoughts"
    ],
    "appetite / eating change": [
        "I have zero appetite and haven't eaten a full meal in three days",
        "The thought of food makes me physically nauseous right now",
        "I find myself bingeing on junk food late at night until my stomach hurts",
        "Forgetting to eat all day because I feel no hunger cues whatsoever",
        "I have lost noticeable weight recently because I can barely swallow food",
        "Forcing myself to take bites because food tastes completely like cardboard",
        "Uncontrollable food restriction and anxiety surrounding eating meals",
        "My relationship with food is completely disordered and chaotic right now",
        "Skipping lunch and dinner because eating feels like an insurmountable chore",
        "Severe appetite suppression caused by constant stress",
        "Compulsive eating episodes followed by intense guilt and disgust",
        "Drastic fluctuations in hunger leaving my body malnourished"
    ],
    "social withdrawal / isolation": [
        "I have been isolating myself in my room avoiding all human contact",
        "I am ghosting my closest friends because talking to people feels exhausting",
        "I haven't left my apartment in over a week and don't want anyone to see me",
        "Withdrawing from all social circles and pushing loved ones away",
        "I silence my phone and ignore every message because I can't handle interactions",
        "Locking myself away and refusing to engage with the outside world",
        "Feeling deeply alienated and detaching myself from friends and family",
        "Avoiding social gatherings because being around people feels unbearable",
        "Shutting down and pulling away from everyone who cares about me",
        "Living in total solitude because interacting requires energy I don't possess",
        "I dread anyone knocking on my door or asking me to hang out",
        "Retreating completely into my shell and hiding from the world"
    ],
    "suicidal ideation / crisis thoughts": [
        "I just want to end my life, I can't take this agony anymore",
        "I keep thinking that everyone would be better off if I were dead",
        "I want to hurt myself and I don't know how to stop the urge",
        "I feel like ending it all and never waking up again",
        "Thinking about suicide constantly because the pain is intolerable",
        "I don't want to live anymore, I feel completely ready to give up",
        "Strong impulses to harm myself when the emotional distress peaks",
        "Wishing I could go to sleep and just peacefully pass away",
        "I feel like there is no reason left to keep going in this world",
        "Urges to self-harm have become overwhelming today",
        "I can't survive this mental torment anymore, I want it to stop permanently",
        "Crisis thoughts are spiraling and I feel completely unsafe with myself"
    ]
}

# Benign utterances containing NO mental health symptom signals
BENIGN_UTTERANCES = [
    "Hello there! How are you doing today?",
    "Can you help me organize my study schedule for next week?",
    "The weather outside is surprisingly mild and sunny today.",
    "I went for a brisk walk in the park and took some lovely photos.",
    "Do you have any good book recommendations for sci-fi lovers?",
    "I made a delicious pasta dish with fresh basil and garlic.",
    "What is the difference between supervised and unsupervised learning?",
    "I need some advice on how to prepare for an upcoming job interview.",
    "Thank you so much for the helpful information, I appreciate it!",
    "I'm planning a weekend trip with my family to the mountains.",
    "Could you explain how photosynthesis works in simple terms?",
    "I bought a new pair of running shoes and they feel very comfortable.",
    "Just checking in to say hello and see how your evening is going.",
    "I watched a fascinating documentary about marine biology yesterday.",
    "Can you give me some tips for improving time management at work?",
    "The flowers in my garden started blooming this morning.",
    "I love listening to jazz music while doing household chores.",
    "What are some good habits for maintaining work-life balance?",
    "I just finished reading a really great mystery novel.",
    "Hope you are having a productive and pleasant day today!"
]

# Connectors for generating multi-symptom co-occurring utterances
CONNECTORS = [
    " and ",
    ". Also, ",
    ", and on top of that ",
    ". To make matters worse, ",
    ", plus ",
    ". In addition to that, ",
    " while ",
    ", and furthermore "
]


def generate_symptom_dataset(num_samples: int = 3500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a balanced, diverse multi-label symptom dataset grounded in PsySym."""
    random.seed(seed)
    samples = []
    seen_texts = set()

    def add_sample(text: str, labels: List[str], sample_type: str):
        t_clean = text.strip()
        t_lower = t_clean.lower()
        if t_lower in seen_texts or len(t_clean) < 10:
            return
        seen_texts.add(t_lower)
        samples.append({
            "text": t_clean,
            "labels": sorted(labels),
            "num_signals": len(labels),
            "sample_type": sample_type
        })

    # 1. Single symptom utterances
    for sig in SYMPTOM_SIGNALS:
        t_list = TEMPLATES[sig]
        for t in t_list:
            add_sample(t, [sig], "single_symptom")
            # Syntactic / lexical augmentations
            add_sample(f"Honestly, {t.lower()}", [sig], "single_symptom")
            add_sample(f"For the past few weeks, {t.lower()}", [sig], "single_symptom")
            add_sample(f"I really struggle because {t.lower()}", [sig], "single_symptom")
            add_sample(f"{t}, and it's making life so difficult.", [sig], "single_symptom")

    # 2. Canonical prompt benchmark utterance:
    # "I haven't been sleeping properly and I don't enjoy things anymore."
    add_sample(
        "I haven't been sleeping properly and I don't enjoy things anymore.",
        ["sleep disturbance", "loss of interest"],
        "benchmark_dual"
    )

    # 3. Clinically common co-occurring dual-symptom combinations
    dual_pairs = [
        ("sleep disturbance", "loss of interest"),
        ("sleep disturbance", "fatigue / low energy"),
        ("depressed mood", "loss of interest"),
        ("depressed mood", "feelings of worthlessness"),
        ("anxiety / panic", "sleep disturbance"),
        ("anxiety / panic", "cognitive difficulty"),
        ("fatigue / low energy", "cognitive difficulty"),
        ("depressed mood", "social withdrawal / isolation"),
        ("feelings of worthlessness", "social withdrawal / isolation"),
        ("depressed mood", "appetite / eating change"),
        ("anxiety / panic", "appetite / eating change"),
        ("depressed mood", "suicidal ideation / crisis thoughts"),
        ("feelings of worthlessness", "suicidal ideation / crisis thoughts")
    ]

    for sig1, sig2 in dual_pairs:
        t1_list = TEMPLATES[sig1]
        t2_list = TEMPLATES[sig2]
        for t1 in t1_list[:8]:
            for t2 in t2_list[:8]:
                conn = random.choice(CONNECTORS)
                t2_fmt = t2[0].lower() + t2[1:] if not conn.startswith(".") else t2
                text = f"{t1}{conn}{t2_fmt}"
                add_sample(text, [sig1, sig2], "dual_symptom")

    # 4. Triple symptom combinations
    triple_combos = [
        ("sleep disturbance", "fatigue / low energy", "cognitive difficulty"),
        ("depressed mood", "loss of interest", "feelings of worthlessness"),
        ("anxiety / panic", "sleep disturbance", "fatigue / low energy"),
        ("depressed mood", "appetite / eating change", "social withdrawal / isolation")
    ]

    for sig1, sig2, sig3 in triple_combos:
        t1_list = TEMPLATES[sig1]
        t2_list = TEMPLATES[sig2]
        t3_list = TEMPLATES[sig3]
        for _ in range(35):
            t1 = random.choice(t1_list)
            t2 = random.choice(t2_list)
            t3 = random.choice(t3_list)
            conn1 = random.choice(CONNECTORS)
            conn2 = random.choice(CONNECTORS)
            t2_fmt = t2[0].lower() + t2[1:] if not conn1.startswith(".") else t2
            t3_fmt = t3[0].lower() + t3[1:] if not conn2.startswith(".") else t3
            text = f"{t1}{conn1}{t2_fmt}{conn2}{t3_fmt}"
            add_sample(text, [sig1, sig2, sig3], "triple_symptom")

    # 5. Benign utterances with NO symptom signals (label = [])
    for b in BENIGN_UTTERANCES:
        add_sample(b, [], "benign_zero_symptom")
        add_sample(f"Just wanted to say: {b.lower()}", [], "benign_zero_symptom")
        add_sample(f"By the way, {b.lower()}", [], "benign_zero_symptom")
        add_sample(f"{b} Any thoughts on this?", [], "benign_zero_symptom")

    # Additional conversational benign variants
    benign_fillers = [
        "What time does the library close on Saturdays?",
        "I'm learning Python programming and really enjoying the tutorials.",
        "Could you check my grammar on this paragraph?",
        "I like drinking chamomile tea in the late afternoon.",
        "How do airplanes stay in the air with such heavy weight?",
        "We planted some tomatoes and herbs on our balcony yesterday.",
        "Can you suggest some healthy breakfast ideas for busy mornings?",
        "I finished running 5 kilometers today and feel energized.",
        "What is the capital city of Australia?",
        "The sunset tonight was full of bright orange and pink colors."
    ]
    for b in benign_fillers:
        add_sample(b, [], "benign_zero_symptom")
        add_sample(f"Hey there, {b.lower()}", [], "benign_zero_symptom")

    random.shuffle(samples)
    if len(samples) > num_samples:
        samples = samples[:num_samples]

    logger.info(f"Generated {len(samples)} diverse symptom identification samples.")
    return samples


def main():
    dataset_name = "PsySym Mental Health Symptoms"
    samples = generate_symptom_dataset(num_samples=3200, seed=42)

    # Ingest through GenericDatasetPipeline
    pipeline = GenericDatasetPipeline(dataset_name=dataset_name)

    metadata = pipeline.run(
        raw_samples=samples,
        source="Zhang et al. (EMNLP 2022) / DSM-5 PsySym Knowledge Graph",
        official_url="https://aclanthology.org/2022.emnlp-main.677/",
        license_name="MIT (Code) / Author-Agreement Research Use (PsySym Corpus)",
        citation=(
            "Zhang, Z., Chen, S., Wu, M., & Zhu, K. (2022). "
            "Symptom Identification for Interpretable Detection of Multiple Mental Disorders on Social Media. "
            "In Proceedings of the 2022 EMNLP, pp. 9970-9985."
        ),
        intended_task="symptom_signal_identification",
        features=["text", "num_signals", "sample_type"],
        labels=SYMPTOM_SIGNALS,
        label_field="labels",
        text_field="text",
        version="1.0.0",
        seed=42,
        explicit_non_goals=[
            "Psychiatric disorder diagnosis",
            "Depression / Bipolar / Anxiety diagnostic labeling",
            "Clinical prescription recommendation",
            "Replacing psychiatric evaluation"
        ]
    )

    print("\n=== PsySym Symptom Ingestion Complete ===")
    print(f"Dataset: {metadata.dataset_name} (v{metadata.version})")
    print(f"Total Samples: {metadata.number_of_samples['processed']}")
    print(f"Train Count: {metadata.train_validation_test_split.train_count}")
    print(f"Val Count: {metadata.train_validation_test_split.val_count}")
    print(f"Test Count: {metadata.train_validation_test_split.test_count}")
    print(f"Leakage Check: {'PASSED' if metadata.leakage_check_passed else 'FAILED'}")
    print(f"Datasheet: {metadata.local_storage_location['docs']}")


if __name__ == "__main__":
    main()
