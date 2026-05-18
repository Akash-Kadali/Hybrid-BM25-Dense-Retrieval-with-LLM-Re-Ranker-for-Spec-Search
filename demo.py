"""
Demo: General-purpose recommendation engine.

This demo shows the engine working with a healthcare/drug dataset,
but it works with ANY domain — e-commerce, legal, finance, HR, etc.
"""

import os
import pandas as pd
from engine import RecommendationEngine


def create_sample_data():
    """Create a sample drug dataset for demonstration."""
    data = {
        "Drug Name": [
            "Ibuprofen", "Acetaminophen", "Aspirin", "Naproxen", "Metformin",
            "Lisinopril", "Atorvastatin", "Omeprazole", "Amoxicillin", "Azithromycin",
            "Cetirizine", "Loratadine", "Diphenhydramine", "Prednisone", "Albuterol",
            "Fluoxetine", "Sertraline", "Gabapentin", "Tramadol", "Cyclobenzaprine",
            "Montelukast", "Pantoprazole", "Clopidogrel", "Warfarin", "Insulin Glargine",
        ],
        "Category": [
            "NSAID", "Analgesic", "NSAID", "NSAID", "Antidiabetic",
            "ACE Inhibitor", "Statin", "Proton Pump Inhibitor", "Antibiotic", "Antibiotic",
            "Antihistamine", "Antihistamine", "Antihistamine", "Corticosteroid", "Bronchodilator",
            "SSRI", "SSRI", "Anticonvulsant", "Opioid Analgesic", "Muscle Relaxant",
            "Leukotriene Inhibitor", "Proton Pump Inhibitor", "Antiplatelet", "Anticoagulant", "Insulin",
        ],
        "Primary Use": [
            "Pain relief, inflammation, fever", "Pain relief, fever reduction", "Pain, fever, blood thinning", "Pain, inflammation, arthritis", "Type 2 diabetes blood sugar control",
            "High blood pressure, heart failure", "High cholesterol reduction", "Acid reflux, GERD, stomach ulcers", "Bacterial infections (ear, throat, urinary)", "Respiratory infections, pneumonia, bronchitis",
            "Seasonal allergies, hay fever, hives", "Allergies without drowsiness", "Allergies, sleep aid, cold symptoms", "Severe inflammation, autoimmune conditions", "Asthma, bronchospasm relief",
            "Depression, anxiety, OCD, panic disorder", "Depression, anxiety, PTSD, OCD", "Nerve pain, seizures, fibromyalgia", "Moderate to severe pain", "Muscle spasms, back pain",
            "Asthma prevention, exercise-induced breathing issues", "Acid reflux, GERD, Zollinger-Ellison", "Prevent blood clots after heart attack/stroke", "Blood clot prevention, atrial fibrillation", "Type 1 and Type 2 diabetes insulin therapy",
        ],
        "Side Effects": [
            "Stomach upset, bleeding risk, kidney issues", "Liver damage at high doses", "Stomach bleeding, Reye's syndrome in children", "GI bleeding, cardiovascular risk", "Nausea, diarrhea, lactic acidosis (rare)",
            "Dry cough, dizziness, hyperkalemia", "Muscle pain, liver enzyme elevation", "Headache, diarrhea, vitamin B12 deficiency", "Diarrhea, rash, allergic reactions", "Nausea, diarrhea, abdominal pain",
            "Drowsiness, dry mouth, headache", "Minimal drowsiness, headache", "Strong drowsiness, dry mouth, urinary retention", "Weight gain, bone loss, immune suppression", "Tremor, rapid heartbeat, headache",
            "Nausea, insomnia, sexual dysfunction", "Nausea, diarrhea, insomnia, dizziness", "Dizziness, drowsiness, weight gain", "Nausea, constipation, dependency risk", "Drowsiness, dry mouth, dizziness",
            "Headache, abdominal pain, mood changes", "Headache, diarrhea, abdominal pain", "Bleeding risk, rash, diarrhea", "Major bleeding risk, requires INR monitoring", "Hypoglycemia, weight gain, injection site reactions",
        ],
        "Contraindications": [
            "Kidney disease, GI bleeding, aspirin allergy", "Severe liver disease, alcoholism", "Children under 16, bleeding disorders", "GI ulcers, severe kidney/liver disease", "Kidney disease, metabolic acidosis",
            "Pregnancy, bilateral renal artery stenosis", "Active liver disease, pregnancy", "None significant, long-term use caution", "Penicillin allergy", "Liver disease, myasthenia gravis",
            "None significant", "None significant", "Glaucoma, urinary retention, MAOIs", "Active infections, live vaccines", "None significant for acute use",
            "MAOIs within 14 days, bipolar without mood stabilizer", "MAOIs, pimozide use", "None significant, dose adjust in renal impairment", "Seizure disorder, MAOIs", "MAOIs, heart failure, hyperthyroidism",
            "Phenylketonuria (chewable tablets)", "None significant", "Active bleeding", "Active major bleeding, pregnancy", "None significant",
        ],
        "Dosage Form": [
            "Tablet, Capsule, Liquid", "Tablet, Liquid, Suppository", "Tablet", "Tablet, Liquid", "Tablet, Extended-release",
            "Tablet", "Tablet", "Capsule, Tablet", "Capsule, Liquid", "Tablet, Liquid, IV",
            "Tablet, Liquid, Chewable", "Tablet, Liquid", "Tablet, Liquid, Capsule", "Tablet, Liquid, Injection", "Inhaler, Nebulizer",
            "Capsule, Liquid", "Tablet, Liquid", "Capsule, Tablet", "Tablet, Capsule", "Tablet",
            "Tablet, Chewable, Granules", "Tablet, IV", "Tablet", "Tablet", "Injection (pen, vial)",
        ],
        "Avg Cost (30 day)": [
            "$8", "$6", "$5", "$12", "$10",
            "$15", "$12", "$20", "$10", "$15",
            "$10", "$12", "$8", "$25", "$35",
            "$15", "$12", "$18", "$20", "$10",
            "$30", "$22", "$15", "$20", "$150",
        ],
    }

    os.makedirs("sample_data", exist_ok=True)
    df = pd.DataFrame(data)
    path = "sample_data/drug_catalog.csv"
    df.to_csv(path, index=False)
    print(f"Created sample dataset: {path} ({len(df)} records)")
    return path


def main():
    print("=" * 70)
    print("  HYBRID RECOMMENDATION ENGINE DEMO")
    print("  BM25 + Semantic Search + Cross-Encoder Re-ranking")
    print("=" * 70)
    print()

    # Create sample data
    data_path = create_sample_data()

    # Initialize engine
    print("Initializing engine (downloading models on first run)...")
    engine = RecommendationEngine(
        embedding_tier="balanced",     # 'fast', 'balanced', or 'quality'
        max_chunk_tokens=256,
        use_reranker=True,             # Cross-encoder re-ranking for precision
    )

    # Ingest data
    print(f"Ingesting data from {data_path}...")
    n_chunks = engine.ingest(data_path)
    print(f"Indexed {n_chunks} chunks")
    print(f"Engine stats: {engine.stats()}")
    print()

    # Run queries
    queries = [
        "What drug is best for headache with minimal side effects?",
        "I have acid reflux and stomach ulcers, what should I take?",
        "Recommend something for anxiety and depression",
        "I need a blood thinner but I'm worried about bleeding",
        "Cheapest option for managing type 2 diabetes",
        "Non-drowsy allergy medication",
        "I have back pain and muscle spasms",
    ]

    for query in queries:
        print("-" * 70)
        print(f"QUERY: {query}")
        print("-" * 70)

        results = engine.recommend(query, top_k=3)
        for r in results:
            print(f"  #{r['rank']} (score: {r['score']:.4f})")
            # Show first 200 chars of text
            text_preview = r["text"][:200].replace("\n", " | ")
            print(f"     {text_preview}")
            print()

    # Interactive mode
    print("=" * 70)
    print("INTERACTIVE MODE — type your queries (or 'quit' to exit)")
    print("=" * 70)

    while True:
        try:
            query = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        results = engine.recommend(query, top_k=5)
        if not results:
            print("  No results found.")
            continue

        for r in results:
            print(f"\n  #{r['rank']} (score: {r['score']:.4f})")
            lines = r["text"].split("\n")
            for line in lines[:6]:
                print(f"     {line}")


if __name__ == "__main__":
    main()
