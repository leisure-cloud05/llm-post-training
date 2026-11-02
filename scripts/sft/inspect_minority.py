from collections import Counter
from datasets import load_dataset
import random

DATASET = "VladislavMarinovich/polaris-support-tickets-v2"

ds = load_dataset(DATASET, split="train")

# Keep ordinary tickets only, matching our current experiment.
ordinary = [
    row for row in ds
    if row["event_id"] is None
    or not str(row["event_id"]).strip()
]

print("Ordinary tickets:", len(ordinary))

print("\nPriority distribution:")
print(dict(Counter(row["priority"] for row in ordinary)))

critical = [
    row for row in ordinary
    if row["priority"] == "critical"
]

print(f"\nCritical tickets: {len(critical)}")

print("\n===== CRITICAL EXAMPLES =====")
for i, row in enumerate(critical[:15], start=1):
    print(f"\nExample {i}")
    print("Topic:", row["topic"])
    print("Sentiment:", row["sentiment"])
    print("Subject:", row["subject"])
    print("Body:", row["body"])

    high = [
    row for row in ordinary
    if row["priority"] == "high"
]

rng = random.Random(42)

print("\n===== HIGH EXAMPLES =====")
for i, row in enumerate(rng.sample(high, 10), start=1):
    print(f"\nExample {i}")
    print("Topic:", row["topic"])
    print("Sentiment:", row["sentiment"])
    print("Subject:", row["subject"])
    print("Body:", row["body"])