"""Synthetic support-routing dataset for validating the training pipeline.

Ten departments, six message templates each with slot fillers. Every row is a rendered message,
a fixed question, the correct department plus one to five distractors in random order. The
correct answer depends on the message, so the shuffled-context control is meaningful. This is
plumbing validation only: a head that scores well here has not been shown to work on real data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

QUESTION = "Which team should handle this request?"
DEPARTMENTS = {
    "billing": ("Billing", "payments, invoices, duplicate charges, and refunds"),
    "technical": ("Technical support", "bugs, crashes, errors, and login problems"),
    "sales": ("Sales", "new purchases, plans, pricing, and product recommendations"),
    "shipping": ("Shipping", "delivery status, tracking, and lost or delayed parcels"),
    "returns": ("Returns", "sending items back, exchanges, and return labels"),
    "account": ("Account security", "password resets, two-factor codes, and suspicious sign-ins"),
    "legal": ("Legal", "terms of service, privacy requests, and data deletion"),
    "careers": ("Careers", "job applications, interviews, and open positions"),
    "partnerships": ("Partnerships", "reseller, affiliate, and integration proposals"),
    "press": ("Press", "media inquiries, interviews, and brand assets"),
}
TEMPLATES = {
    "billing": [
        "I was charged {amount} twice for order {order} and want the duplicate refunded.",
        "My invoice for {product} shows {amount} but the quote was lower. Can someone correct it?",
        "The card on file was billed after I cancelled. Please refund {amount}.",
        "Where can I download the receipt for order {order}?",
        "Please add our VAT number to the invoice from {date}.",
        "Payment for {product} failed three times even though the card works elsewhere.",
    ],
    "technical": [
        "The app crashes every time I open {product} on my phone.",
        "I get an error 500 when uploading anything larger than a few megabytes to {product}.",
        "Since the update on {date}, {product} freezes on the loading screen.",
        "The export button in {product} does nothing in Firefox.",
        "Notifications from {product} stopped arriving on my laptop {days} days ago.",
        "Sync between my devices has been broken for {days} days.",
    ],
    "sales": [
        "We are a team of {count} looking for the right plan for {product}.",
        "Does the enterprise tier of {product} include SSO, and what would {count} seats cost?",
        "Can you recommend which {product} option fits a small agency in {city}?",
        "I would like a quote for upgrading {count} users from the starter plan.",
        "Is there a discount for annual billing on {product}?",
        "Before buying, can {name} get a demo of {product}?",
    ],
    "shipping": [
        "Order {order} was due {days} days ago and the tracking has not moved.",
        "The parcel for order {order} shows delivered in {city} but nothing arrived.",
        "Can I change the delivery address for order {order} before it ships?",
        "The tracking number for order {order} says invalid.",
        "How long does delivery to {city} usually take for {product}?",
        "Order {order} arrived with the box crushed on one side.",
    ],
    "returns": [
        "I want to send back the {product} from order {order}; it does not fit.",
        "How do I get a return label for order {order}?",
        "Can I exchange the {product} from order {order} for a different size?",
        "It has been {days} days since I returned order {order} and I have heard nothing.",
        "The item in order {order} is not what I ordered. What is the return process?",
        "Is the return window still open for an order placed on {date}?",
    ],
    "account": [
        "Someone signed into my account from {city} on {date} and I do not recognise it.",
        "I never receive the two-factor code by text, even after {days} attempts.",
        "I am locked out after too many password attempts on {date}.",
        "Please reset the password for {name}'s account.",
        "I want to enable two-factor authentication for {product} but cannot find the setting.",
        "My account email was changed on {date} without my permission.",
    ],
    "legal": [
        "Please delete all personal data you hold about {name} under GDPR.",
        "Where can I find the current terms of service and the data processing agreement?",
        "I need a copy of your privacy policy for a compliance audit due {date}.",
        "Our counsel in {city} has questions about the liability clause in your contract.",
        "How do I file a copyright takedown for content posted on {date}?",
        "Do you sign custom data processing agreements for customers in {city}?",
    ],
    "careers": [
        "I applied for the engineering role in {city} {days} days ago and have not heard back.",
        "Is the {city} position posted on {date} still open?",
        "Can I reschedule my interview with {name}?",
        "Do you offer internships in {city} this summer?",
        "What is the remote policy for the role {name} interviewed me for?",
        "I would like to withdraw the application I submitted on {date}.",
    ],
    "partnerships": [
        "We run a marketplace in {city} and would like to integrate {product} for our customers.",
        "Is there a reseller programme for agencies of {count} people?",
        "I am interested in becoming an affiliate for {product}; what are the terms?",
        "Our company wants to co-market with you at an event in {city} on {date}.",
        "Can we get API access to build a joint integration with {product}?",
        "Who handles technology partnerships? {name} asked me to reach out.",
    ],
    "press": [
        "I am a journalist writing about {product}; could I interview {name}?",
        "Where can I download your logo and brand assets before {date}?",
        "We would like a comment for an article publishing on {date}.",
        "Is {name} available for a podcast about your founding story?",
        "Please send your latest press kit to our office in {city}.",
        "Can you confirm the numbers in the announcement from {date}?",
    ],
}
SLOTS = {
    "name": ["Ana", "Bilal", "Chen", "Dana", "Emeka", "Fatima", "Gustavo", "Hana", "Ivan", "Jun"],
    "product": ["Ledger", "Beacon", "Relay", "Compass", "Harbor", "Quill", "Vector", "Summit"],
    "amount": ["$19.99", "$49", "$120", "$250", "$1,200", "$8.50", "$399", "$75"],
    "days": ["two", "three", "five", "seven", "ten", "fourteen"],
    "city": [
        "Berlin",
        "Toronto",
        "Lagos",
        "Lisbon",
        "Osaka",
        "Denver",
        "Nairobi",
        "Madrid",
        "Austin",
        "Dublin",
    ],
    "count": ["five", "twelve", "twenty", "forty", "eighty", "two hundred"],
    "date": ["March 3", "April 18", "May 9", "June 27", "August 12", "September 1"],
}
OPENERS = ["", "", "Hi, this is {name}. ", "Hello, {name} here. ", "Quick question: ", "Urgent: "]


def make_row(rng: random.Random) -> dict:
    department = rng.choice(list(DEPARTMENTS))
    fill = {slot: rng.choice(values) for slot, values in SLOTS.items()}
    fill["order"] = f"#{rng.randint(10000, 99999)}"
    context = rng.choice(OPENERS).format(**fill) + rng.choice(TEMPLATES[department]).format(**fill)
    distractors = rng.sample([d for d in DEPARTMENTS if d != department], rng.randint(1, 5))
    keys = [department, *distractors]
    rng.shuffle(keys)
    return {
        "request": {
            "context": context,
            "question": QUESTION,
            "options": [
                {"id": k, "text": f"{DEPARTMENTS[k][0]}: {DEPARTMENTS[k][1]}"} for k in keys
            ],
        },
        "expected_option_id": department,
    }


def generate(
    out: Path, train: int = 2000, validation: int = 400, test: int = 400, seed: int = 7
) -> dict:
    """Write train/validation/test JSONL with unique contexts across all three splits."""
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {}
    for split, size in (("train", train), ("validation", validation), ("test", test)):
        path = out / f"{split}.jsonl"
        if path.exists():
            raise FileExistsError(path)
        rows = []
        attempts = 0
        while len(rows) < size:
            attempts += 1
            if attempts > size * 50:
                raise RuntimeError("Could not generate enough unique contexts")
            row = make_row(rng)
            if row["request"]["context"] in seen:
                continue
            seen.add(row["request"]["context"])
            rows.append(row)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        counts[split] = len(rows)
    meta = {"seed": seed, "departments": len(DEPARTMENTS), "question": QUESTION, **counts}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
