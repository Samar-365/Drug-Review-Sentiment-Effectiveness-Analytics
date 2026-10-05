# -*- coding: utf-8 -*-
"""
Sample Dataset Generator & Curator for Drug Review Sentiment Project.
Generates a realistic, curated 1,000-row sample dataset formatted exactly
like the UCI ML Drugs.com Review Dataset for Demo Mode and testing.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_dataset(output_path="data/sample/sample_drug_reviews.csv", target_rows=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    conditions_data = {
        "Depression": {
            "drugs": ["Sertraline", "Escitalopram", "Fluoxetine", "Bupropion", "Venlafaxine", "Duloxetine", "Citalopram"],
            "positive_reviews": [
                "This medication has truly given me my life back. Within four weeks my constant low mood and fatigue started lifting.",
                "Lexapro was a game changer for my severe depressive episodes. Minimal initial nausea, but smooth sailing afterwards.",
                "Wellbutrin gave me energy and motivation again without the sexual side effects of typical SSRIs. Very satisfied.",
                "After trying three different antidepressants, Prozac was the only one that stabilized my mood without feeling like a zombie.",
                "Zoloft helped lift the dark cloud that hung over me for years. It took about 5 weeks to feel the full therapeutic effect.",
                "Citalopram helped me return to work and enjoy spending time with my family again. The adjustment period was well worth it."
            ],
            "neutral_reviews": [
                "Helped moderately with depressive symptoms, but the persistent emotional numbness and vivid dreams are frustrating.",
                "Mild improvement in mood after 6 weeks, though daytime lethargy and dry mouth make it hard to stay focused at work.",
                "It works somewhat for sadness, but decreased libido and mild weight gain make me question if I should stay on it.",
                "Took the sharp edge off my depression, but didn't completely resolve feelings of low energy and brain fog."
            ],
            "negative_reviews": [
                "Caused extreme insomnia, increased suicidal ideation, and severe night sweats. Had to discontinue under doctor supervision after 10 days.",
                "Horrible experience. My anxiety skyrocketed, experienced terrible akathisia and couldn't sit still for days.",
                "Zero improvement in depression, but gained 15 pounds in two months and suffered unbearable withdrawal brain zaps.",
                "Made me emotionally detached and increased my fatigue tenfold. Did not work for my condition at all."
            ]
        },
        "Acne": {
            "drugs": ["Isotretinoin", "Doxycycline", "Minocycline", "Epiduo", "Spironolactone", "Clindamycin / benzoyl peroxide", "Tretinoin"],
            "positive_reviews": [
                "Accutane completely cured my cystic acne after a 6-month course. Dry lips were tough, but my skin is 100% clear now.",
                "Spironolactone cleared hormonal cystic breakouts along my jawline within three months. Incredible transformation.",
                "Doxycycline cleared up severe inflammatory acne before my wedding. Fast-acting when combined with topical retinoids.",
                "Tretinoin purge lasted 4 weeks, but now at 6 months my skin texture is smoother and free from deep nodules.",
                "Epiduo gel worked wonders on stubborn forehead bumps and prevented new blackheads from forming."
            ],
            "neutral_reviews": [
                "Kept active breakouts at bay while taking it daily, but acne returned as soon as the antibiotic course finished.",
                "Noticeable reduction in redness, though skin peeling and sun sensitivity require constant heavy moisturizing.",
                "Works okay for surface pimples, but does little for hormonal cysts along the chin.",
                "Moderately effective, but application causes significant initial stinging and redness that takes hours to subside."
            ],
            "negative_reviews": [
                "Destroyed my stomach lining with severe acid reflux and didn't even improve my facial cysts.",
                "Caused extreme joint pain, cracked bleeding lips, severe dry eyes, and liver enzyme spikes that forced me to stop early.",
                "Broke out in a terrible allergic rash all over my neck and chest. Complete disaster for my skin.",
                "Made my acne flare up worse than ever before with deep painful cysts that left permanent scarring."
            ]
        },
        "Anxiety": {
            "drugs": ["Alprazolam", "Clonazepam", "Lorazepam", "Buspirone", "Escitalopram", "Hydroxyzine", "Propranolol"],
            "positive_reviews": [
                "Xanax stopped my acute panic attacks in their tracks within 15 minutes. A crucial rescue medication when needed.",
                "Buspar taken twice daily significantly lowered my baseline generalized anxiety and physical tension without sedation.",
                "Propranolol is a lifesaver for performance anxiety and presentation tremors. Keeps the physical racing heart under control.",
                "Clonazepam provided stable all-day relief from debilitating agoraphobia and constant catastrophic thinking.",
                "Lexapro smoothed out daily panic spikes and helped me regain control of my daily social interactions."
            ],
            "neutral_reviews": [
                "Relieves mild social anxiety, but causes noticeable drowsiness and mild dizziness during the afternoon.",
                "Helps me fall asleep when anxious, but does not prevent morning anticipatory anxiety or stress triggers.",
                "Moderate calming effect, though building tolerance quickly requires careful dosage monitoring.",
                "Helps take the edge off panic attacks, but leaves me feeling sluggish and unmotivated for hours."
            ],
            "negative_reviews": [
                "Developed rapid tolerance and experienced terrible rebound anxiety and withdrawal tremors between doses.",
                "Made me extremely groggy, confused, and unable to drive or work safely. Felt completely dissociated.",
                "Did absolutely nothing for my panic disorder and caused intense nausea, dizziness, and heart palpitations.",
                "Paradoxical reaction—made my heart race faster and induced severe restless agitation."
            ]
        },
        "Pain": {
            "drugs": ["Tramadol", "Oxycodone", "Hydrocodone", "Gabapentin", "Naproxen", "Celecoxib", "Ibuprofen"],
            "positive_reviews": [
                "Provided immediate relief following major knee surgery. Allowed me to start physical therapy on schedule.",
                "Gabapentin reduced sharp burning diabetic nerve pain from an 8/10 to a manageable 2/10.",
                "Celebrex manages my chronic osteoarthritis flare-ups effectively without upsetting my sensitive stomach.",
                "Tramadol managed post-operative pain well enough to allow restorative sleep without excessive sedation.",
                "Naproxen 500mg twice daily relieved severe tendonitis inflammation within 48 hours."
            ],
            "neutral_reviews": [
                "Dulls sharp nerve pain, but leaves me in a mental fog with mild morning headaches.",
                "Provides about 4 hours of pain relief, but wears off faster than expected and causes mild constipation.",
                "Helps with joint stiffness in the morning, but pain returns in the evening during strenuous activities.",
                "Mild to moderate reduction in lower back pain, accompanied by lightheadedness and dry mouth."
            ],
            "negative_reviews": [
                "Severe nausea, vomiting, and dizziness within 30 minutes of taking the first tablet. Unbearable side effects.",
                "Did not touch my severe sciatica pain at all, but caused extreme constipation and mental confusion.",
                "Caused alarming stomach ulcers and internal bleeding after just one week of prescribed use.",
                "Severe allergic reaction with full-body hives and breathing difficulty. Avoid at all costs."
            ]
        },
        "Birth Control": {
            "drugs": ["Etonogestrel", "Ethinyl estradiol / levonorgestrel", "Ethinyl estradiol / norethindrone", "Levonorgestrel", "Drospirenone / ethinyl estradiol", "Medroxyprogesterone", "Ethinyl estradiol / norgestimate"],
            "positive_reviews": [
                "Nexplanon implant has been fantastic. No periods, no cramping, and total peace of mind for 3 years.",
                "Yaz completely cleared my hormonal acne, regulated my cycle to exactly 28 days, and eliminated PMS mood swings.",
                "Mirena IUD insertion was mildly uncomfortable for a few seconds, but 4 years of zero periods and no hassle is worth it.",
                "Very predictable light cycles with minimal cramping. No weight gain or mood changes whatsoever.",
                "Sprintec has worked flawlessly for pregnancy prevention and skin clearance with zero adverse effects."
            ],
            "neutral_reviews": [
                "Effective for pregnancy prevention, but experienced irregular spotting and slight breast tenderness for the first 4 months.",
                "Regulated my periods well, but noticed a slight drop in libido and mild water retention before cycles.",
                "Convenient weekly patch, but edges tend to peel slightly and cause minor skin redness at application sites.",
                "Periods became much lighter, though occasional unpredictable breakthrough bleeding occurs mid-pack."
            ],
            "negative_reviews": [
                "Caused severe depression, uncontrollable crying spells, 20-pound weight gain, and cystic acne within 3 months.",
                "Suffered debilitating daily migraines and blood pressure spikes. My doctor instructed immediate removal.",
                "Continuous heavy bleeding for 6 straight weeks accompanied by excruciating ovarian cramps.",
                "Completely destroyed my mental health, caused severe mood swings, and killed my energy levels."
            ]
        },
        "High Blood Pressure": {
            "drugs": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide", "Metoprolol", "Valsartan", "Carvedilol"],
            "positive_reviews": [
                "Brought my blood pressure down from 160/100 to 120/78 within three weeks. No noticeable side effects.",
                "Losartan works smoothly with zero cough or fatigue. My primary care physician is very pleased with the readings.",
                "Metoprolol normalized my resting heart rate and blood pressure while reducing heart palpitations.",
                "Combination of Lisinopril and HCTZ lowered my readings to optimal range with once-daily morning dosing.",
                "Amlodipine effectively controlled stage 2 hypertension with excellent stability across 24-hour ambulatory monitoring."
            ],
            "neutral_reviews": [
                "Effective at lowering BP, but causes mild swelling in my ankles and feet by the end of the day.",
                "Blood pressure is well controlled, though occasional dizziness when standing up quickly requires caution.",
                "Numbers look good on paper, but I feel slightly more fatigued during intense cardio workouts.",
                "Maintains stable readings, but requires regular potassium and kidney function monitoring."
            ],
            "negative_reviews": [
                "Developed the notorious persistent dry hacking cough that kept me awake all night. Had to switch to an ARB.",
                "Caused severe peripheral edema with feet swelling so badly I couldn't put my shoes on.",
                "Made my blood pressure crash too low causing fainting spells and severe dizziness upon standing.",
                "Suffered extreme fatigue, cold extremities, and brain fog that made daily work impossible."
            ]
        },
        "Insomnia": {
            "drugs": ["Zolpidem", "Trazodone", "Temazepam", "Eszopiclone", "Doxepin", "Melatonin"],
            "positive_reviews": [
                "Ambien puts me to sleep in 20 minutes and allows 7 full hours of restful, uninterrupted sleep.",
                "Low-dose Trazodone (50mg) restored natural sleep architecture without dependency or grogginess.",
                "Restoril provided crucial short-term relief during severe sleep disruption without morning hangover."
            ],
            "neutral_reviews": [
                "Helps me fall asleep reliably, but leaves a slight metallic taste and morning grogginess for an hour.",
                "Works for falling asleep, but does not prevent waking up at 4 AM."
            ],
            "negative_reviews": [
                "Experience sleepwalking and memory lapses where I ate entire meals without remembering next morning.",
                "Severe next-day sedation, brain fog, and severe headaches. Felt like a zombie all day."
            ]
        },
        "Migraine": {
            "drugs": ["Sumatriptan", "Rizatriptan", "Topiramate", "Erenumab", "Propranolol"],
            "positive_reviews": [
                "Imitrex aborts acute throbbing migraines within 30 minutes. An indispensable rescue drug.",
                "Maxalt melts in the mouth and stops aura and severe nausea before the headache peaks.",
                "Aimovig monthly injection reduced my monthly migraine days from 14 down to just 2."
            ],
            "neutral_reviews": [
                "Relieves migraine pain, but causes intense chest tightness and neck heaviness for an hour.",
                "Reduces frequency of headaches, but tingling sensations in fingers and word-finding difficulty occur."
            ],
            "negative_reviews": [
                "Topamax caused severe cognitive decline, memory loss, and kidney stone symptoms.",
                "Did not abort the migraine and caused severe rebound headache 24 hours later."
            ]
        }
    }

    start_date = datetime(2010, 1, 1)
    end_date = datetime(2018, 12, 31)
    date_range_days = (end_date - start_date).days

    rows = []
    
    # Class distribution targets:
    # ~55% Positive (ratings 7-10), ~18% Neutral (ratings 4-6), ~27% Negative (ratings 1-3)
    categories = ["positive", "neutral", "negative"]
    category_weights = [0.55, 0.18, 0.27]
    
    rating_maps = {
        "positive": [7, 8, 9, 10],
        "neutral": [4, 5, 6],
        "negative": [1, 2, 3]
    }

    condition_keys = list(conditions_data.keys())
    # Top 6 conditions get higher weighting
    condition_weights = [0.22, 0.20, 0.18, 0.15, 0.15, 0.05, 0.03, 0.02]

    for i in range(target_rows):
        unique_id = 100000 + i + 1
        
        # Select condition
        cond_name = random.choices(condition_keys, weights=condition_weights, k=1)[0]
        cond_info = conditions_data[cond_name]
        
        # Select sentiment category and rating
        sentiment_cat = random.choices(categories, weights=category_weights, k=1)[0]
        rating = random.choice(rating_maps[sentiment_cat])
        
        # Select drug and review text
        drug = random.choice(cond_info["drugs"])
        base_review = random.choice(cond_info[f"{sentiment_cat}_reviews"])
        
        # Add realistic variability to review
        variations = [
            f'"{base_review}"',
            f'"{base_review} Overall rating: {rating}/10."',
            f'"{base_review} Diagnosed with {cond_name.lower()}."'
        ]
        review_text = random.choice(variations)
        
        # Generate random realistic date (e.g., "February 24, 2017")
        random_days = random.randint(0, date_range_days)
        review_date = start_date + timedelta(days=random_days)
        formatted_date = review_date.strftime("%B %d, %Y")
        
        # Generate realistic useful count (higher ratings or detailed reviews often have more useful counts)
        if sentiment_cat == "positive":
            useful_count = int(np.random.exponential(scale=24))
        elif sentiment_cat == "negative":
            useful_count = int(np.random.exponential(scale=18))
        else:
            useful_count = int(np.random.exponential(scale=10))
            
        rows.append({
            "uniqueID": unique_id,
            "drugName": drug,
            "condition": cond_name,
            "review": review_text,
            "rating": rating,
            "date": formatted_date,
            "usefulCount": useful_count
        })

    df = pd.DataFrame(rows)
    
    # Ensure exact column order and format
    cols = ["uniqueID", "drugName", "condition", "review", "rating", "date", "usefulCount"]
    df = df[cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Successfully generated {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
