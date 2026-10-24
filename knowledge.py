"""Static clinical knowledge bases. Edit freely; app.py imports these."""

COMMON_INTERACTIONS = {
    frozenset(["warfarin", "aspirin"]): ("Severe", "Both impair clotting — significantly increased bleeding risk."),
    frozenset(["warfarin", "ibuprofen"]): ("Severe", "NSAIDs raise bleeding risk and can displace warfarin, boosting its effect."),
    frozenset(["warfarin", "amiodarone"]): ("Severe", "Amiodarone inhibits warfarin metabolism, sharply raising INR."),
    frozenset(["warfarin", "ciprofloxacin"]): ("Moderate", "Ciprofloxacin can potentiate warfarin's anticoagulant effect."),
    frozenset(["warfarin", "phenytoin"]): ("Moderate", "Complex two-way interaction affecting both drugs' metabolism."),
    frozenset(["aspirin", "ibuprofen"]): ("Moderate", "Ibuprofen may blunt aspirin's cardioprotective antiplatelet effect."),
    frozenset(["methotrexate", "ibuprofen"]): ("Severe", "NSAIDs reduce methotrexate clearance, risking toxicity."),
    frozenset(["methotrexate", "aspirin"]): ("Severe", "Aspirin reduces methotrexate renal clearance, risking toxicity."),
    frozenset(["lithium", "ibuprofen"]): ("Severe", "NSAIDs reduce lithium clearance, risking lithium toxicity."),
    frozenset(["digoxin", "furosemide"]): ("Moderate", "Furosemide-induced low potassium increases digoxin toxicity risk."),
    frozenset(["digoxin", "amiodarone"]): ("Severe", "Amiodarone raises digoxin levels, risking toxicity."),
    frozenset(["clopidogrel", "omeprazole"]): ("Moderate", "Omeprazole may reduce clopidogrel's antiplatelet activation."),
    frozenset(["sildenafil", "nitroglycerin"]): ("Severe", "Combination can cause life-threatening blood pressure drop."),
    frozenset(["tramadol", "sertraline"]): ("Severe", "Risk of serotonin syndrome from combined serotonergic effect."),
    frozenset(["tramadol", "fluoxetine"]): ("Severe", "Risk of serotonin syndrome from combined serotonergic effect."),
    frozenset(["alprazolam", "tramadol"]): ("Moderate", "Additive CNS/respiratory depression risk."),
    frozenset(["diazepam", "tramadol"]): ("Severe", "Additive sedation and risk of respiratory depression."),
    frozenset(["metronidazole", "alcohol"]): ("Moderate", "Can cause a disulfiram-like reaction: flushing, nausea, palpitations."),
    frozenset(["theophylline", "ciprofloxacin"]): ("Severe", "Ciprofloxacin raises theophylline levels, risking toxicity/seizures."),
    frozenset(["carbamazepine", "oral contraceptive"]): ("Moderate", "Carbamazepine induces metabolism, reducing contraceptive efficacy."),
    frozenset(["simvastatin", "clarithromycin"]): ("Severe", "Clarithromycin raises statin levels, risking rhabdomyolysis."),
    frozenset(["atorvastatin", "clarithromycin"]): ("Moderate", "Raised statin levels increase muscle toxicity risk."),
    frozenset(["insulin", "propranolol"]): ("Moderate", "Beta-blockers can mask hypoglycemia warning symptoms."),
    frozenset(["spironolactone", "lisinopril"]): ("Moderate", "Combined potassium-sparing effect risks hyperkalemia."),
    frozenset(["enalapril", "ibuprofen"]): ("Moderate", "NSAIDs can blunt ACE-inhibitor effect and stress the kidneys."),
    frozenset(["metformin", "contrast dye"]): ("Moderate", "Iodinated contrast can impair renal clearance of metformin, risking lactic acidosis."),
    frozenset(["azithromycin", "warfarin"]): ("Moderate", "Macrolides can potentiate warfarin's anticoagulant effect."),
    frozenset(["fluconazole", "warfarin"]): ("Severe", "Fluconazole strongly inhibits warfarin metabolism, sharply raising INR."),
    frozenset(["losartan", "spironolactone"]): ("Moderate", "Combined effect risks hyperkalemia, especially with renal impairment."),
    frozenset(["omeprazole", "clopidogrel"]): ("Moderate", "Same interaction as above — reduced antiplatelet activation."),
}

# Common herb / supplement → drug-class interaction knowledge base for instant offline
# results, mirroring COMMON_INTERACTIONS above. Anything not covered falls back to AI.
COMMON_HERB_INTERACTIONS = {
    frozenset(["st. john's wort", "warfarin"]): ("Severe", "St. John's Wort induces liver enzymes and can sharply lower warfarin's effect, risking clots."),
    frozenset(["st. john's wort", "sertraline"]): ("Severe", "Both raise serotonin — combined risk of serotonin syndrome."),
    frozenset(["st. john's wort", "oral contraceptive"]): ("Severe", "Induces metabolism of estrogen/progestin, reducing contraceptive efficacy."),
    frozenset(["ginkgo", "warfarin"]): ("Severe", "Ginkgo has antiplatelet activity — combined bleeding risk with warfarin."),
    frozenset(["ginkgo", "aspirin"]): ("Moderate", "Additive antiplatelet effect increases bleeding risk."),
    frozenset(["garlic", "warfarin"]): ("Moderate", "Garlic supplements have mild antiplatelet activity, raising bleeding risk."),
    frozenset(["ginger", "warfarin"]): ("Moderate", "Ginger may have antiplatelet effects, potentially raising bleeding risk."),
    frozenset(["fish oil", "warfarin"]): ("Moderate", "High-dose omega-3 can add to anticoagulant/antiplatelet effect."),
    frozenset(["turmeric", "warfarin"]): ("Moderate", "Curcumin has mild antiplatelet activity — additive bleeding risk at high supplement doses."),
    frozenset(["turmeric", "metformin"]): ("Mild", "Curcumin may modestly lower blood sugar — watch for added hypoglycemia risk."),
    frozenset(["ashwagandha", "levothyroxine"]): ("Moderate", "Ashwagandha may raise thyroid hormone levels — combined use needs monitoring."),
    frozenset(["ashwagandha", "metformin"]): ("Mild", "May have additive blood-sugar lowering effect."),
    frozenset(["ashwagandha", "alprazolam"]): ("Moderate", "Additive sedative/CNS depressant effect."),
    frozenset(["licorice", "lisinopril"]): ("Moderate", "Licorice can raise blood pressure and cause potassium loss, opposing ACE-inhibitor effect."),
    frozenset(["licorice", "digoxin"]): ("Severe", "Licorice-induced low potassium significantly raises digoxin toxicity risk."),
    frozenset(["vitamin k", "warfarin"]): ("Severe", "Vitamin K directly opposes warfarin's mechanism, reducing its anticoagulant effect."),
    frozenset(["calcium", "levothyroxine"]): ("Moderate", "Calcium supplements can reduce levothyroxine absorption if taken together."),
    frozenset(["iron", "levothyroxine"]): ("Moderate", "Iron supplements reduce levothyroxine absorption — separate dosing by 4 hours."),
    frozenset(["green tea extract", "warfarin"]): ("Moderate", "High vitamin K content in concentrated extracts can reduce warfarin's effect."),
    frozenset(["echinacea", "immunosuppressant"]): ("Moderate", "Echinacea's immune-stimulating effect may counteract immunosuppressive therapy."),
    frozenset(["melatonin", "diazepam"]): ("Mild", "Additive sedative effect — increased drowsiness."),
    frozenset(["valerian", "alprazolam"]): ("Moderate", "Additive CNS depressant/sedative effect."),
}


# Brand / alternate names -> the generic key used in the dictionaries above.
DRUG_ALIASES = {
    # analgesics / antiplatelets / anticoagulants
    "dolo": "paracetamol", "crocin": "paracetamol", "calpol": "paracetamol", "acetaminophen": "paracetamol",
    "ecosprin": "aspirin", "disprin": "aspirin", "loprin": "aspirin", "acetylsalicylic acid": "aspirin",
    "brufen": "ibuprofen", "advil": "ibuprofen",
    "coumadin": "warfarin", "warf": "warfarin", "uniwarf": "warfarin",
    "plavix": "clopidogrel", "clopilet": "clopidogrel",
    # diabetes / cardio / lipids
    "glycomet": "metformin", "glucophage": "metformin",
    "lasix": "furosemide", "lanoxin": "digoxin", "aldactone": "spironolactone",
    "zestril": "lisinopril", "listril": "lisinopril", "cozaar": "losartan", "losar": "losartan",
    "lipitor": "atorvastatin", "atorva": "atorvastatin", "zocor": "simvastatin",
    "cordarone": "amiodarone", "gtn": "nitroglycerin", "glyceryl trinitrate": "nitroglycerin",
    # GI / anti-infective
    "prilosec": "omeprazole", "omez": "omeprazole", "omee": "omeprazole",
    "flagyl": "metronidazole", "cipro": "ciprofloxacin", "ciplox": "ciprofloxacin",
    "biaxin": "clarithromycin", "zithromax": "azithromycin", "azithral": "azithromycin",
    "diflucan": "fluconazole",
    # CNS
    "ultram": "tramadol", "zoloft": "sertraline", "prozac": "fluoxetine",
    "xanax": "alprazolam", "valium": "diazepam", "tegretol": "carbamazepine", "dilantin": "phenytoin",
    "viagra": "sildenafil",
    # endocrine
    "thyronorm": "levothyroxine", "eltroxin": "levothyroxine", "thyroxine": "levothyroxine",
    "ocp": "oral contraceptive", "birth control": "oral contraceptive",
    "oral contraceptives": "oral contraceptive", "contraceptive pill": "oral contraceptive",
    # herbs / supplements
    "st johns wort": "st. john's wort", "st john's wort": "st. john's wort", "hypericum": "st. john's wort",
    "ginkgo biloba": "ginkgo", "curcumin": "turmeric", "haldi": "turmeric",
    "withania": "ashwagandha", "mulethi": "licorice", "liquorice": "licorice",
    "omega-3": "fish oil", "omega 3": "fish oil", "adrak": "ginger", "lehsun": "garlic",
    "vitamin k1": "vitamin k", "vitamin k2": "vitamin k", "green tea": "green tea extract",
    "calcium carbonate": "calcium", "iron tablet": "iron", "ferrous sulfate": "iron",
}


# ── Additional well-established interactions (review with a pharmacist before relying on them) ──
COMMON_INTERACTIONS.update({
    frozenset(["warfarin", "metronidazole"]): ("Severe", "Metronidazole inhibits warfarin metabolism, sharply raising INR and bleeding risk."),
    frozenset(["warfarin", "diclofenac"]): ("Severe", "NSAIDs raise bleeding risk and can add to warfarin's effect."),
    frozenset(["warfarin", "rifampicin"]): ("Severe", "Rifampicin induces warfarin metabolism and can markedly reduce its effect, risking clots."),
    frozenset(["warfarin", "carbamazepine"]): ("Moderate", "Carbamazepine can reduce warfarin's effect - INR needs close monitoring."),
    frozenset(["warfarin", "paracetamol"]): ("Mild", "Regular high-dose paracetamol can raise INR - monitor if used daily."),
    frozenset(["sertraline", "ibuprofen"]): ("Moderate", "SSRIs plus NSAIDs increase the risk of stomach and other bleeding."),
    frozenset(["fluoxetine", "ibuprofen"]): ("Moderate", "SSRIs plus NSAIDs increase the risk of stomach and other bleeding."),
    frozenset(["sertraline", "aspirin"]): ("Moderate", "SSRIs plus aspirin increase bleeding risk."),
    frozenset(["linezolid", "sertraline"]): ("Severe", "Linezolid has MAO-inhibitor activity - risk of serotonin syndrome."),
    frozenset(["lisinopril", "ibuprofen"]): ("Moderate", "NSAIDs can blunt ACE-inhibitor effect and stress the kidneys."),
    frozenset(["losartan", "ibuprofen"]): ("Moderate", "NSAIDs can blunt the blood-pressure effect and stress the kidneys."),
    frozenset(["diclofenac", "aspirin"]): ("Moderate", "Combined NSAIDs raise the risk of stomach ulcers and bleeding."),
    frozenset(["ibuprofen", "prednisolone"]): ("Moderate", "Steroid plus NSAID raises the risk of stomach ulcers and bleeding."),
    frozenset(["clopidogrel", "aspirin"]): ("Moderate", "Higher bleeding risk. Sometimes prescribed together on purpose - follow the doctor's plan."),
    frozenset(["amlodipine", "simvastatin"]): ("Moderate", "Amlodipine raises simvastatin levels - the simvastatin dose is usually capped."),
    frozenset(["amiodarone", "simvastatin"]): ("Moderate", "Amiodarone raises simvastatin levels, increasing muscle-damage risk."),
    frozenset(["allopurinol", "azathioprine"]): ("Severe", "Allopurinol blocks azathioprine breakdown, risking serious bone-marrow toxicity."),
    frozenset(["methotrexate", "cotrimoxazole"]): ("Severe", "Both affect folate and blood counts - risk of severe bone-marrow suppression."),
    frozenset(["glimepiride", "propranolol"]): ("Moderate", "Beta-blockers can mask the warning signs of low blood sugar."),
    frozenset(["metformin", "alcohol"]): ("Moderate", "Alcohol raises the risk of lactic acidosis and low blood sugar."),
    frozenset(["phenytoin", "fluconazole"]): ("Moderate", "Fluconazole can raise phenytoin levels, risking toxicity."),
    frozenset(["rifampicin", "oral contraceptive"]): ("Moderate", "Rifampicin can reduce contraceptive effectiveness - use another method."),
    frozenset(["sildenafil", "tamsulosin"]): ("Moderate", "Both lower blood pressure - risk of dizziness or fainting."),
})

DRUG_ALIASES.update({
    "voveran": "diclofenac", "voltaren": "diclofenac",
    "septran": "cotrimoxazole", "bactrim": "cotrimoxazole", "co-trimoxazole": "cotrimoxazole",
    "zyloric": "allopurinol", "imuran": "azathioprine", "zyvox": "linezolid",
    "norvasc": "amlodipine", "amlip": "amlodipine", "amlong": "amlodipine",
    "wysolone": "prednisolone", "deltacortril": "prednisolone",
    "amaryl": "glimepiride", "glimy": "glimepiride",
    "rifampin": "rifampicin", "r-cin": "rifampicin",
    "inderal": "propranolol", "ciplar": "propranolol",
})
