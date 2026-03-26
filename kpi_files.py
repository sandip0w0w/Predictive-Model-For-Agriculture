import pandas as pd

df = pd.read_csv('soil_data.csv')


def nutrient_score(df, crop_name):
    crop_df = df.query(f"crop == '{crop_name}'")
    
    nutrients = {
        "Nitrogen": crop_df['N'].mean(),
        "Phosphorous" : crop_df['P'].mean(),
        "Potassium": crop_df['K'].mean(),
    }

    dominant_nutrient = max(nutrients, key = nutrients.get)
    dominant_value = nutrients[dominant_nutrient]
    ph_min = crop_df["ph"].min()
    ph_max = crop_df["ph"].max()


    return dominant_nutrient, dominant_value, ph_min, ph_max


def npk_balance(df, crop_name, tolerance = 0.05):

    crop_df = df.query(f"crop == '{crop_name}'")

    total = crop_df['N'].mean() + crop_df['P'].mean() + crop_df['K'].mean()

    ideal_ratios = {
    "rice": {"N": 0.50, "P": 0.25, "K": 0.25},       # Nitrogen-heavy for tillering and yield 
    "maize": {"N": 0.45, "P": 0.25, "K": 0.30},      # Heavy feeder, high N demand 
    "chickpea": {"N": 0.20, "P": 0.40, "K": 0.40},   # Legume, lower N need, higher P for nodulation 
    "kidneybeans": {"N": 0.25, "P": 0.35, "K": 0.40},# Balanced, legumes fix N but still need P/K 
    "pigeonpeas": {"N": 0.20, "P": 0.40, "K": 0.40}, # Nitrogen-fixing, higher P/K 
    "mothbeans": {"N": 0.25, "P": 0.35, "K": 0.40},  # Drought-tolerant legume, balanced P/K 
    "mungbean": {"N": 0.25, "P": 0.35, "K": 0.40},   # Legume, moderate N, higher P/K 
    "blackgram": {"N": 0.20, "P": 0.40, "K": 0.40},  # Often fertilized with 20-20-20 or 10-20-20 
    "lentil": {"N": 0.20, "P": 0.40, "K": 0.40},     # Responsive to phosphorus, moderate K 
    "pomegranate": {"N": 0.33, "P": 0.33, "K": 0.34},# Balanced 8-8-8 or 10-10-10 
    "banana": {"N": 0.30, "P": 0.20, "K": 0.50},     # Very high K demand for fruiting 
    "mango": {"N": 0.30, "P": 0.20, "K": 0.50},      # Balanced, higher K for fruit yield 
    "grapes": {"N": 0.30, "P": 0.20, "K": 0.50},     # Nitrogen early, potassium during fruiting 
    "watermelon": {"N": 0.25, "P": 0.35, "K": 0.40}, # Higher P/K for fruit development 
    "muskmelon": {"N": 0.25, "P": 0.35, "K": 0.40},  # Often 5-10-10 or 10-10-10 
    "apple": {"N": 0.33, "P": 0.33, "K": 0.34},      # Balanced 10-10-10 
    "orange": {"N": 0.40, "P": 0.20, "K": 0.40},     # Citrus needs high N and K 
    "papaya": {"N": 0.30, "P": 0.20, "K": 0.50},     # Heavy feeders, high N and K 
    "coconut": {"N": 0.35, "P": 0.25, "K": 0.40},    # Balanced, often 12-12-17 or 17-17-17 
    "cotton": {"N": 0.40, "P": 0.30, "K": 0.30},     # High N demand, balanced P/K 
    "jute": {"N": 0.35, "P": 0.30, "K": 0.35},       # Balanced NPK for fibre yield 
    "coffee": {"N": 0.25, "P": 0.20, "K": 0.55}      # Often 2:1:3 ratio, high K for beans 
}

    ratios = {
        "N" : crop_df['N'].mean()/ total,
         "P": crop_df['P'].mean()/ total,
         "K" :crop_df['K'].mean()/ total                                                                              
    }

    def calculate_diff(actual, ideal):
        return actual - ideal
    
    ideal = ideal_ratios.get(crop_name)
    status = {nutrient: calculate_diff(ratios[nutrient], ideal[nutrient]) for nutrient in ratios}
    

    return ratios, status





    
    

