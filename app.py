from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import os

app = Flask(__name__)

# 1. DATABASE CONFIGURATION
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'postgresql://postgres:R%40chit2004@localhost:33421/packaging_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 2. DATABASE MODEL
class ProductHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100))
    sub_category = db.Column(db.String(100))
    weight = db.Column(db.Float)
    recommended_material = db.Column(db.String(100))
    predicted_cost_inr = db.Column(db.Float)
    predicted_co2 = db.Column(db.Float)

# 3. AUTOMATIC TABLE CREATION
with app.app_context():
    db.create_all()

# 4. LOAD TRAINED MODELS & DATA
# 4. LOAD TRAINED MODELS & DATA
pipeline = joblib.load('pipeline.pkl')
rf_cost_model = joblib.load('rf_cost_model.pkl')
xgb_co2_model = joblib.load('xgb_co2_model.pkl')
le_cat = joblib.load('le_cat.pkl')
le_sub = joblib.load('le_sub.pkl') # ADDED: Load the sub-category encoder
df = pd.read_csv('final_engineered_dataset.csv')

# 5. RECOMMENDATION LOGIC
def get_recommendations(input_category, input_subcategory, input_weight, is_fragile, eco_pref):
    input_category = input_category.strip().title()
    input_subcategory = input_subcategory.strip().title()
    user_fragile_val = 1 if is_fragile.lower() == 'yes' else 0

    # Encode the inputs
    try:
        cat_encoded = le_cat.transform([input_category])[0]
        sub_encoded = le_sub.transform([input_subcategory])[0]
    except:
        # Fallback if the user types something the model doesn't know
        cat_encoded = 0 
        sub_encoded = 0

    # Step 1: Filter the dataset for the actual Product Category entered
    # This ensures "Electronics" results don't show up for "Kitchenware"
    filtered_df = df[df['Product Category'] == input_category]
    
    # If no exact match, use the whole dataset as fallback
    if filtered_df.empty:
        filtered_df = df

    unique_materials = filtered_df.groupby('Packaging material').agg({
        'Packaging_Material_Encoded': 'first',
        'Strength_Encoded': 'first',
        'Strength': 'first',
        'Biodegradability score': 'mean',
        'Recyclability %': 'mean'
    }).reset_index()

    sim_data = unique_materials.copy()
    sim_data['Product_Category_Encoded'] = cat_encoded
    # Note: Ensure your training features use this or remove if not in pipeline
    sim_data['Weight Capacity (kg)'] = float(input_weight)
    
    # Features must match your pipeline exactly
    features = ['Product_Category_Encoded', 'Packaging_Material_Encoded', 
                'Strength_Encoded', 'Biodegradability score', 'Recyclability %', 'Weight Capacity (kg)']
    
    X_scaled = pipeline.transform(sim_data[features])
    sim_data['Predicted_Cost'] = rf_cost_model.predict(X_scaled)
    sim_data['Predicted_CO2'] = xgb_co2_model.predict(X_scaled)
    
    # Custom scoring
    sim_data['Env_Score'] = (sim_data['Predicted_CO2'] * 0.7) + (sim_data['Predicted_Cost'] * 0.3)
    
    if user_fragile_val == 1:
        sim_data.loc[sim_data['Strength_Encoded'] < 3, 'Env_Score'] += 10.0

    sim_data['Is_Biodegradable'] = sim_data['Biodegradability score'].apply(lambda x: 'YES' if x > 0.5 else 'NO')
    
    # Filter by user preference
    if eco_pref == 'yes':
        sim_data = sim_data[sim_data['Is_Biodegradable'] == 'YES']
    elif eco_pref == 'no':
        sim_data = sim_data[sim_data['Is_Biodegradable'] == 'NO']
    
    return sim_data.sort_values('Env_Score').head(5)

# 6. ROUTES
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()
    
    results_df = get_recommendations(
        data['category'], 
        data['sub_category'], 
        data['weight'], 
        data['is_fragile'],
        data.get('eco_preference', 'both')
    )
    
    try:
        if not results_df.empty:
            top_row = results_df.iloc[0] 
            
            new_entry = ProductHistory(
                category=data['category'],
                sub_category=data['sub_category'],
                weight=float(data['weight']),
                recommended_material=top_row['Packaging material'],
                predicted_cost_inr=float(top_row['Predicted_Cost']),
                predicted_co2=float(top_row['Predicted_CO2'])
            )
            db.session.add(new_entry)
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"PostgreSQL Error: {e}")
    
    return jsonify(results_df.to_dict(orient='records'))

# 7. STARTUP LOGIC
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)