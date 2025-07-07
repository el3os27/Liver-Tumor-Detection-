from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os
import cv2

app = Flask(__name__)

# Load the trained model
model_path = "liver_tumor_segmentation_final.keras"
model = load_model(model_path)

def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the path.")
    
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def calculate_tumor_volume(mask, pixel_spacing=(1, 1, 1)):
    tumor_pixels = np.sum(mask > 0.5)
    voxel_volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]
    volume = tumor_pixels * voxel_volume
    return volume

def determine_liver_disease_type(image_path, mask):
    mask = mask.squeeze()
    tumor_area = np.sum(mask > 0.5)
    total_area = mask.shape[0] * mask.shape[1]
    tumor_percentage = (tumor_area / total_area) * 100
    tumor_volume = calculate_tumor_volume(mask, (1, 1, 1))
    
    if tumor_percentage < 1:
        return {
            "type": "No Tumor Detected",
            "description": "No significant tumor was detected in the liver scan.",
            "treatment": "No specific treatment needed. Regular checkups recommended.",
            "causes": "1. Chronic Viral Hepatitis (B/C): Leading cause worldwide\n"
                      "2. Alcohol Abuse: Long-term excessive consumption\n"
                      "3. Non-Alcoholic Fatty Liver Disease (NAFLD): Associated with obesity/diabetes\n"
                      "4. Aflatoxin Exposure: From contaminated foods\n"
                      "5. Genetic Factors: Family history increases risk\n"
                      "6. Metabolic Disorders: Hemochromatosis, Wilson's disease",
            "prevention": "1. Vaccination: Hepatitis B vaccine for all\n"
                         "2. Alcohol Moderation: Limit to recommended levels\n"
                         "3. Healthy Diet: Mediterranean diet recommended\n"
                         "4. Regular Exercise: 150+ minutes/week\n"
                         "5. Screening: For high-risk individuals\n"
                         "6. Toxin Avoidance: Proper food storage",
            "volume": tumor_volume,
            "volume_category": "Very Small (<0.1 cm³)"
        }
    elif tumor_percentage < 10:
        return {
            "type": "Benign Liver Tumor (e.g., Hemangioma)",
            "description": f"Small benign tumor detected (Volume: {tumor_volume:.2f} mm³). These are usually non-cancerous and don't spread. Common types include hemangiomas, focal nodular hyperplasia (FNH), and hepatic adenomas.",
            "treatment": "Monitoring with regular imaging. Treatment usually not needed unless symptoms develop. Surgical removal may be considered for large or symptomatic tumors.",
            "causes": "1. Often congenital (present from birth)\n"
                     "2. Hormonal factors (particularly oral contraceptive use in hepatic adenomas)\n"
                     "3. Vascular malformations\n"
                     "4. Unknown causes in many cases",
            "prevention": "1. Limit hormone medications when possible\n"
                         "2. Regular medical checkups for those with known benign tumors\n"
                         "3. Avoid unnecessary medications\n"
                         "4. Maintain healthy lifestyle",
            "volume": tumor_volume,
            "volume_category": "Small (0.1-1 cm³)"
        }
    elif tumor_percentage < 30:
        return {
            "type": "Early Stage Liver Cancer (HCC)",
            "description": f"Moderate-sized tumor detected (Volume: {tumor_volume:.2f} mm³, ~{tumor_volume/1000:.1f} cm³). This may indicate early-stage hepatocellular carcinoma (HCC), the most common type of primary liver cancer.",
            "treatment": "Surgical options include resection or liver transplant. Localized treatments like radiofrequency ablation or TACE (transarterial chemoembolization) may be options. Systemic therapies like sorafenib may be considered.",
            "causes": "1. Chronic liver disease (cirrhosis) from hepatitis B/C\n"
                     "2. Alcohol-related liver disease\n"
                     "3. NAFLD/NASH (non-alcoholic steatohepatitis)\n"
                     "4. Genetic predisposition\n"
                     "5. Environmental toxins (aflatoxins)",
            "prevention": "1. Vaccination against hepatitis B\n"
                         "2. Treatment of chronic hepatitis C\n"
                         "3. Management of fatty liver disease\n"
                         "4. Alcohol moderation\n"
                         "5. Regular screening for high-risk patients",
            "volume": tumor_volume,
            "volume_category": "Medium (1-30 cm³)"
        }
    else:
        return {
            "type": "Advanced Liver Cancer or Metastasis",
            "description": f"Large tumor detected (Volume: {tumor_volume:.2f} mm³, ~{tumor_volume/1000:.1f} cm³), possibly indicating advanced liver cancer or metastasis from other organs. This may include HCC, cholangiocarcinoma, or metastatic tumors from colon, breast or other cancers.",
            "treatment": "Multidisciplinary approach including systemic therapy (chemotherapy, targeted therapy, immunotherapy), radiation (SBRT), or palliative care depending on stage. Clinical trials may be an option.",
            "causes": "1. Advanced chronic liver disease\n"
                     "2. Untreated viral hepatitis\n"
                     "3. Environmental carcinogens (aflatoxins)\n"
                     "4. Metastatic spread from other cancers\n"
                     "5. Long-term alcohol abuse",
            "prevention": "1. Early detection through regular screening in high-risk patients\n"
                         "2. Management of underlying liver disease\n"
                         "3. Avoidance of known risk factors\n"
                         "4. Healthy lifestyle choices\n"
                         "5. Regular medical follow-ups",
            "volume": tumor_volume,
            "volume_category": "Large (>30 cm³)"
        }

def generate_segmentation_visualization(image_path, mask):
    original_img = Image.open(image_path)
    original_img = original_img.resize((256, 256))
    original_img = np.array(original_img)
    
    if len(original_img.shape) == 2:
        original_img = np.stack((original_img,)*3, axis=-1)
    
    mask = mask.squeeze()
    mask = (mask > 0.5).astype(np.uint8)
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask == 1] = [255, 0, 0]
    
    alpha = 0.5
    blended = cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    ax3.imshow(blended)
    ax3.set_title('Tumor Detection')
    ax3.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_relationship_diagrams(age, gender, chronic_diseases, liver_enzymes, tumor_volume):
    try:
        clean_liver_enzymes = float(''.join(c for c in str(liver_enzymes) if c.isdigit() or c == '.'))
    except ValueError:
        clean_liver_enzymes = 0
        
    diagrams = []
    
    # Age risk diagram
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    age_categories = ["<40", "40-60", ">60"]
    age_risk_values = [0, 0, 0]
    
    if int(age) < 40:
        age_risk_values[0] = 1
    elif 40 <= int(age) <= 60:
        age_risk_values[1] = 1
    else:
        age_risk_values[2] = 1
    
    ax1.pie(age_risk_values, labels=age_categories, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_title("Age vs Liver Disease Risk")
    diagrams.append(fig1)
    
    # Gender risk diagram
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    gender_labels = ['Male', 'Female']
    gender_risk_values = [1 if gender == "Male" else 0]
    gender_risk_percent = [gender_risk_values[0] * 100, (1 - gender_risk_values[0]) * 100]
    ax2.pie(gender_risk_percent, labels=gender_labels, autopct='%1.1f%%', colors=['lightblue', 'pink'])
    ax2.set_title("Gender vs Liver Disease Risk")
    diagrams.append(fig2)
    
    # Chronic diseases diagram
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    chronic_categories = ["Patient Chronic Diseases"]
    chronic_risk_values = [1 if chronic_diseases.lower() in ["hepatitis", "cirrhosis", "fatty liver"] else 0]
    ax3.bar(chronic_categories, chronic_risk_values, color=['red'])
    ax3.set_ylim(0, 1)
    ax3.set_title("Chronic Diseases vs Liver Disease Risk")
    ax3.set_ylabel("Chronic Liver Conditions (1=Yes, 0=No)")
    diagrams.append(fig3)
    
    # Liver enzymes diagram
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    enzyme_labels = ['High Risk (Elevated)', 'Normal']
    enzyme_risk_values = [1 if clean_liver_enzymes > 40 else 0]
    enzyme_risk_percent = [enzyme_risk_values[0] * 100, (1 - enzyme_risk_values[0]) * 100]
    ax4.pie(enzyme_risk_percent, labels=enzyme_labels, autopct='%1.1f%%', colors=['red', 'green'])
    ax4.set_title("Liver Enzymes vs Disease Risk")
    diagrams.append(fig4)
    
    # Tumor Volume Diagram
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    volume_categories = ["<1cm³", "1-5cm³", "5-10cm³", ">10cm³"]
    volume_values = [0, 0, 0, 0]
    
    vol_cm3 = tumor_volume / 1000
    if vol_cm3 < 1: volume_values[0] = 1
    elif 1 <= vol_cm3 < 5: volume_values[1] = 1
    elif 5 <= vol_cm3 < 10: volume_values[2] = 1
    else: volume_values[3] = 1
    
    ax5.bar(volume_categories, volume_values, color=['green', 'yellow', 'orange', 'red'])
    ax5.set_title("Tumor Volume Category")
    ax5.set_ylabel("Risk Level")
    diagrams.append(fig5)
    
    # Cause-Prevention Relationship Diagram
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    causes = ['Hepatitis', 'Alcohol', 'NAFLD', 'Toxins', 'Genetics']
    prevention = ['Vaccination', 'Moderation', 'Diet/Exercise', 'Avoidance', 'Screening']
    relation_strength = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    y_pos = np.arange(len(causes))
    ax6.barh(y_pos, relation_strength, align='center', color='#4caf50')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f"{c}\n↓\n{p}" for c, p in zip(causes, prevention)])
    ax6.invert_yaxis()
    ax6.set_xlabel('Prevention Effectiveness')
    ax6.set_title('Cause-Prevention Relationship')
    diagrams.append(fig6)
    
    return diagrams

def predict_and_generate_report(image_path, name, national_id, nationality, age, mobile_number, gender, 
                              chronic_diseases, liver_enzymes, bilirubin, albumin, weight, height):
    img_array = preprocess_image(image_path)
    mask = model.predict(img_array)
    
    disease_info = determine_liver_disease_type(image_path, mask)
    visualization_buf = generate_segmentation_visualization(image_path, mask)
    
    try:
        bmi = float(weight) / ((float(height)/100) ** 2)
    except ValueError:
        bmi = 0
    
    try:
        clean_liver_enzymes = float(''.join(c for c in str(liver_enzymes) if c.isdigit() or c == '.'))
    except ValueError:
        clean_liver_enzymes = 0
    
    report = (f"Name: {name}\nNational ID: {national_id}\nNationality: {nationality}\nAge: {age}\n"
              f"Mobile Number: {mobile_number}\nGender: {gender}\nChronic Diseases: {chronic_diseases}\n"
              f"Liver Enzymes (ALT): {clean_liver_enzymes} IU/L\nBilirubin: {bilirubin} mg/dL\n"
              f"Albumin: {albumin} g/dL\nWeight: {weight} kg\nHeight: {height} cm\nBMI: {bmi:.1f}\n"
              f"Tumor Volume: {disease_info['volume']:.2f} mm³ (~{disease_info['volume']/1000:.1f} cm³)\n"
              f"Tumor Size Category: {disease_info['volume_category']}\n\n")
    
    report += f"Diagnosis: {disease_info['type']}\n\n"
    report += f"Description: {disease_info['description']}\n\n"
    
    report += "Potential Causes:\n"
    report += f"{disease_info['causes']}\n\n"
    
    report += "Recommended Treatment:\n"
    report += f"{disease_info['treatment']}\n\n"
    
    report += "Prevention Strategies:\n"
    report += f"{disease_info['prevention']}\n\n"
    
    if "No Tumor" in disease_info['type']:
        report += (f"{name} shows no signs of liver tumors in the scan. No further treatment is required. "
                   "However, if the patient has risk factors like hepatitis or alcohol use, "
                   "regular monitoring is recommended.\n\n")
    else:
        report += (f"{name} has a liver condition that requires medical attention. "
                   "Consultation with a hepatologist or oncologist is strongly recommended. "
                   "Depending on the diagnosis, further tests like biopsy or additional imaging "
                   "may be needed to confirm the diagnosis and plan treatment.\n\n")
        
        report += ("Additional Recommendations:\n"
                   "- Avoid alcohol completely\n"
                   "- Maintain healthy weight\n"
                   "- Get vaccinated against hepatitis A and B if not already immune\n"
                   "- Monitor liver function regularly\n"
                   "- Follow a liver-friendly diet (low fat, moderate protein)\n"
                   "- Stay hydrated\n"
                   "- Avoid unnecessary medications that can stress the liver\n\n")
    
    report += "\nRelationships between Input Data and Diagnosis:\n"
    
    if int(age) > 60:
        report += "- Age: Patients over 60 years old are at higher risk of liver cancer.\n"
    else:
        report += "- Age: Younger patients have a lower risk of liver cancer.\n"
    
    if gender == "Male":
        report += "- Gender: Males have higher risk of liver cancer compared to females.\n"
    else:
        report += "- Gender: Females have lower risk of liver cancer compared to males.\n"
    
    if chronic_diseases.lower() in ["hepatitis", "cirrhosis", "fatty liver"]:
        report += "- Chronic Diseases: Patients with chronic liver diseases are at higher risk of complications.\n"
    else:
        report += "- Chronic Diseases: No significant chronic liver diseases were reported.\n"
    
    if clean_liver_enzymes > 40:
        report += "- Liver Enzymes: Elevated liver enzymes may indicate liver damage or inflammation.\n"
    else:
        report += "- Liver Enzymes: Normal liver enzyme levels.\n"
    
    vol_cm3 = disease_info['volume'] / 1000
    if vol_cm3 < 1:
        report += "- Tumor Volume: Very small tumor (<1 cm³), lower risk.\n"
    elif 1 <= vol_cm3 < 5:
        report += "- Tumor Volume: Small tumor (1-5 cm³), moderate risk.\n"
    elif 5 <= vol_cm3 < 10:
        report += "- Tumor Volume: Medium tumor (5-10 cm³), high risk.\n"
    else:
        report += "- Tumor Volume: Large tumor (>10 cm³), very high risk.\n"
    
    if bmi > 30:
        report += "- BMI: Obesity is a risk factor for fatty liver disease and liver cancer.\n"
    elif bmi > 25:
        report += "- BMI: Overweight status may contribute to liver disease risk.\n"
    else:
        report += "- BMI: Healthy weight reduces liver disease risk.\n"
    
    diagrams = create_relationship_diagrams(age, gender, chronic_diseases, liver_enzymes, disease_info['volume'])
    
    return report, visualization_buf, disease_info, diagrams

@app.route('/', methods=['GET', 'POST'])
def index():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    if request.method == 'POST':
        name = request.form['name']
        national_id = request.form['national_id']
        nationality = request.form['nationality']
        age = request.form['age']
        mobile_number = request.form['mobile_number']
        gender = request.form['gender']
        chronic_diseases = request.form['chronic_diseases']
        liver_enzymes = request.form['liver_enzymes']
        bilirubin = request.form['bilirubin']
        albumin = request.form['albumin']
        weight = request.form['weight']
        height = request.form['height']
        
        if 'image' not in request.files:
            return "No image uploaded", 400
        image = request.files['image']
        if image.filename == '':
            return "No image selected", 400
        
        image_path = os.path.join('uploads', image.filename)
        image.save(image_path)
        
        report, visualization_buf, disease_info, diagrams = predict_and_generate_report(
            image_path, name, national_id, nationality, age, mobile_number, gender,
            chronic_diseases, liver_enzymes, bilirubin, albumin, weight, height)
        
        visualization_path = os.path.join('static', 'visualization.png')
        with open(visualization_path, 'wb') as f:
            f.write(visualization_buf.getbuffer())
        
        diagram_paths = []
        for i, diagram in enumerate(diagrams):
            buf = io.BytesIO()
            diagram.savefig(buf, format='png')
            buf.seek(0)
            path = os.path.join('static', f'diagram_{i}.png')
            with open(path, 'wb') as f:
                f.write(buf.getbuffer())
            diagram_paths.append(path)
            plt.close(diagram)
        
        return render_template('result.html', 
                             report=report, 
                             visualization=visualization_path,
                             diagrams=diagram_paths,
                             disease_info=disease_info,
                             age=age,
                             gender=gender,
                             chronic_diseases=chronic_diseases,
                             liver_enzymes=liver_enzymes)
    
    return render_template('index.html')

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        report = request.form['report']
        visualization_path = request.form['visualization']
        age = request.form['age']
        gender = request.form['gender']
        chronic_diseases = request.form['chronic_diseases']
        liver_enzymes = request.form['liver_enzymes']
        
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add title page
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, "Liver Tumor Detection Report", ln=True, align='C')
        pdf.ln(20)
        
        # Add patient information table
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Patient Information:", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(60, 7, "Field", border=1)
        pdf.cell(0, 7, "Value", border=1, ln=True)
        
        pdf.set_font("Arial", size=10)
        lines = report.split("\n")
        
        patient_data = {
            "Name": lines[0].split(": ")[1],
            "National ID": lines[1].split(": ")[1],
            "Nationality": lines[2].split(": ")[1],
            "Age": lines[3].split(": ")[1],
            "Mobile Number": lines[4].split(": ")[1],
            "Gender": lines[5].split(": ")[1],
            "Chronic Diseases": lines[6].split(": ")[1],
            "Liver Enzymes (ALT)": lines[7].split(": ")[1],
            "Bilirubin": lines[8].split(": ")[1],
            "Albumin": lines[9].split(": ")[1],
            "Weight": lines[10].split(": ")[1],
            "Height": lines[11].split(": ")[1],
            "BMI": lines[12].split(": ")[1]
        }
        
        for field, value in patient_data.items():
            pdf.cell(60, 7, field, border=1)
            pdf.cell(0, 7, value, border=1, ln=True)
        
        pdf.ln(10)
        
        # Add diagnosis section
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Diagnosis:", ln=True)
        pdf.set_font("Arial", size=10)
        
        diagnosis_start = next((i for i, line in enumerate(lines) if "Diagnosis:" in line), len(lines))
        risk_start = next((i for i, line in enumerate(lines) if "Relationships between Input Data and Diagnosis:" in line), len(lines))
        
        for line in lines[diagnosis_start:risk_start]:
            if line.strip():
                if "Diagnosis:" in line or "Description:" in line or "Recommended Treatment:" in line:
                    pdf.set_font("Arial", style='B', size=11)
                    pdf.cell(0, 7, line, ln=True)
                    pdf.set_font("Arial", size=10)
                else:
                    pdf.multi_cell(0, 7, line)
        
        pdf.ln(5)
        
        # Add tumor volume info
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Tumor Volume Analysis:", ln=True)
        pdf.set_font("Arial", size=10)
        
        tumor_volume_line = next((line for line in lines if "Tumor Volume:" in line), "")
        tumor_category_line = next((line for line in lines if "Tumor Size Category:" in line), "")
        
        pdf.cell(0, 7, tumor_volume_line, ln=True)
        pdf.cell(0, 7, tumor_category_line, ln=True)
        pdf.ln(5)
        
        # Add causes and prevention
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Potential Causes:", ln=True)
        pdf.set_font("Arial", size=10)
        
        causes = next((line.split("Potential Causes:")[1].strip() for line in lines if "Potential Causes:" in line), "")
        pdf.multi_cell(0, 7, causes)
        pdf.ln(5)
        
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Prevention Strategies:", ln=True)
        pdf.set_font("Arial", size=10)
        
        prevention = next((line.split("Prevention Strategies:")[1].strip() for line in lines if "Prevention Strategies:" in line), "")
        pdf.multi_cell(0, 7, prevention)
        pdf.ln(5)
        
        # Add risk factors
        if risk_start < len(lines):
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(0, 10, "Risk Factors Analysis:", ln=True)
            pdf.set_font("Arial", size=10)
            
            for line in lines[risk_start:]:
                if line.strip():
                    if line.startswith("-"):
                        pdf.cell(10)
                        pdf.cell(0, 7, line[2:], ln=True)
                    else:
                        pdf.cell(0, 7, line, ln=True)
            pdf.ln(5)
        
        # Add visualization page
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Liver Scan Analysis:", ln=True)
        pdf.image(visualization_path, x=10, y=None, w=180)
        
        pdf_path = os.path.join(static_dir, 'liver_report.pdf')
        pdf.output(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Failed to create PDF file at {pdf_path}")
        
        return send_file(pdf_path, as_attachment=True)
    
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return f"An error occurred while generating the PDF: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)