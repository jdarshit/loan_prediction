from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import joblib
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from functools import wraps
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Handle production database (Render uses DATABASE_URL)
database_url = os.getenv('DATABASE_URL')
if database_url:
    # Render provides postgresql, convert to compatible format
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_predictions.db'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize database on app startup
with app.app_context():
    try:
        db.create_all()
        print("✓ Database tables created/verified")
    except Exception as e:
        print(f"Database initialization warning: {e}")

# Load model and scaler
import os
model_path = os.path.join(os.path.dirname(__file__), "final_xgboost_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "final_scaler.pkl")
feature_path = os.path.join(os.path.dirname(__file__), "final_feature_names.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_path)
    print("✓ Using Final XGBoost Model (85.37% accuracy) - final_xgboost_model.pkl")
except FileNotFoundError as e:
    print(f"⚠️ Model file not found: {e}")
    print("Model files must be present for deployment")
    raise

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    prediction_result = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    approved_prob = db.Column(db.Float)
    rejected_prob = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def preprocess_input(data_dict):
    """Preprocess input data with engineered features for XGBoost"""
    required_fields = {
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Age', 'Property_Area_Urban', 'Property_Area_Semiurban'
    }
    
    missing_fields = required_fields - set(data_dict.keys())
    if missing_fields:
        raise ValueError(f"Missing fields: {', '.join(missing_fields)}")
    
    try:
        # Calculate engineered features
        applicant_income = float(data_dict['ApplicantIncome'])
        coapplicant_income = float(data_dict['CoapplicantIncome'])
        loan_amount = float(data_dict['LoanAmount'])
        loan_term = float(data_dict['Loan_Amount_Term'])
        
        # Feature Engineering
        total_income = applicant_income + coapplicant_income
        income_loan_ratio = total_income / (loan_amount + 1)
        
        # EMI Calculation
        interest_rate = 0.10/12  # 10% annual rate
        emi = (loan_amount * 1000 * interest_rate * 
               np.power(1 + interest_rate, loan_term)) / \
              (np.power(1 + interest_rate, loan_term) - 1)
        emi_income_ratio = emi / (total_income/12 + 1)
        
        # Create DataFrame with features in correct order
        input_data = pd.DataFrame([[
            int(data_dict['Gender']),
            int(data_dict['Married']),
            int(data_dict['Dependents']),
            int(data_dict['Education']),
            int(data_dict['Self_Employed']),
            loan_amount,
            loan_term,
            float(data_dict['Credit_History']),
            float(data_dict['Age']),
            total_income,
            income_loan_ratio,
            emi_income_ratio,
            int(data_dict['Property_Area_Semiurban']),
            int(data_dict['Property_Area_Urban'])
        ]], columns=[
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Age',
            'TotalIncome', 'Income_Loan_Ratio', 'EMI_Income_Ratio',
            'Property_Area_Semiurban', 'Property_Area_Urban'
        ])
        
        # Scale features using the saved scaler
        try:
            input_data_scaled = scaler.transform(input_data)
            input_data = pd.DataFrame(input_data_scaled, columns=input_data.columns)
        except:
            pass  # Use unscaled if scaler not available
        return input_data
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid data type: {str(e)}")

def make_prediction(input_data):
    """Make prediction"""
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    status = "Approved" if pred == 1 else "Rejected"
    confidence = max(prob) * 100
    
    return {
        'prediction': status,
        'confidence': float(round(float(confidence), 2)),
        'prob_rejected': float(round(float(prob[0]) * 100, 2)),
        'prob_approved': float(round(float(prob[1]) * 100, 2))
    }

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Main Routes
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    prediction = None
    confidence = None
    prob_rejected = None
    prob_approved = None
    error_message = None

    if request.method == 'POST':
        try:
            input_data_dict = request.form.to_dict()
            # Convert single Property_Area dropdown to binary fields
            area = input_data_dict.get('Property_Area', 'Rural')
            input_data_dict['Property_Area_Urban']     = '1' if area == 'Urban'     else '0'
            input_data_dict['Property_Area_Semiurban'] = '1' if area == 'Semiurban' else '0'
            input_data = preprocess_input(input_data_dict)
            result = make_prediction(input_data)
            
            # Save to database
            pred = Prediction(
                user_id=current_user.id,
                prediction_result=result['prediction'],
                confidence=result['confidence'],
                input_data=input_data_dict,
                approved_prob=result['prob_approved'],
                rejected_prob=result['prob_rejected']
            )
            db.session.add(pred)
            db.session.commit()
            
            prediction = result['prediction']
            confidence = result['confidence']
            prob_rejected = result['prob_rejected']
            prob_approved = result['prob_approved']
            
        except (ValueError, KeyError) as e:
            error_message = f"Invalid input: {str(e)}"
        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template('index.html', 
                         prediction=prediction,
                         confidence=confidence,
                         prob_rejected=prob_rejected,
                         prob_approved=prob_approved,
                         error_message=error_message)

@app.route('/history')
@login_required
def history():
    search = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    
    query = Prediction.query.filter_by(user_id=current_user.id)
    
    if search == 'approved':
        query = query.filter_by(prediction_result='Approved')
    elif search == 'rejected':
        query = query.filter_by(prediction_result='Rejected')
    
    predictions = query.order_by(Prediction.timestamp.desc()).paginate(page=page, per_page=20)
    
    return render_template('history.html', predictions=predictions, search=search)

@app.route('/delete_prediction/<int:pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    prediction = db.get_or_404(Prediction, pred_id)
    if prediction.user_id != current_user.id:
        return "Unauthorized", 403
    db.session.delete(prediction)
    db.session.commit()
    return redirect(url_for('history'))

@app.route('/download_template')
@login_required
def download_template():
    import csv, io as _io
    output = _io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Gender','Married','Dependents','Education','Self_Employed',
                     'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
                     'Credit_History','Age','Property_Area'])
    writer.writerow([1, 1, 0, 1, 0, 5000, 1500, 130, 360, 1, 30, 'Urban'])
    writer.writerow([0, 0, 1, 0, 1, 3000, 0,    100, 180, 0, 45, 'Rural'])
    writer.writerow([1, 1, 2, 1, 0, 8000, 2000, 200, 360, 1, 35, 'Semiurban'])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='loan_prediction_template.csv'
    )

@app.route('/download_report/<int:pred_id>')
@login_required
def download_report(pred_id):
    prediction = db.get_or_404(Prediction, pred_id)
    
    if prediction.user_id != current_user.id:
        return "Unauthorized", 403
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        elements = []
        
        styles = getSampleStyleSheet()
        result_color = colors.HexColor('#155724') if prediction.prediction_result == 'Approved' else colors.HexColor('#721c24')
        
        # Title
        title = Paragraph("Loan Prediction Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Result badge
        result_style = ParagraphStyle('result', parent=styles['Normal'],
                                       fontSize=16, fontName='Helvetica-Bold',
                                       textColor=result_color)
        elements.append(Paragraph(f"Result: {prediction.prediction_result}", result_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary table
        summary_data = [
            ['Field', 'Value'],
            ['Date', prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Result', prediction.prediction_result],
            ['Confidence', f"{prediction.confidence:.2f}%"],
            ['Approval Probability', f"{prediction.approved_prob:.2f}%"],
            ['Rejection Probability', f"{prediction.rejected_prob:.2f}%"],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Input data section
        if prediction.input_data:
            elements.append(Paragraph("Input Details", styles['Heading2']))
            elements.append(Spacer(1, 0.1*inch))
            
            inp = prediction.input_data
            
            def get_val(key):
                v = inp.get(key)
                if v is None:
                    return None
                if isinstance(v, list):
                    v = v[0] if v else None
                return str(v).strip() if v is not None else None
            
            # Determine property area
            u = get_val('Property_Area_Urban')
            s = get_val('Property_Area_Semiurban')
            if u in ('1', '1.0', 'True'):
                area = 'Urban'
            elif s in ('1', '1.0', 'True'):
                area = 'Semiurban'
            else:
                area = 'Rural'
            
            # Human-readable mappings
            gender_map   = {'1': 'Male',        '0': 'Female'}
            married_map  = {'1': 'Yes',          '0': 'No'}
            edu_map      = {'1': 'Graduate',     '0': 'Not Graduate'}
            emp_map      = {'1': 'Yes',          '0': 'No'}
            credit_map   = {'1': 'Good (1)',     '0': 'Bad (0)'}
            dep_map      = {'3': '3+'}
            
            # Loan amount: stored in thousands, show actual value
            loan_raw = get_val('LoanAmount')
            try:
                loan_display = f"Rs. {int(float(loan_raw) * 1000):,}" if loan_raw else 'N/A'
            except Exception:
                loan_display = loan_raw or 'N/A'
            
            # Incomes
            def fmt_income(key):
                v = get_val(key)
                try:
                    return f"Rs. {int(float(v)):,}" if v else 'N/A'
                except Exception:
                    return v or 'N/A'
            
            age_val = get_val('Age')
            age_display = (age_val + ' years') if age_val and age_val not in ('', 'None') else 'Not provided'
            
            term_raw = get_val('Loan_Amount_Term')
            try:
                term_display = f"{int(float(term_raw))} months" if term_raw else 'N/A'
            except Exception:
                term_display = term_raw or 'N/A'
            
            input_rows = [['Field', 'Value'],
                ['Gender',            gender_map.get(get_val('Gender') or '', get_val('Gender') or 'N/A')],
                ['Married',           married_map.get(get_val('Married') or '', get_val('Married') or 'N/A')],
                ['Dependents',        dep_map.get(get_val('Dependents') or '', get_val('Dependents') or 'N/A')],
                ['Education',         edu_map.get(get_val('Education') or '', get_val('Education') or 'N/A')],
                ['Self Employed',     emp_map.get(get_val('Self_Employed') or '', get_val('Self_Employed') or 'N/A')],
                ['Applicant Income',  fmt_income('ApplicantIncome')],
                ['Co-applicant Income', fmt_income('CoapplicantIncome')],
                ['Loan Amount',       loan_display],
                ['Loan Term',         term_display],
                ['Credit History',    credit_map.get(get_val('Credit_History') or '', get_val('Credit_History') or 'N/A')],
                ['Age',               age_display],
                ['Property Area',     area],
            ]
            
            input_table = Table(input_rows, colWidths=[2.5*inch, 3.5*inch])
            input_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ]))
            elements.append(input_table)
        
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("Generated by Loan Prediction System", styles['Normal']))
        
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                        download_name=f'loan_report_{pred_id}.pdf')
    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        return "Access Denied", 403
    
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    approved_count = Prediction.query.filter_by(prediction_result='Approved').count()
    rejected_count = Prediction.query.filter_by(prediction_result='Rejected').count()
    
    avg_confidence = db.session.query(db.func.avg(Prediction.confidence)).scalar() or 0
    
    stats = {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'approved': approved_count,
        'rejected': rejected_count,
        'avg_confidence': round(avg_confidence, 2)
    }
    
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/metrics', methods=['GET'])
def metrics():
    if current_user.is_authenticated:
        user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    else:
        user_predictions = []
    
    metrics_data = {
        'accuracy': 0.8537,
        'precision': 0.8317,
        'recall': 0.9882,
        'f1_score': 0.9032,
        'roc_auc': 0.8771,
        'total_predictions': len(user_predictions),
        'approved_count': sum(1 for p in user_predictions if p.prediction_result == 'Approved'),
        'rejected_count': sum(1 for p in user_predictions if p.prediction_result == 'Rejected'),
        'avg_confidence': round(sum(p.confidence for p in user_predictions) / len(user_predictions), 2) if user_predictions else 0
    }
    
    return render_template('metrics.html', metrics=metrics_data)

@app.route('/batch_predict', methods=['GET', 'POST'])
@login_required
def batch_predict():
    error_message = None
    results = None
    
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                error_message = "No file uploaded"
            else:
                file = request.files['file']
                if not file.filename.endswith('.csv'):
                    error_message = "Please upload a CSV file"
                else:
                    df = pd.read_csv(file)
                    # Support both Property_Area (new) and binary columns (legacy)
                    if 'Property_Area' in df.columns:
                        df['Property_Area_Urban']     = (df['Property_Area'] == 'Urban').astype(int)
                        df['Property_Area_Semiurban'] = (df['Property_Area'] == 'Semiurban').astype(int)
                    required_cols = {
                        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                        'Credit_History', 'Age', 'Property_Area_Urban', 'Property_Area_Semiurban'
                    }
                    missing = required_cols - set(df.columns)
                    if missing:
                        error_message = f"CSV missing columns: {', '.join(sorted(missing))}"
                    else:
                        predictions = []
                        for idx, row in df.iterrows():
                            try:
                                input_data = preprocess_input(row.to_dict())
                                result = make_prediction(input_data)
                                pred = Prediction(
                                    user_id=current_user.id,
                                    prediction_result=result['prediction'],
                                    confidence=result['confidence'],
                                    input_data=row.to_dict(),
                                    approved_prob=result['prob_approved'],
                                    rejected_prob=result['prob_rejected']
                                )
                                predictions.append({
                                    'row': idx + 1,
                                    'prediction': result['prediction'],
                                    'confidence': f"{result['confidence']:.2f}",
                                    'prob_approved': f"{result['prob_approved']:.2f}",
                                    'prob_rejected': f"{result['prob_rejected']:.2f}"
                                })
                                db.session.add(pred)
                            except Exception as e:
                                predictions.append({'row': idx + 1, 'error': str(e)})
                        db.session.commit()
                        results = predictions
        except Exception as e:
            error_message = f"Error: {str(e)}"
    
    return render_template('batch_predict.html', error_message=error_message, results=results)

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """JSON API endpoint for predictions"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    try:
        # Handle Property_Area
        area = data.get('Property_Area', None)
        if area:
            data['Property_Area_Urban']     = 1 if area == 'Urban'     else 0
            data['Property_Area_Semiurban'] = 1 if area == 'Semiurban' else 0
        input_data = preprocess_input(data)
        result = make_prediction(input_data)
        return jsonify({'status': 'success', **result})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """JSON API endpoint for prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                  .order_by(Prediction.timestamp.desc()).limit(20).all()
    return jsonify({
        'total_predictions': len(predictions),
        'recent': [
            {
                'id': p.id,
                'result': p.prediction_result,
                'confidence': p.confidence,
                'timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            } for p in predictions
        ]
    })

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # Production: debug=False, Development: debug=True
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
