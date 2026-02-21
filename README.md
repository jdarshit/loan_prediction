# Loan Prediction System - Complete Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Documentation](#api-documentation)
6. [Model Performance](#model-performance)
7. [Deployment](#deployment)
8. [Contributing](#contributing)
9. [License](#license)

---

## 📊 Project Overview

The **Loan Prediction System** is a machine learning application that predicts loan approval status based on applicant characteristics. It combines a tuned Logistic Regression model with a user-friendly Flask web interface.

**Key Stats:**
- **Model Accuracy**: 78.86%
- **ROC-AUC Score**: 0.7701
- **F1-Score**: 0.8587
- **Training Data**: 614 loan applications

---

## ✨ Features

### Single Prediction
- **Web Form**: Intuitive interface for individual loan predictions
- **Instant Results**: Get approval status with confidence scores
- **Probability Display**: See approved/rejected probability percentages
- **Error Handling**: User-friendly error messages

### Batch Prediction  
- **CSV Upload**: Process multiple loan applications at once
- **Template Download**: Pre-formatted CSV template provided
- **Detailed Results**: Row-by-row prediction results
- **Statistics**: Summary statistics from batch processing

### Model Performance Dashboard
- **Key Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Prediction Stats**: Total predictions, approval rate, confidence scores
- **Model Info**: Algorithm details and parameters

### API Endpoints
- **REST API**: Programmatic access to predictions
- **JSON Format**: Easy integration with other systems
- **Historical Data**: Track predictions over time
- **CSV Download**: Export results

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip package manager
- Git (for version control)

### Local Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd loan-prediction

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py

# 6. Open browser to
http://127.0.0.1:5000/
```

### Docker Installation (Optional)

```bash
# Build Docker image
docker build -t loan-prediction .

# Run container
docker run -p 5000:5000 loan-prediction
```

---

## 💻 Usage

### Web Application

**Single Prediction:**
1. Navigate to the homepage
2. Fill in the loan application form
3. Click "Predict"
4. View instant results with confidence score

**Batch Prediction:**
1. Go to "Batch Upload" tab
2. Download CSV template
3. Fill in your data
4. Upload file
5. View results table with statistics

**View Metrics:**
1. Click "Metrics" tab
2. See real-time model performance
3. Track prediction statistics

### Command Line

```bash
# Run Flask development server
python app.py

# Run with specific host/port
python app.py --host 0.0.0.0 --port 8000

# Production mode
export FLASK_ENV=production
gunicorn app:app
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Single Prediction
**POST** `/api/predict`

**Request:**
```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 0,
  "Education": 1,
  "Self_Employed": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area_Urban": 1,
  "Property_Area_Semiurban": 0
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Approved",
  "confidence": 86.23,
  "prob_approved": 86.23,
  "prob_rejected": 13.77
}
```

**Error Response:**
```json
{
  "error": "Invalid data type: ..."
}
```

#### 2. Get History
**GET** `/api/history`

**Response:**
```json
{
  "total_predictions": 25,
  "recent": [
    {
      "timestamp": "2024-02-14 10:30:45",
      "prediction": "Approved",
      "confidence": 86.23
    }
  ]
}
```

#### 3. Download Template
**GET** `/download_template`

Returns CSV file with sample data.

---

## 📈 Model Performance

### Algorithm
- **Type**: Tuned Logistic Regression
- **Best Parameters**: C=1, Solver=lbfgs
- **Cross-Validation**: 5-fold

### Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 78.86% |
| Precision | 75.96% |
| Recall | 98.75% |
| F1-Score | 85.87% |
| ROC-AUC | 77.01% |

### Feature Importance

The model uses 12 features:
1. Gender (1=Male, 0=Female)
2. Married Status (1=Yes, 0=No)
3. Number of Dependents (0, 1, 2, 3)
4. Education (1=Graduate, 0=Not)
5. Self Employment (1=Yes, 0=No)
6. Applicant Income
7. Coapplicant Income
8. Loan Amount
9. Loan Term
10. Credit History (1=Good, 0=Bad)
11. Property Urban (1=Yes, 0=No)
12. Property Semiurban (1=Yes, 0=No)

---

## 📁 Project Structure

```
loan-prediction/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── requirements_production.txt      # Production dependencies
├── Procfile                        # Heroku deployment
├── runtime.txt                     # Python version
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── DEPLOYMENT.md                   # Deployment guide
├── FIXES_APPLIED.md               # Bug fix history
├── rf_loan_model.pkl              # Original trained model
├── tuned_loan_model.pkl           # Tuned model (recommended)
├── data/
│   ├── raw/
│   │   └── train_u6lujuX_CVtuZ9i.csv
│   └── processed/
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   └── model_comparison.ipynb      # Model comparison & tuning
├── templates/
│   ├── index.html                 # Single prediction page
│   ├── batch_predict.html         # Batch upload page
│   └── metrics.html               # Metrics dashboard
├── static/                        # CSS, JS, images (optional)
└── tests/
    ├── test_app.py                # Unit tests
    ├── test_preprocessing.py       # Preprocessing tests
    └── test_predictions.py         # Prediction tests
```

---

## 🧪 Testing

### Unit Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_app.py -v
```

Run with coverage:
```bash
pytest --cov=. tests/
```

### Test Coverage

- Web routes (GET/POST)
- Input validation
- Prediction accuracy
- Error handling
- Batch processing

---

## 🌍 Deployment

### Quick Deploy to Heroku

```bash
# Login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment guides.

---

## 🔧 Configuration

### Environment Variables

```bash
FLASK_ENV=production          # Set to 'production' for production
FLASK_DEBUG=False             # Disable debug mode
FLASK_APP=app.py              # Main app file
SECRET_KEY=your-secret-key    # Flask secret key
PORT=5000                     # Server port
```

### Model Configuration

Edit `app.py` to switch models:

```python
# Use tuned model (recommended)
model = joblib.load("tuned_loan_model.pkl")

# Or use original model
model = joblib.load("rf_loan_model.pkl")
```

---

## 📚 Input Validation

All inputs are validated before prediction:

| Field | Type | Range | Required |
|-------|------|-------|----------|
| Gender | Integer | 0-1 | Yes |
| Married | Integer | 0-1 | Yes |
| Dependents | Integer | 0-3 | Yes |
| Education | Integer | 0-1 | Yes |
| Self_Employed | Integer | 0-1 | Yes |
| ApplicantIncome | Float | >0 | Yes |
| CoapplicantIncome | Float | >=0 | Yes |
| LoanAmount | Float | >0 | Yes |
| Loan_Amount_Term | Float | >0 | Yes |
| Credit_History | Integer | 0-1 | Yes |
| Property_Area_Urban | Integer | 0-1 | Yes |
| Property_Area_Semiurban | Integer | 0-1 | Yes |

---

## 🐛 Troubleshooting

### Issue: Model not found
**Solution**: Ensure `tuned_loan_model.pkl` is in project root directory.

### Issue: Port already in use
**Solution**: Change port in `app.py` or use:
```bash
python app.py --port 8000
```

### Issue: CSV upload fails
**Solution**: Ensure CSV has all required columns. Download template for reference.

### Issue: Predictions always same
**Solution**: Check model file is loaded correctly. Try:
```python
import joblib
model = joblib.load("tuned_loan_model.pkl")
print(model.get_params())  # Verify model loaded
```

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs: `python app.py` (development mode)
3. Check input data format
4. Verify model file exists

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset: Kaggle Loan Prediction Dataset
- Framework: Flask
- ML Library: scikit-learn
- Model: Logistic Regression (tuned with GridSearchCV)

---

## 📈 Future Improvements

- [ ] Add user authentication
- [ ] Implement database for predictions
- [ ] Create admin dashboard
- [ ] Add more model options
- [ ] Implement caching for batch predictions
- [ ] Add data visualization charts
- [ ] Mobile app version
- [ ] Real-time model monitoring

---

**Last Updated**: February 14, 2024
**Version**: 2.0.0 (Enhanced)
