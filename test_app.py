"""
Unit tests for Loan Prediction Application
Run with: pytest test_app.py -v
"""

import pytest
import json
from app import app as flask_app, preprocess_input, make_prediction, db, User
import pandas as pd


@pytest.fixture(scope='session')
def app():
    flask_app.config['TESTING'] = True
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    flask_app.config['WTF_CSRF_ENABLED'] = False
    with flask_app.app_context():
        db.create_all()
        if not User.query.filter_by(username='testuser').first():
            u = User(username='testuser', email='test@test.com')
            u.set_password('testpass')
            db.session.add(u)
            db.session.commit()
    yield flask_app


@pytest.fixture
def client(app):
    """Unauthenticated test client"""
    return app.test_client()


@pytest.fixture
def auth_client(app):
    """Authenticated test client"""
    c = app.test_client()
    c.post('/login', data={'username': 'testuser', 'password': 'testpass'})
    return c


@pytest.fixture
def valid_input():
    """Valid sample input data"""
    return {
        'Gender': 1,
        'Married': 1,
        'Dependents': 0,
        'Education': 1,
        'Self_Employed': 0,
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 0,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Age': 35,
        'Property_Area_Urban': 1,
        'Property_Area_Semiurban': 0
    }


class TestPreprocessing:
    """Test input preprocessing and validation"""
    
    def test_valid_input(self, valid_input):
        """Test valid input preprocessing"""
        result = preprocess_input(valid_input)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 14)
        assert list(result.columns) == [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Age',
            'TotalIncome', 'Income_Loan_Ratio', 'EMI_Income_Ratio',
            'Property_Area_Semiurban', 'Property_Area_Urban'
        ]
    
    def test_missing_field(self, valid_input):
        """Test error handling for missing fields"""
        del valid_input['Gender']
        with pytest.raises(ValueError, match="Missing fields"):
            preprocess_input(valid_input)
    
    def test_invalid_type(self, valid_input):
        """Test error handling for invalid data types"""
        valid_input['Gender'] = 'invalid'
        with pytest.raises(ValueError, match="Invalid data type"):
            preprocess_input(valid_input)
    
    def test_negative_income(self, valid_input):
        """Test handling of negative income - preprocess still builds TotalIncome"""
        valid_input['ApplicantIncome'] = -5000
        result = preprocess_input(valid_input)
        # Scaler transforms the value; just verify column exists and is numeric
        assert 'TotalIncome' in result.columns
        assert pd.api.types.is_numeric_dtype(result['TotalIncome'])

    def test_float_conversion(self, valid_input):
        """Test string income gets converted to numeric in TotalIncome"""
        valid_input['ApplicantIncome'] = '5000'
        result = preprocess_input(valid_input)
        assert isinstance(result['TotalIncome'].values[0], (int, float))


class TestPrediction:
    """Test prediction functionality"""
    
    def test_prediction_output_format(self, valid_input):
        """Test prediction returns correct format"""
        input_data = preprocess_input(valid_input)
        result = make_prediction(input_data)
        
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'prob_approved' in result
        assert 'prob_rejected' in result
    
    def test_prediction_values_valid(self, valid_input):
        """Test prediction values are valid"""
        input_data = preprocess_input(valid_input)
        result = make_prediction(input_data)
        
        assert result['prediction'] in ['Approved', 'Rejected']
        assert 0 <= result['confidence'] <= 100
        assert 0 <= result['prob_approved'] <= 100
        assert 0 <= result['prob_rejected'] <= 100
        # Probabilities should sum to 100
        assert abs((result['prob_approved'] + result['prob_rejected']) - 100) < 0.01
    
    def test_different_inputs_different_predictions(self):
        """Test that different inputs can produce different predictions"""
        input1 = preprocess_input({
            'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': 1,
            'Self_Employed': 0, 'ApplicantIncome': 10000, 'CoapplicantIncome': 0,
            'LoanAmount': 500, 'Loan_Amount_Term': 360, 'Credit_History': 1,
            'Age': 30,
            'Property_Area_Urban': 1, 'Property_Area_Semiurban': 0
        })

        input2 = preprocess_input({
            'Gender': 0, 'Married': 0, 'Dependents': 3, 'Education': 0,
            'Self_Employed': 1, 'ApplicantIncome': 500, 'CoapplicantIncome': 0,
            'LoanAmount': 50, 'Loan_Amount_Term': 360, 'Credit_History': 0,
            'Age': 55,
            'Property_Area_Urban': 0, 'Property_Area_Semiurban': 1
        })
        
        pred1 = make_prediction(input1)
        pred2 = make_prediction(input2)
        
        # At least one should be different
        assert pred1['prediction'] != pred2['prediction'] or \
               abs(pred1['confidence'] - pred2['confidence']) > 5


class TestWebRoutes:
    """Test Flask web routes"""

    def test_home_page_redirects_when_not_logged_in(self, client):
        """Unauthenticated GET to home redirects to login"""
        response = client.get('/')
        assert response.status_code == 302

    def test_home_page_get(self, auth_client):
        """Authenticated GET to home page returns 200"""
        response = auth_client.get('/')
        assert response.status_code == 200
        assert b'Loan Prediction Form' in response.data

    def test_home_page_post_valid(self, auth_client, valid_input):
        """POST with valid data returns prediction"""
        valid_input['Property_Area'] = 'Urban'
        del valid_input['Property_Area_Urban']
        del valid_input['Property_Area_Semiurban']
        response = auth_client.post('/', data=valid_input)
        assert response.status_code == 200
        assert b'Approved' in response.data or b'Rejected' in response.data

    def test_home_page_post_invalid(self, auth_client, valid_input):
        """POST with missing field shows error"""
        del valid_input['Gender']
        valid_input['Property_Area'] = 'Urban'
        del valid_input['Property_Area_Urban']
        del valid_input['Property_Area_Semiurban']
        response = auth_client.post('/', data=valid_input)
        assert response.status_code == 200
        assert b'Error' in response.data or b'error' in response.data

    def test_batch_predict_page(self, auth_client):
        """Batch prediction page loads for logged-in user"""
        response = auth_client.get('/batch_predict')
        assert response.status_code == 200
        assert b'Batch Prediction' in response.data

    def test_metrics_page(self, client):
        """Metrics page is publicly accessible"""
        response = client.get('/metrics')
        assert response.status_code == 200
        assert b'Model Performance' in response.data

    def test_download_template(self, auth_client):
        """CSV template download works"""
        response = auth_client.get('/download_template')
        assert response.status_code == 200
        assert b'Gender' in response.data
        assert b'Married' in response.data


class TestAPI:
    """Test JSON API endpoints"""

    def test_api_predict_valid(self, auth_client, valid_input):
        """API prediction with valid data returns success"""
        valid_input['Property_Area'] = 'Urban'
        del valid_input['Property_Area_Urban']
        del valid_input['Property_Area_Semiurban']
        response = auth_client.post(
            '/api/predict',
            data=json.dumps(valid_input),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['prediction'] in ['Approved', 'Rejected']
        assert 'confidence' in data

    def test_api_predict_invalid(self, auth_client):
        """API with missing fields returns 400"""
        response = auth_client.post(
            '/api/predict',
            data=json.dumps({'invalid': 'data'}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_api_predict_no_json(self, auth_client):
        """API without JSON content-type returns 400"""
        response = auth_client.post('/api/predict')
        assert response.status_code == 400

    def test_api_history(self, auth_client):
        """API history endpoint returns valid structure"""
        response = auth_client.get('/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'total_predictions' in data
        assert 'recent' in data


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_income(self):
        """Test with zero income — model handles gracefully"""
        input_data = preprocess_input({
            'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': 1,
            'Self_Employed': 0, 'ApplicantIncome': 0, 'CoapplicantIncome': 0,
            'LoanAmount': 100, 'Loan_Amount_Term': 360, 'Credit_History': 1,
            'Age': 30,
            'Property_Area_Urban': 1, 'Property_Area_Semiurban': 0
        })
        result = make_prediction(input_data)
        assert result['prediction'] in ['Approved', 'Rejected']

    def test_very_high_loan_amount(self):
        """Test with very high loan amount"""
        input_data = preprocess_input({
            'Gender': 1, 'Married': 1, 'Dependents': 0, 'Education': 1,
            'Self_Employed': 0, 'ApplicantIncome': 1000000, 'CoapplicantIncome': 500000,
            'LoanAmount': 500000, 'Loan_Amount_Term': 360, 'Credit_History': 1,
            'Age': 40,
            'Property_Area_Urban': 1, 'Property_Area_Semiurban': 0
        })
        result = make_prediction(input_data)
        assert result['prediction'] in ['Approved', 'Rejected']

    def test_all_zeros(self):
        """Test with all minimum values"""
        input_data = preprocess_input({
            'Gender': 0, 'Married': 0, 'Dependents': 0, 'Education': 0,
            'Self_Employed': 0, 'ApplicantIncome': 1, 'CoapplicantIncome': 0,
            'LoanAmount': 1, 'Loan_Amount_Term': 1, 'Credit_History': 0,
            'Age': 18,
            'Property_Area_Urban': 0, 'Property_Area_Semiurban': 0
        })
        result = make_prediction(input_data)
        assert result['prediction'] in ['Approved', 'Rejected']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
