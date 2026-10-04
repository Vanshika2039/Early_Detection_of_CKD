
"""
CKD Early Detection System — Flask Backend
==========================================
Run:  python app_code.py
App:  http://127.0.0.1:5000
"""

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import joblib
import os

# ── App Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
app = Flask(__name__)
CORS(app)

# ── Database Config (SQLite) ───────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, 'ckd_users.db')
app.config['SQLALCHEMY_DATABASE_URI']        = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

db = SQLAlchemy(app)


# ── Database Models ────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id          = db.Column(db.Integer,     primary_key=True)
    username    = db.Column(db.String(80),  unique=True, nullable=False)
    email       = db.Column(db.String(120), unique=True, nullable=False)
    name        = db.Column(db.String(120), nullable=False)
    phone       = db.Column(db.String(20),  nullable=False)
    password    = db.Column(db.String(256), nullable=False)
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    __tablename__   = 'predictions'
    id              = db.Column(db.Integer,    primary_key=True)
    user_id         = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=False)
    result          = db.Column(db.String(20), nullable=False)
    risk_level      = db.Column(db.String(10), nullable=False)
    ckd_probability = db.Column(db.Float,      nullable=False)
    created_at      = db.Column(db.DateTime,   default=datetime.utcnow)


with app.app_context():
    db.create_all()
    print(f"✅ Database ready: {DB_PATH}")


# ── Load ML Model ──────────────────────────────────────────────────────────────
model    = joblib.load(os.path.join(BASE_DIR, 'ckd_model.pkl'))
scaler   = joblib.load(os.path.join(BASE_DIR, 'ckd_scaler.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'ckd_features.pkl'))
print(f"✅ Model loaded:  {type(model).__name__}")
print(f"✅ Features:      {features}")


# ── Symptom → Clinical Mapping ─────────────────────────────────────────────────
def symptom_to_clinical(u):
    age   = float(u.get('age', 40))
    htn   = 1 if u.get('hypertension')  == 'yes' else 0
    dm    = 1 if u.get('diabetes')      == 'yes' else 0
    smoke = u.get('smoke_alcohol')      == 'yes'
    tired = u.get('fatigue')            == 'yes'
    swell = u.get('swelling')           == 'yes'
    nause = u.get('poor_appetite')      == 'yes'
    urine = u.get('urination', 'normal')
    pain  = u.get('back_pain')          == 'yes'
    famhx = u.get('family_history')     == 'yes'
    obese = u.get('obesity')            == 'yes'

    bp    = 90 if htn else 70
    if smoke: bp += 10
    sg    = 1.010 if (tired or swell or urine != 'normal') else 1.020
    al    = 2 if (swell or urine == 'less') else 0
    su    = 2 if dm else 0
    rbc   = 1 if (tired or pain) else 0
    pc    = 1 if pain else 0
    pcc   = 1 if pain else 0
    ba    = 1 if pain else 0
    bgr   = 180 if dm else 110
    if obese: bgr += 30
    bu    = 60 if (tired and swell) else 30
    if famhx: bu += 15
    sc    = 2.5 if (tired and swell and urine == 'less') else 1.0
    if age > 60: sc += 0.5
    sod   = 130 if swell else 140
    pot   = 5.5 if (urine == 'less' and tired) else 4.5
    hemo  = 10.5 if tired else 14.0
    if famhx: hemo -= 1.0
    pcv   = 32 if tired else 44
    wbc   = 9000 if pain else 7500
    rbc_c = 3.5 if tired else 5.0
    cad   = 1 if (htn and age > 50) else 0
    appet = 1 if nause else 0
    pe    = 1 if swell else 0
    ane   = 1 if tired else 0

    return {
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
        'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
        'hemo': hemo, 'pcv': pcv,
        'wc': wbc,   'wbcc': wbc,
        'rc': rbc_c, 'rbcc': rbc_c,
        'htn': htn, 'dm': dm, 'cad': cad,
        'appet': appet, 'pe': pe, 'ane': ane
    }


def get_recommendation(pred, risk):
    if pred == 1:
        return ("⚠️ Our analysis indicates a HIGH risk of Chronic Kidney Disease. "
                "Please consult a nephrologist as soon as possible. "
                "Request blood tests (creatinine, urea) and urine analysis. "
                "Early treatment can significantly slow disease progression.")
    elif risk == 'Medium':
        return ("🟡 You show some moderate risk factors for CKD. "
                "We recommend scheduling a routine kidney health checkup. "
                "Maintain hydration, control blood pressure, and avoid NSAIDs regularly.")
    else:
        return ("✅ Based on your symptoms, your kidney health appears to be low risk. "
                "Continue maintaining a healthy lifestyle — stay hydrated, exercise regularly, "
                "control blood sugar and blood pressure, and get annual health checkups.")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Serve Frontend ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    possible = ['ckd-app-v3.html', 'ckd-app-v3', 'ckd_app_v3.html', 'ckd-app.html', 'index.html']
    for name in possible:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            print(f"✅ Serving: {path}")
            return send_file(path)
    files = os.listdir(BASE_DIR)
    return jsonify({'error': 'HTML not found', 'files': files}), 404


# ── REGISTER ───────────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data     = request.get_json()
    username = data.get('username', '').strip()
    email    = data.get('email',    '').strip().lower()
    name     = data.get('name',     '').strip()
    phone    = data.get('phone',    '').strip()
    password = data.get('password', '')

    if not all([username, email, name, phone, password]):
        return jsonify({'success': False, 'message': '❌ All fields are required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': '❌ Email already registered. Please sign in.'}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': '❌ Username already taken. Try another.'}), 409

    db.session.add(User(
        username=username, email=email, name=name,
        phone=phone, password=generate_password_hash(password)
    ))
    db.session.commit()
    print(f"✅ Registered: {email}")
    return jsonify({'success': True, 'message': '✅ Registration successful!', 'name': name})


# ── LOGIN ──────────────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json()
    email    = data.get('email',    '').strip().lower()
    password = data.get('password', '')

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'success': False, 'message': '❌ Email not found. Please register first.'}), 404
    if not check_password_hash(user.password, password):
        return jsonify({'success': False, 'message': '❌ Incorrect password. Try again.'}), 401

    print(f"✅ Login: {email}")
    return jsonify({
        'success':  True,
        'message':  '✅ Login successful!',
        'name':     user.name,
        'username': user.username,
        'email':    user.email
    })


# ── PREDICT ────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        clinical   = symptom_to_clinical(user_input)
        row        = [clinical.get(f, 0) for f in features]
        row_scaled = scaler.transform([row])
        pred       = model.predict(row_scaled)[0]
        prob       = model.predict_proba(row_scaled)[0]
        ckd_prob   = round(float(prob[1]) * 100, 1)
        risk_level = 'High' if prob[1] > 0.65 else 'Medium' if prob[1] > 0.35 else 'Low'

        # Save to database
        email = user_input.get('email', '').strip().lower()
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                db.session.add(Prediction(
                    user_id=user.id,
                    result='CKD' if pred == 1 else 'No CKD',
                    risk_level=risk_level,
                    ckd_probability=ckd_prob
                ))
                db.session.commit()
                print(f"✅ Prediction saved → {'CKD' if pred==1 else 'No CKD'} for {email}")

        return jsonify({
            'success':          True,
            'prediction':       int(pred),
            'label':            'CKD Detected' if pred == 1 else 'No CKD Detected',
            'ckd_probability':  ckd_prob,
            'safe_probability': round(float(prob[0]) * 100, 1),
            'risk_level':       risk_level,
            'recommendation':   get_recommendation(pred, risk_level)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ── VIEW ALL USERS ─────────────────────────────────────────────────────────────
@app.route('/api/users', methods=['GET'])
def get_users():
    all_users = User.query.all()
    return jsonify({
        'total': len(all_users),
        'users': [{
            'id':          u.id,
            'username':    u.username,
            'name':        u.name,
            'email':       u.email,
            'phone':       u.phone,
            'registered':  u.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(u.predictions)
        } for u in all_users]
    })


# ── VIEW PREDICTION HISTORY ────────────────────────────────────────────────────
@app.route('/api/history/<email>', methods=['GET'])
def get_history(email):
    user = User.query.filter_by(email=email.lower()).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    return jsonify({
        'success': True,
        'name':    user.name,
        'history': [{
            'result':      p.result,
            'risk_level':  p.risk_level,
            'probability': p.ckd_probability,
            'date':        p.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for p in user.predictions]
    })


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CKD Detection System")
    print("  URL:      http://127.0.0.1:5000")
    print("  Database: ckd_users.db (SQLite)")
    print("  Project:  " + BASE_DIR)
    print("="*50 + "\n")
    app.run(debug=True, port=5000)