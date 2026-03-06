from flask import Flask, request, redirect, url_for, render_template, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz
import joblib
import pandas as pd
from statistics import mode, StatisticsError
import traceback

import os
import json
import threading
import time
import paho.mqtt.client as mqtt


def get_indian_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///sensor_data.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = 'dwaipayan_1705'


# ===== HiveMQ Cloud vars (set in Render Environment) =====
# NOTE: Prefer to set these ONLY via Render Environment (no hardcoded secrets).
MQTT_HOST = os.getenv("MQTT_HOST", "c31eaaf0bc9140159734684c588e5bef.s1.eu.hivemq.cloud")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USER", "Satrajit")
MQTT_PASS = os.getenv("MQTT_PASS", "Sm22072003#")
TLS_ENABLED = os.getenv("TLS_ENABLED", "1").lower() in ("1", "true", "yes", "on")

TOPIC_READINGS_SUB = os.getenv("TOPIC_READINGS_SUB", "srcs/readings/#")
TOPIC_PRED_PREFIX  = os.getenv("TOPIC_PRED_PREFIX",  "srcs/predictions/")

DEVICE_ID = os.getenv("DEVICE_ID", "SRCS_S3_01")
# =======================================================


class sensor_data(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, default=get_indian_time)
    A_410 = db.Column(db.Float, nullable=False)
    B_435 = db.Column(db.Float, nullable=False)
    C_460 = db.Column(db.Float, nullable=False)
    D_485 = db.Column(db.Float, nullable=False)
    E_510 = db.Column(db.Float, nullable=False)
    F_535 = db.Column(db.Float, nullable=False)
    G_560 = db.Column(db.Float, nullable=False)
    H_585 = db.Column(db.Float, nullable=False)
    R_610 = db.Column(db.Float, nullable=False)
    I_645 = db.Column(db.Float, nullable=False)
    S_680 = db.Column(db.Float, nullable=False)
    J_705 = db.Column(db.Float, nullable=False)
    T_730 = db.Column(db.Float, nullable=False)
    U_760 = db.Column(db.Float, nullable=False)
    V_810 = db.Column(db.Float, nullable=False)
    W_860 = db.Column(db.Float, nullable=False)
    K_900 = db.Column(db.Float, nullable=False)
    L_940 = db.Column(db.Float, nullable=False)


def mqtt_ingest_worker():
    if not MQTT_HOST:
        print("MQTT_HOST not set. MQTT ingest disabled.")
        return

    client = mqtt.Client()

    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    if TLS_ENABLED:
        # Basic TLS (for HiveMQ Cloud 8883)
        client.tls_set()
        # For troubleshooting TLS cert issues on Render, you may enable this:
        # client.tls_insecure_set(True)

    def on_connect(c, userdata, flags, rc):
        print("MQTT connected rc=", rc)
        c.subscribe(TOPIC_READINGS_SUB)
        print("Subscribed:", TOPIC_READINGS_SUB)

    def on_message(c, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore")
        try:
            data = json.loads(payload)
            channels = data.get("channels")

            if not isinstance(channels, list) or len(channels) != 18:
                raise ValueError("Expected channels length 18")

            new_entry = sensor_data(
                A_410=channels[0], B_435=channels[1], C_460=channels[2], D_485=channels[3],
                E_510=channels[4], F_535=channels[5], G_560=channels[6], H_585=channels[7],
                R_610=channels[8], I_645=channels[9], S_680=channels[10], J_705=channels[11],
                T_730=channels[12], U_760=channels[13], V_810=channels[14], W_860=channels[15],
                K_900=channels[16], L_940=channels[17]
            )

            with app.app_context():
                db.session.add(new_entry)
                db.session.commit()

            print("Stored MQTT reading")

        except Exception as e:
            print("MQTT ingest error:", e)
            print("Payload:", payload)

    client.on_connect = on_connect
    client.on_message = on_message

    # Keep trying forever (helps if network blips)
    while True:
        try:
            print(f"Connecting MQTT to {MQTT_HOST}:{MQTT_PORT} TLS={TLS_ENABLED} ...")
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print("MQTT worker crashed, retrying in 5s:", e)
            time.sleep(5)


# --- IMPORTANT: Start MQTT thread even under gunicorn ---
_mqtt_thread_started = False

def start_mqtt_thread_once():
    global _mqtt_thread_started
    if _mqtt_thread_started:
        return
    _mqtt_thread_started = True
    threading.Thread(target=mqtt_ingest_worker, daemon=True).start()
    print("✅ MQTT ingest thread started")


# Create DB + start MQTT thread at import time (works on Render/Gunicorn)
with app.app_context():
    db.create_all()

start_mqtt_thread_once()
# ------------------------------------------------------


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')


@app.route('/index', methods=['GET'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    latest = sensor_data.query.order_by(sensor_data.time.desc()).limit(5).all()
    prediction = request.args.get('prediction')
    preprocess = request.args.get('preprocess')
    model_name = request.args.get('model_name')
    last_updated = datetime.now().strftime("%H:%M:%S") if prediction else None

    return render_template(
        'index_mod.html',
        data=latest,
        prediction=prediction,
        last_updated=last_updated,
        preprocess=preprocess,
        model_name=model_name
    )


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def predict():
    preprocess = request.form.get('preprocess')
    model_name = request.form.get('model')

    latest_data = sensor_data.query.order_by(sensor_data.time.desc()).limit(5).all()
    if not latest_data:
        return redirect(url_for('index', prediction="No data found"))

    df = pd.DataFrame([
        {col.name: getattr(row, col.name) for col in sensor_data.__table__.columns}
        for row in reversed(latest_data)
    ])
    df = df.drop(columns=['sno', 'time'])

    try:
        if preprocess == 'Raw Data':
            processed = df.values
        elif preprocess == 'Scaled data (Standard Scaler)':
            scaler = joblib.load('models/scaler.pkl')
            processed = scaler.transform(df)
        elif preprocess == 'Principal Component Analysis':
            pca = joblib.load('models/pca.pkl')
            processed = pca.transform(df)
        elif preprocess == 'Linear Discriminant Analysis':
            lda = joblib.load('models/lda.pkl')
            processed = lda.transform(df)
        else:
            return redirect(url_for('index', prediction="Invalid preprocessing selected"))
    except Exception:
        traceback.print_exc()
        return redirect(url_for('index', prediction="Preprocessing failed. Check server logs."))

    try:
        model_map = {
            'Support Vector Machine': 'models/svm_model.pkl',
            'k Nearest Neighbour': 'models/knn_model.pkl',
            'Random Forest':  'models/rf_model.pkl',
            'Gaussian Processes':  'models/gp_model.pkl'
        }
        if model_name not in model_map:
            return redirect(url_for('index', prediction="Invalid model selected"))
        model = joblib.load(model_map[model_name])
    except Exception as e:
        return redirect(url_for('index', prediction=f"Model loading failed: {e}"))

    try:
        predictions = model.predict(processed)
        try:
            modal_value = mode(predictions)
        except StatisticsError:
            modal_value = predictions[0]
        prediction_text = str(modal_value)
    except Exception as e:
        prediction_text = f"Prediction failed: {str(e)}"

    # Publish prediction back to ESP32
    try:
        if MQTT_HOST:
            pred_topic = TOPIC_PRED_PREFIX + DEVICE_ID
            pred_payload = {
                "device_id": DEVICE_ID,
                "label": prediction_text,
                "confidence": None,
                "preprocess": preprocess,
                "model_name": model_name,
                "ts": time.time()
            }

            pub = mqtt.Client()
            if MQTT_USER:
                pub.username_pw_set(MQTT_USER, MQTT_PASS)
            if TLS_ENABLED:
                pub.tls_set()
                # pub.tls_insecure_set(True)  # troubleshooting only

            pub.connect(MQTT_HOST, MQTT_PORT, 60)
            pub.publish(pred_topic, json.dumps(pred_payload))
            pub.disconnect()

            print("Published prediction to:", pred_topic, pred_payload)
    except Exception as e:
        print("MQTT publish error:", e)

    return redirect(url_for('index', prediction=prediction_text, preprocess=preprocess, model_name=model_name))


if __name__ == '__main__':
    port = int(os.getenv("PORT", "5000"))
    app.run(host='0.0.0.0', port=port)
