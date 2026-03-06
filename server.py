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
app.secret_key = 'dwaipayan_1705'
db = SQLAlchemy(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ===== HiveMQ Cloud vars (prefer Render Environment) =====
MQTT_HOST = os.getenv("MQTT_HOST", "c31eaaf0bc9140159734684c588e5bef.s1.eu.hivemq.cloud")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USER", "Satrajit")
MQTT_PASS = os.getenv("MQTT_PASS", "Sm22072003#")
TLS_ENABLED = os.getenv("TLS_ENABLED", "1").lower() in ("1", "true", "yes", "on")

TOPIC_READINGS_SUB = os.getenv("TOPIC_READINGS_SUB", "srcs/readings/#")
TOPIC_PRED_PREFIX = os.getenv("TOPIC_PRED_PREFIX", "srcs/predictions/")
DEVICE_ID = os.getenv("DEVICE_ID", "SRCS_S3_01")
# ========================================================


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
        print("MQTT_HOST not set. MQTT ingest disabled.", flush=True)
        return

    client = mqtt.Client()

    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    if TLS_ENABLED:
        client.tls_set()
        # client.tls_insecure_set(True)  # only for troubleshooting

    def on_connect(c, userdata, flags, rc):
        print("MQTT connected rc =", rc, flush=True)
        c.subscribe(TOPIC_READINGS_SUB)
        print("Subscribed:", TOPIC_READINGS_SUB, flush=True)

    def on_message(c, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore")
        try:
            data = json.loads(payload)
            channels = data.get("channels")

            if not isinstance(channels, list) or len(channels) != 18:
                raise ValueError("Expected channels length 18")

            new_entry = sensor_data(
                A_410=float(channels[0]),
                B_435=float(channels[1]),
                C_460=float(channels[2]),
                D_485=float(channels[3]),
                E_510=float(channels[4]),
                F_535=float(channels[5]),
                G_560=float(channels[6]),
                H_585=float(channels[7]),
                R_610=float(channels[8]),
                I_645=float(channels[9]),
                S_680=float(channels[10]),
                J_705=float(channels[11]),
                T_730=float(channels[12]),
                U_760=float(channels[13]),
                V_810=float(channels[14]),
                W_860=float(channels[15]),
                K_900=float(channels[16]),
                L_940=float(channels[17])
            )

            with app.app_context():
                db.session.add(new_entry)
                db.session.commit()

            print("Stored MQTT reading", flush=True)

        except Exception as e:
            print("MQTT ingest error:", str(e), flush=True)
            print("Payload:", payload, flush=True)
            traceback.print_exc()

    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            print(f"Connecting MQTT to {MQTT_HOST}:{MQTT_PORT} TLS={TLS_ENABLED} ...", flush=True)
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print("MQTT worker crashed, retrying in 5s:", str(e), flush=True)
            traceback.print_exc()
            time.sleep(5)


_mqtt_thread_started = False


def start_mqtt_thread_once():
    global _mqtt_thread_started
    if _mqtt_thread_started:
        return
    _mqtt_thread_started = True
    threading.Thread(target=mqtt_ingest_worker, daemon=True).start()
    print("MQTT ingest thread started", flush=True)


with app.app_context():
    db.create_all()

start_mqtt_thread_once()


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

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

    print("FORM DATA:", request.form, flush=True)
    print("preprocess =", preprocess, flush=True)
    print("model_name =", model_name, flush=True)

    latest_data = sensor_data.query.order_by(sensor_data.time.desc()).limit(5).all()
    if not latest_data:
        return redirect(url_for('index', prediction="No data found"))

    try:
        df = pd.DataFrame([
            {col.name: getattr(row, col.name) for col in sensor_data.__table__.columns}
            for row in reversed(latest_data)
        ])

        df = df.drop(columns=['sno', 'time'])
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0.0)
        df = df.astype(float)

        print("DF shape:", df.shape, flush=True)
        print("DF columns:", df.columns.tolist(), flush=True)

    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('index', prediction=f"Data preparation failed: {str(e)}"))

    try:
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        pca_path = os.path.join(MODELS_DIR, 'pca.pkl')
        lda_path = os.path.join(MODELS_DIR, 'lda.pkl')

        if preprocess == 'raw':
            processed = df.values

        elif preprocess == 'scaled':
            scaler = joblib.load(scaler_path)
            processed = scaler.transform(df)

        elif preprocess == 'pca':
            pca = joblib.load(pca_path)
            processed = pca.transform(df)

        elif preprocess == 'lda':
            lda = joblib.load(lda_path)
            processed = lda.transform(df)

        else:
            return redirect(url_for('index', prediction=f"Invalid preprocessing selected: {preprocess}"))

        print("Processed shape:", processed.shape, flush=True)

    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('index', prediction=f"Preprocessing failed: {str(e)}"))

    try:
        model_map = {
            'svm': os.path.join(MODELS_DIR, 'svm_model.pkl'),
            'knn': os.path.join(MODELS_DIR, 'knn_model.pkl'),
            'rf': os.path.join(MODELS_DIR, 'rf_model.pkl'),
            'gp': os.path.join(MODELS_DIR, 'gp_model.pkl')
        }

        if model_name not in model_map:
            return redirect(url_for('index', prediction=f"Invalid model selected: {model_name}"))

        model = joblib.load(model_map[model_name])

    except Exception as e:
        traceback.print_exc()
        return redirect(url_for('index', prediction=f"Model loading failed: {str(e)}"))

    try:
        predictions = model.predict(processed)

        try:
            modal_value = mode(predictions)
        except StatisticsError:
            modal_value = predictions[0]

        prediction_text = str(modal_value)
        print("Prediction:", prediction_text, flush=True)

    except Exception as e:
        traceback.print_exc()
        prediction_text = f"Prediction failed: {str(e)}"

    # Publish prediction back to ESP32
    try:
        print("Entering MQTT publish block", flush=True)

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

            print("Publishing to topic:", pred_topic, flush=True)
            print("Payload:", pred_payload, flush=True)

            pub = mqtt.Client()

            if MQTT_USER:
                pub.username_pw_set(MQTT_USER, MQTT_PASS)

            if TLS_ENABLED:
                pub.tls_set()
                # pub.tls_insecure_set(True)  # only for troubleshooting

            pub.connect(MQTT_HOST, MQTT_PORT, 60)
            pub.loop_start()

            result = pub.publish(pred_topic, json.dumps(pred_payload), qos=1)
            result.wait_for_publish()

            print("Publish rc:", result.rc, flush=True)

            time.sleep(1)

            pub.loop_stop()
            pub.disconnect()

            print("Published prediction successfully", flush=True)

    except Exception as e:
        print("MQTT publish error:", str(e), flush=True)
        traceback.print_exc()

    return redirect(url_for(
        'index',
        prediction=prediction_text,
        preprocess=preprocess,
        model_name=model_name
    ))


if __name__ == '__main__':
    port = int(os.getenv("PORT", "5000"))
    app.run(host='0.0.0.0', port=port)
