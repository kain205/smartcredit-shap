import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import requests
import joblib

# Load model and encoders (assume you have saved them, else retrain here)
# For demo, retrain model each time (not recommended for production)
def train_model():
    df = pd.read_csv('loan_data.csv')
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    from sklearn.preprocessing import LabelEncoder
    cat_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder= False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    return clf, encoders, X_train

clf, encoders, X_train = train_model()

st.title('Dự đoán phê duyệt đơn vay & Giải thích bằng SHAP')
st.markdown(
    """
    <div class="overlay"></div>
    <style>
    .stApp {
        background-image: url('https://i.postimg.cc/x8Ymdw0y/background.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: brightness(1.5);
    }

    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            rgba(10, 30, 60, 0.45),  
            rgba(5, 20, 40, 0.55)  
        );
        z-index: -1;
    }

    
    </style>
    """,
    unsafe_allow_html=True
)


# Input form
def user_input_features():
    df = pd.read_csv('loan_data.csv')
    str_cols = df.select_dtypes(include=['object']).columns
    select_inputs = {col: df[col].dropna().unique().tolist() for col in str_cols}

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input('Tuổi', min_value=18, max_value=100, value=30)
        person_gender = st.selectbox('Giới tính', select_inputs['person_gender'])
        person_education = st.selectbox('Trình độ học vấn', select_inputs['person_education'])
        person_income = st.number_input('Thu nhập', min_value=0, value=50000)
        person_home_ownership = st.selectbox('Hình thức sở hữu nhà', select_inputs['person_home_ownership'])
        loan_intent = st.selectbox('Mục đích vay', select_inputs['loan_intent'])
        previous_loan_defaults_on_file = st.selectbox('Có nợ xấu trước đây?', select_inputs['previous_loan_defaults_on_file'])

    with col2:
        person_emp_exp = st.number_input('Kinh nghiệm làm việc (năm)', min_value=0, value=5)
        loan_amnt = st.number_input('Số tiền vay', min_value=0, value=10000)
        loan_int_rate = st.number_input('Lãi suất (%)', min_value=0.0, value=10.0)
        loan_percent_income = st.number_input('Tỷ lệ vay/thu nhập (%)', min_value=0.0, value=20.0)
        cb_person_cred_hist_length = st.number_input('Thời gian lịch sử tín dụng', min_value=0, value=5)
        credit_score = st.number_input('Điểm tín dụng', min_value=0, value=700)

    data = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Encode categorical features
for col, le in encoders.items():
    input_df[col] = le.transform(input_df[col].astype(str))

# Dự đoán
if st.button('Dự đoán'):
    pred = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0][1]
    st.write(f'Kết quả dự đoán: {"Được phê duyệt" if pred==1 else "Không được phê duyệt"}')
    st.write(f'Xác suất được phê duyệt: {proba:.2f}')

    # SHAP explain
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(input_df)
    st.subheader('Giải thích trực quan hóa SHAP')
    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_df.iloc[0])

    st.pyplot(fig)

    # Gọi dify.ai API để lấy giải thích AI (giả sử bạn có API endpoint và key)
    st.subheader('Giải thích từ AI (dify.ai)')
    try:
        api_url = st.secrets["dify"]["api_url"]
        api_key = st.secrets["dify"]["api_key"]
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        # Tạo input dạng paragraph
        input_text = '\n'.join([f"{k}: {v}" for k, v in input_df.to_dict(orient='records')[0].items()])
        prediction_text = "Được phê duyệt" if pred==1 else "Không được phê duyệt"
        # query là nội dung chính để AI xử lý
        query = f"Thông tin khách hàng:\n{input_text}\nKết quả dự đoán: {prediction_text}\nHãy giải thích chi tiết lý do vì sao hệ thống đưa ra quyết định này."
        payload = {
            "inputs": {"input": input_text, "prediction": prediction_text},
            "query": query,
            "response_mode": "blocking",
            "user": "user1"
        }
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            # Dify trả về answer trong trường 'answer'
            st.write(response.json().get('answer', 'Không có giải thích trả về.'))
        else:
            st.write(f'Không thể lấy giải thích từ AI. Mã lỗi: {response.status_code}')
            try:
                st.write('Chi tiết lỗi:', response.json())
            except Exception:
                st.write('Không đọc được nội dung lỗi chi tiết từ response.')
    except Exception as e:
        st.write('Lỗi khi gọi API:', str(e))
