import streamlit as st
import pandas as pd
import joblib

# ===== LOAD MODEL =====
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")
st.markdown("Nhập thông tin khách hàng để dự đoán khả năng rời bỏ")

# ===== INPUT =====
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method",
                               ["Electronic check", "Mailed check", "Credit card (automatic)"])

# ===== PREPROCESS FUNCTION =====
def preprocess():
    df = pd.DataFrame()

    # Numeric
    df['tenure'] = [tenure]
    df['MonthlyCharges'] = [monthly]
    df['TotalCharges'] = [total]

    # Binary
    df['gender'] = [1 if gender == "Male" else 0]
    df['SeniorCitizen'] = [senior]
    df['Partner'] = [1 if partner == "Yes" else 0]
    df['Dependents'] = [1 if dependents == "Yes" else 0]

    df['PhoneService'] = [1]
    df['MultipleLines'] = [0]
    df['OnlineSecurity'] = [0]
    df['OnlineBackup'] = [0]
    df['DeviceProtection'] = [0]
    df['TechSupport'] = [0]
    df['StreamingTV'] = [0]
    df['StreamingMovies'] = [0]
    df['PaperlessBilling'] = [1]

    # Contract
    df['Contract_One year'] = 1 if contract == "One year" else 0
    df['Contract_Two year'] = 1 if contract == "Two year" else 0

    # Internet
    df['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
    df['InternetService_No'] = 1 if internet == "No" else 0

    # Payment
    df['PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
    df['PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0
    df['PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0

    # Tenure group
    df['tenure_group_6-12'] = 1 if 6 <= tenure < 12 else 0
    df['tenure_group_12+'] = 1 if tenure >= 12 else 0

    # ===== ALIGN COLUMNS =====
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    return df

# ===== PREDICT =====
if st.button("🔍 Predict"):

    input_df = preprocess()

    prob = model.predict_proba(input_df)[0][1]
    pred = "Yes" if prob > 0.5 else "No"

    st.subheader("📈 Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2%}")
    st.write(f"**Prediction:** {pred}")

    # Progress bar
    st.progress(int(prob * 100))

    # ===== GỢI Ý HÀNH ĐỘNG =====
    st.subheader("💡 Gợi ý hành động")

    if prob > 0.7:
        st.error("🔴 Nguy cơ cao: Cần giữ chân ngay (giảm giá, ưu đãi, chăm sóc đặc biệt)")
    elif prob > 0.4:
        st.warning("🟠 Nguy cơ trung bình: Cải thiện chất lượng dịch vụ, tăng tương tác khách hàng")
    else:
        st.success("🟢 Nguy cơ thấp: Duy trì chăm sóc và có thể upsell dịch vụ")