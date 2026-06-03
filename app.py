import streamlit as st
import requests

st.title("Predict Credit Card Defaulters")
st.write("Enter the details of the credit card holder to predict if they will default on their payment!")

col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input("Credit Card Limit Balance", min_value=0.0, step=1000.0, value=50000.0)
    sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
    education = st.selectbox('Education level', options=[1, 2, 3, 4], format_func=lambda x: 'Graduate school' if x == 1 else ('University' if x == 2 else ('High school' if x == 3 else 'Others')))
    marriage = st.selectbox('Marriage status', options=[1, 2, 3], format_func=lambda x: 'Married' if x == 1 else ('Single' if x == 2 else 'Others'))
    age = st.slider('Age', min_value=18, max_value=100, value=22)

with col2:
    utilisation_rate = st.slider('Utilisation Rate', min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    pay_to_bill_ratio = st.slider('Pay to Bill Ratio', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    delinquency_score = st.slider('Delinquency Score', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

if st.button("Predict"):
    input_data = {
        "limit_bal": limit_bal,
        "sex": sex,
        "education": education,
        "marriage": marriage,
        "age": age,
        "utilisation_rate": utilisation_rate,
        "pay_to_bill_ratio": pay_to_bill_ratio,
        "delinquency_score": delinquency_score,
        
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
    if response.status_code == 200:
        result = response.json()
        
        st.markdown("---") # Adds a clean visual divider
        st.subheader("🎯 Prediction Result")
        
        # 1. Use Columns to display metrics side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Credit Score", value=int(result['credit_score']))
        with col2:
            # Convert probability to a clean percentage string
            prob_percent = float(result['defaulting_probability']) * 100
            st.metric(label="Default Probability", value=f"{prob_percent:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True) # Tiny spacing buffer

        # 2. Use color-coded banners based on the actual decision
        decision = str(result['decision'])
        risk = str(result['risk_category'])
        
        if decision == 'Approved':
            st.success(f"**Decision: {decision}** — {risk}")
        elif decision == 'Approved with Caution':
            st.warning(f"**Decision: {decision}** — {risk}")
        else:
            st.error(f"**Decision: {decision}** — {risk}")

        # 3. Tuck the long explanation inside a clickable expander to keep the UI clean
        with st.expander("ℹ️ How is this decision made?"):
            st.write("""
            **Optimal Risk Threshold: 11% (0.11)** The system classifies applicants based on their calculated probability of defaulting:
            * **Less than 11%:** Low Risk - Approved for maximum of limit $50000
            * **11% to 25%:** Medium Risk - Approved with caution, maximum of limit $30000
            * **Above 25%:** High Risk - Rejected $0 limit.
            """)

    else:
        st.error(f"🚨 Error {response.status_code}: Failed to get a valid response from the API.")
        st.write(response.text) # Shows the exact API error to the user for easier debugging

