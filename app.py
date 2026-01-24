import streamlit as st
from motogp_model import (
    load_and_prepare_data,
    train_dnf_classifier,
    train_finish_regressor,
    predict_result
)

# ---------- Page config ----------
st.set_page_config(
    page_title="MotoGP Race Predictor",
    page_icon="🏁",
    layout="centered"
)

st.title("🏍️ MotoGP Race Predictor")
st.caption("Predict DNF probability & finishing position using ML")

# ---------- Load & train models ----------
@st.cache_resource
def load_models():
    df = load_and_prepare_data("MGP2025.csv")
    dnf_model, input_dtypes = train_dnf_classifier(df)
    finish_model = train_finish_regressor(df)
    return dnf_model, finish_model, input_dtypes, df

dnf_model, finish_model, input_dtypes, df = load_models()

# ---------- UI inputs ----------
st.subheader("Race Inputs")

riders = sorted(df["RiderName"].unique())

rider = st.selectbox("Rider", riders)
grid = st.slider("Grid Position", 1, 22, 10)
sprint = st.slider("Sprint Finish", 1, 23, 10)

# ---------- Prediction ----------
if st.button("🔮 Predict Result"):
    result = predict_result(
        dnf_model,
        finish_model,
        input_dtypes,
        rider_name=rider,
        grid_position=grid,
        sprint_finish=sprint
    )

    st.divider()

    st.metric("DNF Probability", f"{result['DNF_Probability']*100:.2f}%")

    if result["Prediction"] == "Likely DNF":
        st.error("❌ Likely DNF")
    else:
        st.success(f"🏁 Predicted Finish Position: **{result['FinishPosition']}**")

