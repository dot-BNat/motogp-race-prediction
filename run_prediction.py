from motogp_model import (
    load_and_prepare_data,
    train_dnf_classifier,
    train_finish_regressor,
    predict_result
)

df = load_and_prepare_data("MGP2025.csv")

dnf_model, input_dtypes = train_dnf_classifier(df)
finish_model = train_finish_regressor(df)

result = predict_result(
    dnf_model,
    finish_model,
    input_dtypes,
    rider_name="Pedro Acosta",
    grid_position=2,
    sprint_finish=2
)

print(result)
