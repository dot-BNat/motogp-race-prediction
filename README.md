# 🏁 MotoGP Race Outcome Prediction (2025)

This project predicts:
- 🛑 DNF probability
- 🏎️ Final race finishing position (if rider finishes)

using machine learning models trained on 2025 MotoGP race data.

## 📊 Features Used
- Rider Name
- Grid Position
- Sprint Race Finish

## 🧠 Models
- RandomForestClassifier → DNF prediction
- RandomForestRegressor → Finish position prediction

## 🔄 Pipeline
1. Preprocess rider & race data
2. Predict DNF probability
3. If rider finishes → predict final position

## 🧪 Example
```python
predict_result("Pedro Acosta", grid=2, sprint=2)
