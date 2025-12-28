from model import train_model
import pandas as pd

# Обучаем модель на данных
model, vectorizer = train_model("data/sample_data.csv")

# Читаем CSV
df = pd.read_csv("data/sample_data.csv")

# Для вопросов без difficulty предсказываем
for index, row in df.iterrows():
    if pd.isna(row['difficulty']) or row['difficulty'] == '':
        question = row['question']
        vec = vectorizer.transform([question])
        prediction = model.predict(vec)
        df.at[index, 'difficulty'] = prediction[0]
        print(f"Вопрос: {question} -> Сложность: {prediction[0]}")

# Сохраняем обновленный CSV
df.to_csv("data/sample_data.csv", index=False)
