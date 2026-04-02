import os
import pandas as pd

# -----------------------------
# 1. Пути к файлам
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(BASE_DIR, "data", "raw", "booking.csv")
output_path = os.path.join(BASE_DIR, "data", "processed", "booking_clean.csv")

# -----------------------------
# 2. Загрузка данных
# -----------------------------
df = pd.read_csv(input_path)

# -----------------------------
# 3. Удаление ненужных колонок
# -----------------------------
if "Booking_ID" in df.columns:
    df = df.drop(columns=["Booking_ID"])

# -----------------------------
# 4. Обработка пропусков
# -----------------------------
print("Пропуски до обработки:\n", df.isnull().sum())

# Числовые → медиана
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Категориальные → мода
cat_cols = df.select_dtypes(include=["object", "string"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Пропуски после обработки:\n", df.isnull().sum())

# -----------------------------
# 5. Удаление дубликатов
# -----------------------------
print("Дубликаты до:", df.duplicated().sum())
df = df.drop_duplicates()
print("Дубликаты после:", df.duplicated().sum())

# -----------------------------
# 6. Преобразование target
# -----------------------------
df["booking status"] = df["booking status"].map({
    "Canceled": 1,
    "Not_Canceled": 0
})

# -----------------------------
# 7. Работа с датой
# -----------------------------
df["date of reservation"] = pd.to_datetime(
    df["date of reservation"],
    format="mixed",
    errors="coerce"
)

print("\nНекорректные даты:", df["date of reservation"].isna().sum())

df = df.dropna(subset=["date of reservation"])  

df["reservation_month"] = df["date of reservation"].dt.month
df["reservation_dayofweek"] = df["date of reservation"].dt.dayofweek

df = df.drop(columns=["date of reservation"])

# -----------------------------
# 8. Feature Engineering
# -----------------------------
df["total_nights"] = df["number of weekend nights"] + df["number of week nights"]

df["price_per_night"] = df["average price"] / (df["total_nights"] + 1)

df["is_family"] = (df["number of children"] > 0).astype(int)

# -----------------------------
# 9. Сохранение обработанного датасета
# -----------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

# -----------------------------
# 10. Итог
# -----------------------------
print("Размер датасета:", df.shape)

print("\nСписок признаков:")
print(df.columns.tolist())

print("\nПервые 5 строк:")
print(df.head())

# запуск из терминала: python src/preprocessing.py