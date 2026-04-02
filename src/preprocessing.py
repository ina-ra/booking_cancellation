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
print("Размер датасета до обработки:", df.shape)
# сохраняем исходные признаки
original_columns = df.columns.tolist()

# -----------------------------
# 3. Сохранение Booking_ID
# -----------------------------
# Сохраняем ID отдельно, чтобы не потерять связь
# между предсказанием и конкретным бронированием
booking_ids = None
if "Booking_ID" in df.columns:
    booking_ids = df["Booking_ID"].copy()
    df = df.drop(columns=["Booking_ID"])

# -----------------------------
# 4. Обработка пропусков
# -----------------------------
# print("Пропуски до обработки:\n", df.isnull().sum())

# Числовые признаки заполняем медианой
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Категориальные признаки заполняем модой
cat_cols = df.select_dtypes(include=["object", "string"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# print("Пропуски после обработки:\n", df.isnull().sum())

# -----------------------------
# 5. Удаление дубликатов
# -----------------------------
print("Дубликаты:", df.duplicated().sum())
df = df.drop_duplicates()
# print("Дубликаты после:", df.duplicated().sum())

# -----------------------------
# 6. Преобразование target
# -----------------------------
# Таргет — это переменная, которую модель будет предсказывать
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

# print("\nНекорректные даты:", df["date of reservation"].isna().sum())

# Удаляем строки с некорректной датой
df = df.dropna(subset=["date of reservation"])

# Создаём признаки из даты
df["reservation_month"] = df["date of reservation"].dt.month
df["reservation_dayofweek"] = df["date of reservation"].dt.dayofweek

# Исходную колонку с датой удаляем
df = df.drop(columns=["date of reservation"])

# -----------------------------
# 8. Feature Engineering
# -----------------------------
# Общее количество ночей
df["total_nights"] = df["number of weekend nights"] + df["number of week nights"]

# Цена за ночь
df["price_per_night"] = df["average price"] / (df["total_nights"] + 1)

# Есть ли дети
df["is_family"] = (df["number of children"] > 0).astype(int)

# -----------------------------
# 9. Возвращаем Booking_ID
# -----------------------------
# Возвращаем только те ID, которые соответствуют строкам,
# оставшимся после удаления дубликатов и некорректных дат
if booking_ids is not None:
    df["Booking_ID"] = booking_ids.loc[df.index]

# -----------------------------
# 10. Сохранение обработанного датасета
# -----------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

# -----------------------------
# 11. Итог
# -----------------------------
print("Размер датасета:", df.shape)

print("\nСписок признаков:")
print(df.columns.tolist())

print("\nДобавленные признаки:")
new_columns = df.columns.tolist()
added_features = list(set(new_columns) - set(original_columns))
print(added_features)

removed_features = list(set(original_columns) - set(new_columns))

print("\nУдалённые признаки:")
print(removed_features)

# print("\nПервые 5 строк:")
# print(df.head())

# запуск из терминала: python src/preprocessing.py