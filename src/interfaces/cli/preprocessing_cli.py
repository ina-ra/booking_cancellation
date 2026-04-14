def main():
    from src.infrastructure.data.preprocessing import save_processed_training_dataset

    processed_df, summary = save_processed_training_dataset()
    print("Размер датасета до обработки:", summary["original_shape"])
    print("Дубликаты:", summary["duplicates_count"])
    print("Некорректные даты:", summary["invalid_dates_count"])
    print("Размер датасета:", summary["processed_shape"])
    print("\nСписок признаков:")
    print(processed_df.columns.tolist())
    print("\nДобавленные признаки:")
    print(summary["added_features"])
    print("\nУдалённые признаки:")
    print(summary["removed_features"])
    print("\nПервые 5 строк:")
    print(processed_df.head())


if __name__ == "__main__":
    main()
