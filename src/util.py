import pandas


# Читает таблицу и приводит в формат списка без стобца ID
def get_samples(filepath):
    excel_data = pandas.read_excel(filepath)
    data = pandas.DataFrame(excel_data).drop('ID', axis=1)
    return data


min_max_values = {}


# Разделение вещественных чисел на группы размером group_size
# Возвращается новая таблица, данные в оригинальной не меняются
def segregate_floats(original_table, interval, learning_sample=True):
    table = original_table.copy(deep=True)
    for column in table.columns:
        #table[column] = table[column].round(round_to)

        if learning_sample:
            min_max_values[column] = {
                'min': min(table[column]),
                'max': max(table[column])
            }
        else:
            table[column].where(table[column] < min_max_values[column]['max'], min_max_values[column]['max'], inplace=True)
            table[column].where(table[column] > min_max_values[column]['min'], min_max_values[column]['min'], inplace=True)

        if table[column].dtype != object:
            table[column] = (table[column] / interval).round()

    return table
