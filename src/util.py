import pandas


# Читает таблицу и приводит в формат списка без стобца ID
def get_samples(filepath):
    excel_data = pandas.read_excel(filepath)
    data = pandas.DataFrame(excel_data).drop('ID', axis=1)
    return data


min_max_values = {}


# Разделение вещественных чисел на группы округляя их до round_to знаков после запятой
def segregate_floats(table, round_to, learning_sample=True):
    for column in table.columns:
        table[column] = table[column].round(round_to)
        if not learning_sample:
            table[column].where(table[column] < min_max_values[column]['max'], min_max_values[column]['max'], inplace=True)
            table[column].where(table[column] > min_max_values[column]['min'], min_max_values[column]['min'], inplace=True)

        if learning_sample:
            min_max_values[column] = {
                'min': min(table[column]),
                'max': max(table[column])
            }

    return table
