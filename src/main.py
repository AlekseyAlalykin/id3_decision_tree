import util
import warnings
import math
from decision_tree import DecisionTree


def main():
    learning_sample = util.get_samples("AppleQualityDataset_Learning.xlsx")
    test_sample = util.get_samples("AppleQualityDataset_Test.xlsx")

    for i in range(1, 51):
        interval = i / 10

        segregated_learning_sample = util.segregate_floats(learning_sample, interval)
        segregated_test_sample = util.segregate_floats(test_sample, interval, False)

        dt = DecisionTree(segregated_learning_sample, "Качество", "good")

        success_counter = 0
        for item in segregated_test_sample.to_dict('records'):
            prediction = dt.predict(item)
            if item["Качество"] == prediction:
                success_counter += 1

        accuracy = round(success_counter / len(test_sample), 3)
        print(f"Group size: {interval}, Accuracy:{accuracy}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
