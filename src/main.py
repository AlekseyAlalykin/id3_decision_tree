import util
import warnings
from decision_tree import DecisionTree


def main():
    learning_sample = util.segregate_floats(util.get_samples("AppleQualityDataset_Learning.xlsx"), 0)
    test_sample = util.segregate_floats(util.get_samples("AppleQualityDataset_Test.xlsx"), 0, False)

    dt = DecisionTree(learning_sample, "Качество", "good", '', None)

    success_counter = 0
    for item in test_sample.to_dict('records'):
        prediction = dt.predict(item)
        if item["Качество"] == prediction:
            success_counter += 1

    print(f"Success rate: {round((success_counter/len(test_sample))*100)}%")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
