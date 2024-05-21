from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

label = 'OB_LEVEL'


def automl_train_evaluate(obesity, automl_model_filepath) -> None:
    train, test = train_test_split(obesity, test_size=0.2)
    predictor = TabularPredictor(label=label, path=automl_model_filepath).fit(train_data=train)
    performance = predictor.evaluate(test)
    print(performance)
    predictor.export(automl_model_filepath)
