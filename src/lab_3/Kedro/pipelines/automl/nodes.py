from autogluon.tabular import TabularPredictor

label = 'OB_LEVEL'


def automl_train_evaluate(data, automl_model_filepath):
    predictor = TabularPredictor(label=label, path=automl_model_filepath).fit(train_data=data)
    performance = predictor.evaluate(data)
    predictor.export(automl_model_filepath)
    return predictor, performance
