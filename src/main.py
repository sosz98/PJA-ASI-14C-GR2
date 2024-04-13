from Data_collecting import collect_data
from Data_preparation import prepare_data
from Data_validation import validate_data
from Model_creation import create_model
from Model_evaluation import model_evaluation
from Model_save import model_save

if __name__ == '__main__':
    obesity = collect_data.collect_data()
    prepare_data.data_preparation(obesity)
    validate_data.validate_column_types(obesity)
    validate_data.validate_categorical_column_ranges(obesity)
    model_bean = create_model.create(obesity)
    model_evaluation.evaluate_model(model_bean)
    model_save.save_model(model_bean.rfc, '../')
