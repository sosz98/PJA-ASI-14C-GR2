from kedro.pipeline import Pipeline, node
from .nodes import automl_train_evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=automl_train_evaluate,
            inputs=["obesity", "params:automl_model_filepath"],
            outputs=None,
            name="automl_node"
        )
    ])
