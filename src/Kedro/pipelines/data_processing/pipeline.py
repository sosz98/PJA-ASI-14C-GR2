from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_preparation, create, evaluate_model, save_model, validate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_preparation,
                inputs="obesity",
                outputs="preprocessed_obesity",
                name="preprocess_obesity_node"
            ),
            node(
                func=validate_data,
                inputs="preprocessed_obesity",
                outputs="validated_obesity",
                name="validation_node"
            ),
            node(
                func=create,
                inputs="validated_obesity",
                outputs="initial_model",
                name="model_creation_node"
            ),
            node(
                func=evaluate_model,
                inputs="initial_model",
                outputs="scored_model",
                name="model_score_node"
            ),
            node(
                func=save_model,
                inputs=["scored_model", "params:model_filepath"],
                outputs=None,
                name="save_model_node"
            ),
        ]
    )
