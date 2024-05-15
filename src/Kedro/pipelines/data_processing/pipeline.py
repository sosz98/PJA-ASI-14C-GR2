from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create, validate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
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
            )
        ]
    )


data_processing_pipeline = create_pipeline()
