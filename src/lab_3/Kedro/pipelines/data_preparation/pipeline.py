from kedro.pipeline import Pipeline, pipeline, node
from .nodes import data_preparation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=data_preparation,
            inputs="obesity",
            outputs="preprocessed_obesity",
            name="preprocess_obesity_node"
        ),
    ])


