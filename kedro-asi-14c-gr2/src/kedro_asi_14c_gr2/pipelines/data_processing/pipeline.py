from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_preparation, create, evaluate_model, save_model


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
                func=create,
                inputs="preprocessed_obesity",
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
                inputs=["scored_model", "../kedro-asi-14c-gr2"],
                outputs="",
                name="save_model_node"
            ),


        ]
    )
