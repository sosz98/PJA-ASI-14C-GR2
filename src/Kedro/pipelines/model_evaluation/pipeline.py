from kedro.pipeline import Pipeline, pipeline, node
from .nodes import evaluate_model, save_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
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
    ])


model_evaluation_pipeline = create_pipeline()
