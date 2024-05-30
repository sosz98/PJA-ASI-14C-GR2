from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_champion, compare_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_champion,
                inputs="challenger_model",
                outputs="champion_model",
                name="loading_champion",
            ),
            node(
                func=compare_models,
                inputs=["champion_model", "challenger_model"],
                outputs=None,
                name="model_score_node",
            ),
        ]
    )


model_evaluation_pipeline = create_pipeline()
