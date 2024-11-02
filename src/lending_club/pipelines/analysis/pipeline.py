"""
This is a boilerplate pipeline 'analysis'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import features_eng, eda_df

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=features_eng,
            inputs=['intermediate_lc_clean', 'primary_analysis', 'parameters'],
            outputs='features_new',
            name='features_eng_node',
            tags='Primary'
        ),
        node(
            func=eda_df,
            inputs=['intermediate_lc_clean', 'intermediate_encoded', 'features_new', 'parameters'],
            outputs='primary_analysis',
            name='eda_df_node',
            tags='Primary'
        )
    ]) # type: ignore
