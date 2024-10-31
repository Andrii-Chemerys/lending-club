"""
This is a boilerplate pipeline 'outliers'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import outliers_handler

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=outliers_handler,
            inputs= ['intermediate_lc_clean', 'parameters'],
            outputs= 'intermediate_outl_log',
            name='outliers_handler_node',
            tags='Intermediate'
        )
    ]) # type: ignore
