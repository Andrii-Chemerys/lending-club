"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import processing_lc

# Define pipeline for processing raw 'original_lc_dataset' to 'intermediate_lc_dataset'
# (see conf/base/catalog.yml) with node.processing_lc() function that change specific
# features to proper type
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=processing_lc,
            inputs='original_lc_dataset',
            outputs='intermediate_lc_dataset',
        )
    ])
