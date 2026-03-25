"""
data_pipeline — Dataset synthesis and validation package.

Phases covered:
    1. Generator  : Calls an external LLM API to produce (dialog, DialogState) pairs.
    2. Validator  : Checks each pair against the Pydantic schema and heuristics.
    3. DatasetBuilder: Assembles validated pairs into HuggingFace Dataset objects.
"""
