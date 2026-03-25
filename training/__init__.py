"""
training -- LoRA fine-tuning and evaluation package (Phase 2).

Modules:
    lora_trainer        : Configure and run PEFT LoRA fine-tuning on Llama-3B.
    evaluator           : Measure exact-match JSON accuracy and schema validity rate.
    constrained_decoder : Enforce JSON structure via grammar-based decoding at inference.
"""
