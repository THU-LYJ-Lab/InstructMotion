from .value_policy_heads import AutoModelForSeq2SeqLMWithValueHead, AutoModelForSeq2SeqLMSepValueHead, AutoModelForSeq2SeqLMSepPolicyHead

SUPPORTED_ARCHITECTURES = (
    AutoModelForSeq2SeqLMSepValueHead,
    AutoModelForSeq2SeqLMSepPolicyHead,
    AutoModelForSeq2SeqLMWithValueHead,
)