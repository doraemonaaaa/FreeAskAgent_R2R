"""BertConfig shim: re-export the vendored pytorch_transformers BertConfig."""
from .transformer.pytorch_transformer.modeling_bert import BertConfig  # noqa: F401
