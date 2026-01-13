"""Basic import tests to verify package structure."""

import pytest


class TestImports:
    """Test that all public modules can be imported."""

    def test_import_main_package(self):
        """Test importing the main package."""
        import reasoning_models_lie

        assert hasattr(reasoning_models_lie, "__version__")
        assert reasoning_models_lie.__version__ == "0.1.0"

    def test_import_client(self):
        """Test importing the reasoning model client."""
        from reasoning_models_lie import ReasoningModelClientLangChain, LangChainModelType

        assert ReasoningModelClientLangChain is not None
        assert LangChainModelType is not None

    def test_import_exceptions(self):
        """Test importing exception classes."""
        from reasoning_models_lie import (
            APIError,
            PromptingError,
            ResponseParsingError,
            ValidationError,
        )

        assert APIError is not None
        assert PromptingError is not None
        assert ResponseParsingError is not None
        assert ValidationError is not None



    def test_model_types_available(self):
        """Test that expected model types are available."""
        from reasoning_models_lie import LangChainModelType

        assert hasattr(LangChainModelType, "CLAUDE_4_5_SONNET")
        assert hasattr(LangChainModelType, "CLAUDE_4_5_HAIKU")
        assert hasattr(LangChainModelType, "CLAUDE_3_7_SONNET")
        assert hasattr(LangChainModelType, "DEEPSEEK_R1")


class TestDataLoaders:
    """Test that data loader modules can be imported."""

    def test_import_gpqa_loader(self):
        """Test importing GPQA data loader."""
        from reasoning_models_lie.data_loaders import gpqa

        assert hasattr(gpqa, "GPQA_SPLITS")

    def test_import_mmlu_pro_loader(self):
        """Test importing MMLU-Pro data loader."""
        from reasoning_models_lie.data_loaders import mmlu_pro

        assert hasattr(mmlu_pro, "MMLU_PRO_SPLITS")

    def test_import_constants(self):
        """Test importing data loader constants."""
        from reasoning_models_lie.data_loaders.constants import (
            HINT_TYPES,
            SETTINGS,
            TASKS,
        )

        assert isinstance(HINT_TYPES, (list, tuple, set, frozenset))
        assert isinstance(SETTINGS, (list, tuple, set, frozenset))
        assert isinstance(TASKS, (list, tuple, set, frozenset))


class TestEvaluation:
    """Test that evaluation modules can be imported."""

    def test_import_multiple_choice(self):
        """Test importing multiple choice evaluation."""
        from reasoning_models_lie.evaluation import multiple_choice

        assert multiple_choice is not None

    def test_import_faithfulness(self):
        """Test importing faithfulness evaluation."""
        from reasoning_models_lie.evaluation import faithfulness

        assert faithfulness is not None
