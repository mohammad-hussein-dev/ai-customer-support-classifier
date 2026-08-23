"""Unit tests for text preprocessing."""

from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    def test_default_initialization(self):
        preprocessor = TextPreprocessor()
        assert preprocessor.lowercase is True
        assert preprocessor.remove_punctuation is True

    def test_clean_lowercase(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean("HELLO WORLD")
        assert result == "hello world"

    def test_clean_punctuation_removal(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean("Hello, world!!!")
        assert "," not in result
        assert "!" not in result

    def test_clean_stopword_removal(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean("this is a test")
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_transform_list(self):
        preprocessor = TextPreprocessor()
        texts = ["Hello world", "Test sentence"]
        results = preprocessor.transform(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_non_string_input(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.clean(12345)
        assert result == ""
