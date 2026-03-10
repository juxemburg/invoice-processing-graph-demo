"""Unit tests for pure helper functions in invoice_agent.services.extractor."""

from pathlib import Path

from invoice_agent.services.extractor import get_mime_type, read_image_bytes


class TestGetMimeType:
    def test_jpg_extension(self):
        # Arrange
        path = Path("invoice.jpg")
        # Act
        result = get_mime_type(path)
        # Assert
        assert result == "image/jpeg"

    def test_jpeg_extension(self):
        # Arrange
        path = Path("invoice.jpeg")
        # Act
        result = get_mime_type(path)
        # Assert
        assert result == "image/jpeg"

    def test_png_extension(self):
        # Arrange
        path = Path("scan.png")
        # Act
        result = get_mime_type(path)
        # Assert
        assert result == "image/png"

    def test_pdf_extension(self):
        # Arrange
        path = Path("invoice.pdf")
        # Act
        result = get_mime_type(path)
        # Assert
        assert result == "application/pdf"

    def test_uppercase_extension_handled(self):
        # Arrange
        path = Path("INVOICE.JPG")
        # Act
        result = get_mime_type(path)
        # Assert
        assert result == "image/jpeg"

    def test_unknown_extension_defaults_to_png(self):
        # Arrange
        path = Path("weird.bmp")
        # Act
        result = get_mime_type(path)
        # Assert
        assert result == "image/png"


class TestReadImageBytes:
    def test_returns_file_contents(self, tmp_path: Path):
        # Arrange
        test_file = tmp_path / "test.png"
        expected = b"\x89PNG\r\n\x1a\n"
        test_file.write_bytes(expected)
        # Act
        result = read_image_bytes(test_file)
        # Assert
        assert result == expected
