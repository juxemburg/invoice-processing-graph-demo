"""Unit tests for invoice_agent.services.normalizer."""

from decimal import Decimal

from invoice_agent.services.normalizer import infer_currency, parse_amount, parse_date


class TestParseAmount:
    def test_usd_with_dollar_sign_and_commas(self):
        # Arrange
        raw = "$6,204.19"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result == Decimal("6204.19")

    def test_gbp_with_pound_sign(self):
        # Arrange
        raw = "£155.45"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result == Decimal("155.45")

    def test_eu_format_space_thousands_comma_decimal(self):
        # Arrange — EU format as seen in batch1-0001.jpg
        raw = "1 394,67"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result == Decimal("1394.67")

    def test_eu_format_dot_thousands_comma_decimal(self):
        # Arrange
        raw = "1.234,56"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result == Decimal("1234.56")

    def test_negative_parentheses(self):
        # Arrange
        raw = "(500.00)"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result == Decimal("-500.00")

    def test_zero_amount(self):
        # Arrange
        raw = "0.00"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result == Decimal("0.00")

    def test_none_input_returns_none(self):
        # Arrange
        raw = None
        # Act
        result = parse_amount(raw)
        # Assert
        assert result is None

    def test_unparseable_returns_none(self):
        # Arrange
        raw = "not a number"
        # Act
        result = parse_amount(raw)
        # Assert
        assert result is None

    def test_empty_string_returns_none(self):
        # Arrange
        raw = ""
        # Act
        result = parse_amount(raw)
        # Assert
        assert result is None


class TestParseDate:
    def test_european_dot_format(self):
        # Arrange — as seen in 052.png
        raw = "26.06.2004"
        # Act
        result = parse_date(raw)
        # Assert
        assert result == "2004-06-26"

    def test_us_slash_format(self):
        # Arrange — as seen in batch1-0001.jpg
        raw = "04/13/2013"
        # Act
        result = parse_date(raw)
        # Assert
        assert result == "2013-04-13"

    def test_iso_format_passes_through(self):
        # Arrange
        raw = "2024-01-15"
        # Act
        result = parse_date(raw)
        # Assert
        assert result == "2024-01-15"

    def test_long_month_name(self):
        # Arrange
        raw = "January 15, 2024"
        # Act
        result = parse_date(raw)
        # Assert
        assert result == "2024-01-15"

    def test_unparseable_returns_none(self):
        # Arrange
        raw = "garbage date"
        # Act
        result = parse_date(raw)
        # Assert
        assert result is None

    def test_none_input_returns_none(self):
        # Arrange
        raw = None
        # Act
        result = parse_date(raw)
        # Assert
        assert result is None

    def test_leading_trailing_whitespace_stripped(self):
        # Arrange
        raw = "  26.06.2004  "
        # Act
        result = parse_date(raw)
        # Assert
        assert result == "2004-06-26"

    def test_slash_date_ambiguity_assumes_us_format(self):
        # Arrange — documents the MM/DD assumption
        raw = "04/03/2013"
        # Act
        result = parse_date(raw)
        # Assert
        assert result == "2013-04-03"  # April 3rd (MM/DD assumed)


class TestInferCurrency:
    def test_infer_gbp_from_pound_symbol_in_total(self):
        # Arrange
        currency_raw, total_raw = None, "£155.45"
        # Act
        result = infer_currency(currency_raw, total_raw)
        # Assert
        assert result == "GBP"

    def test_infer_gbp_from_explicit_code(self):
        # Arrange
        currency_raw, total_raw = "GBP", "155.45"
        # Act
        result = infer_currency(currency_raw, total_raw)
        # Assert
        assert result == "GBP"

    def test_infer_usd_from_dollar_symbol(self):
        # Arrange
        currency_raw, total_raw = None, "$6,204.19"
        # Act
        result = infer_currency(currency_raw, total_raw)
        # Assert
        assert result == "USD"

    def test_infer_eur_from_euro_symbol(self):
        # Arrange
        currency_raw, total_raw = "€", "100.00"
        # Act
        result = infer_currency(currency_raw, total_raw)
        # Assert
        assert result == "EUR"

    def test_returns_none_when_no_hint(self):
        # Arrange
        currency_raw, total_raw = None, "155.45"
        # Act
        result = infer_currency(currency_raw, total_raw)
        # Assert
        assert result is None
