"""
Tests for Report Generator Module
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.core.report_generator import (
    ReportGenerator,
    format_number,
    format_percent,
    format_price,
    translate_signal,
    translate_trend,
    get_confidence_interpretation,
    calculate_price_position,
    get_text,
)


class TestFormatFunctions:
    """Test formatting utility functions."""

    def test_format_number_trillions(self):
        """Test formatting numbers in trillions."""
        result = format_number(3780000000000, '$')
        assert result == "$3.78T"

    def test_format_number_billions(self):
        """Test formatting numbers in billions."""
        result = format_number(416160000000, '$')
        assert result == "$416.16B"

    def test_format_number_millions(self):
        """Test formatting numbers in millions."""
        result = format_number(78860000, '$')
        assert result == "$78.86M"

    def test_format_number_thousands(self):
        """Test formatting numbers in thousands."""
        result = format_number(5000, '$')
        assert result == "$5.00K"

    def test_format_number_none(self):
        """Test formatting None value."""
        result = format_number(None)
        assert result == "N/A"

    def test_format_percent_positive(self):
        """Test formatting positive percentage."""
        result = format_percent(15.5)
        assert result == "+15.50%"

    def test_format_percent_negative(self):
        """Test formatting negative percentage."""
        result = format_percent(-11.47)
        assert result == "-11.47%"

    def test_format_percent_none(self):
        """Test formatting None percentage."""
        result = format_percent(None)
        assert result == "N/A"

    def test_format_price(self):
        """Test price formatting."""
        result = format_price(255.52)
        assert result == "$255.52"

    def test_format_price_none(self):
        """Test price formatting with None."""
        result = format_price(None)
        assert result == "N/A"


class TestTranslationFunctions:
    """Test translation functions."""

    def test_translate_signal_buy_english(self):
        """Test signal translation to English."""
        result = translate_signal("SIGNALSTRENGTH.BUY", "en")
        assert result == "BUY"

    def test_translate_signal_buy_chinese(self):
        """Test signal translation to Chinese."""
        result = translate_signal("SIGNALSTRENGTH.BUY", "zh")
        assert result == "买入"

    def test_translate_trend_bullish_english(self):
        """Test trend translation to English."""
        result = translate_trend("TRENDDIRECTION.STRONG_BULLISH", "en")
        assert result == "STRONG BULLISH"

    def test_translate_trend_bullish_chinese(self):
        """Test trend translation to Chinese."""
        result = translate_trend("TRENDDIRECTION.STRONG_BULLISH", "zh")
        assert result == "强烈看涨"

    def test_get_text_english(self):
        """Test text retrieval in English."""
        result = get_text('current_price', 'en')
        assert result == "Current Price"

    def test_get_text_chinese(self):
        """Test text retrieval in Chinese."""
        result = get_text('current_price', 'zh')
        assert result == "当前价格"


class TestConfidenceInterpretation:
    """Test confidence score interpretation."""

    def test_high_confidence(self):
        """Test high confidence interpretation."""
        result = get_confidence_interpretation(0.75, 'en')
        assert "High confidence" in result

    def test_medium_confidence(self):
        """Test medium confidence interpretation."""
        result = get_confidence_interpretation(0.55, 'en')
        assert "Medium confidence" in result

    def test_low_confidence(self):
        """Test low confidence interpretation."""
        result = get_confidence_interpretation(0.45, 'en')
        assert "Low confidence" in result

    def test_very_low_confidence(self):
        """Test very low confidence interpretation."""
        result = get_confidence_interpretation(0.30, 'en')
        assert "Very low confidence" in result


class TestPricePosition:
    """Test price position calculation."""

    def test_calculate_price_position_mid_range(self):
        """Test price position at mid-range."""
        result = calculate_price_position(
            current=225.0,
            high=275.0,
            low=175.0
        )
        assert result['position'] == 50.0
        assert result['from_high'] < 0
        assert result['from_low'] > 0

    def test_calculate_price_position_at_high(self):
        """Test price position at 52-week high."""
        result = calculate_price_position(
            current=275.0,
            high=275.0,
            low=175.0
        )
        assert result['position'] == 100.0
        assert result['from_high'] == 0.0

    def test_calculate_price_position_none_values(self):
        """Test price position with None values."""
        result = calculate_price_position(None, 275.0, 175.0)
        assert result['position'] is None


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_init_english(self):
        """Test initialization with English language."""
        gen = ReportGenerator(lang='en')
        assert gen.lang == 'en'

    def test_init_chinese(self):
        """Test initialization with Chinese language."""
        gen = ReportGenerator(lang='zh')
        assert gen.lang == 'zh'

    def test_t_shortcut(self):
        """Test translation shortcut method."""
        gen = ReportGenerator(lang='en')
        result = gen.t('market_data')
        assert result == "Market Overview"

    def test_extract_data_empty_result(self):
        """Test data extraction from empty result."""
        gen = ReportGenerator()
        mock_result = Mock()
        mock_result.raw = {}

        data = gen.extract_data(mock_result)

        assert data['market_data'] == {}
        assert data['technical'] == {}

    def test_extract_data_with_market_data(self):
        """Test data extraction with market data."""
        gen = ReportGenerator()

        mock_result = Mock()
        mock_result.raw = {
            'results': {
                'market_data': {
                    'current_price': 255.52,
                    'market_cap': 3780000000000,
                    'pe_ratio': 34.25
                }
            }
        }

        data = gen.extract_data(mock_result)

        assert data['market_data']['current_price'] == 255.52
        assert data['market_data']['market_cap'] == 3780000000000

    def test_generate_terminal_summary(self):
        """Test terminal summary generation."""
        gen = ReportGenerator(lang='en')

        mock_result = Mock()
        mock_result.raw = {
            'results': {
                'market_data': {
                    'current_price': 255.52,
                    'market_cap': 3780000000000,
                    'pe_ratio': 34.25
                }
            }
        }

        summary = gen.generate_terminal_summary(mock_result, 'AAPL')

        assert 'AAPL' in summary
        assert '$255.52' in summary
        assert '$3.78T' in summary

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        gen = ReportGenerator(lang='en')

        mock_result = Mock()
        mock_result.raw = {
            'results': {
                'market_data': {
                    'current_price': 255.52,
                    'market_cap': 3780000000000,
                    'pe_ratio': 34.25,
                    'fifty_two_week_high': 288.62,
                    'fifty_two_week_low': 169.21
                }
            }
        }

        report = gen.generate_markdown_report(mock_result, 'AAPL')

        assert '# AAPL' in report
        assert 'Investment Analysis Report' in report
        assert 'Executive Summary' in report
        assert '$255.52' in report
        assert 'DISCLAIMER' in report
        assert 'Yahoo Finance' in report


class TestBilingualSupport:
    """Test bilingual support in reports."""

    def test_english_report_headers(self):
        """Test that English report uses English headers."""
        gen = ReportGenerator(lang='en')
        mock_result = Mock()
        mock_result.raw = {'results': {'market_data': {'current_price': 100}}}

        report = gen.generate_markdown_report(mock_result, 'TEST')

        assert 'Market Overview' in report
        assert 'Executive Summary' in report

    def test_chinese_report_headers(self):
        """Test that Chinese report uses Chinese headers."""
        gen = ReportGenerator(lang='zh')
        mock_result = Mock()
        mock_result.raw = {'results': {'market_data': {'current_price': 100}}}

        report = gen.generate_markdown_report(mock_result, 'TEST')

        assert '市场概览' in report
        assert '摘要' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
