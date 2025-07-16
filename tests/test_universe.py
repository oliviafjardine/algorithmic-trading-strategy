import pandas as pd
import numpy as np
import pytest
from src.core.universe_filter import filter_top_liquid


class TestFilterTopLiquid:
    """Test suite for filter_top_liquid function"""
    
    @pytest.fixture
    def sample_data_multiindex(self):
        """Create sample data with MultiIndex (date, ticker)"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        # Generate sample data
        np.random.seed(42)
        n_rows = len(index)
        
        data = pd.DataFrame({
            'adj close': np.random.uniform(100, 300, n_rows),
            'volume': np.random.randint(1000000, 10000000, n_rows),
            'open': np.random.uniform(100, 300, n_rows),
            'high': np.random.uniform(100, 300, n_rows),
            'low': np.random.uniform(100, 300, n_rows),
        }, index=index)
        
        return data
    
    @pytest.fixture
    def sample_data_columns(self):
        """Create sample data with date and ticker as columns"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # Create all combinations
        data_list = []
        np.random.seed(42)
        
        for date in dates:
            for ticker in tickers:
                data_list.append({
                    'date': date,
                    'ticker': ticker,
                    'adj close': np.random.uniform(100, 300),
                    'volume': np.random.randint(1000000, 10000000),
                    'open': np.random.uniform(100, 300),
                    'high': np.random.uniform(100, 300),
                    'low': np.random.uniform(100, 300),
                })
        
        return pd.DataFrame(data_list)
    
    @pytest.fixture
    def sample_data_with_dollar_volume(self):
        """Create sample data that already has dollar_volume column"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        np.random.seed(42)
        n_rows = len(index)
        
        data = pd.DataFrame({
            'adj close': np.random.uniform(100, 300, n_rows),
            'volume': np.random.randint(1000000, 10000000, n_rows),
            'dollar_volume': np.random.uniform(1e8, 1e10, n_rows),  # Pre-existing dollar volume
        }, index=index)
        
        return data
    
    def test_basic_functionality_multiindex(self, sample_data_multiindex):
        """Test basic functionality with MultiIndex data"""
        result = filter_top_liquid(sample_data_multiindex, n=3, window=5)
        
        # Check that result has the expected structure
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['date', 'ticker']
        
        # Check that new columns are added
        assert 'dollar_volume' in result.columns
        assert 'rolling_dollar_vol' in result.columns
        assert 'dollar_vol_rank' in result.columns
        
        # Check that at most n tickers per date (after rolling window)
        for date in result.index.get_level_values('date').unique():
            date_data = result.loc[date]
            if not date_data.empty:
                assert len(date_data) <= 3
    
    def test_basic_functionality_columns(self, sample_data_columns):
        """Test basic functionality with date/ticker as columns"""
        result = filter_top_liquid(sample_data_columns, n=3, window=5)
        
        # Check that result has the expected structure
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'ticker' in result.columns
        assert 'dollar_volume' in result.columns
        assert 'rolling_dollar_vol' in result.columns
        assert 'dollar_vol_rank' in result.columns
        
        # Check that at most n tickers per date (after rolling window)
        for date in result['date'].unique():
            date_data = result[result['date'] == date]
            if not date_data.empty:
                assert len(date_data) <= 3
    
    def test_dollar_volume_calculation(self, sample_data_multiindex):
        """Test that dollar volume is calculated correctly"""
        result = filter_top_liquid(sample_data_multiindex, n=5, window=5)
        
        # Check that dollar_volume = adj close * volume
        expected_dollar_vol = sample_data_multiindex['adj close'] * sample_data_multiindex['volume']
        
        # Compare values where both exist (accounting for filtering)
        for idx in result.index:
            if idx in expected_dollar_vol.index:
                assert abs(result.loc[idx, 'dollar_volume'] - expected_dollar_vol.loc[idx]) < 1e-6
    
    def test_existing_dollar_volume(self, sample_data_with_dollar_volume):
        """Test that existing dollar_volume column is preserved"""
        original_dollar_vol = sample_data_with_dollar_volume['dollar_volume'].copy()
        
        result = filter_top_liquid(sample_data_with_dollar_volume, n=2, window=3)
        
        # Check that original dollar_volume values are preserved
        for idx in result.index:
            if idx in original_dollar_vol.index:
                assert result.loc[idx, 'dollar_volume'] == original_dollar_vol.loc[idx]
    
    def test_ranking_logic(self, sample_data_multiindex):
        """Test that ranking works correctly"""
        result = filter_top_liquid(sample_data_multiindex, n=2, window=5)
        
        # Check that ranks are 1 and 2 for each date (where data exists)
        for date in result.index.get_level_values('date').unique():
            date_data = result.loc[date]
            if not date_data.empty:
                ranks = sorted(date_data['dollar_vol_rank'].values)
                expected_ranks = list(range(1, len(date_data) + 1))
                assert ranks == expected_ranks
                assert max(ranks) <= 2  # Since n=2
    
    def test_window_parameter(self, sample_data_multiindex):
        """Test different window sizes"""
        result_short = filter_top_liquid(sample_data_multiindex, n=3, window=3)
        result_long = filter_top_liquid(sample_data_multiindex, n=3, window=10)
        
        # Both should have the same columns
        assert set(result_short.columns) == set(result_long.columns)
        
        # Rolling averages should be different
        assert not result_short['rolling_dollar_vol'].equals(result_long['rolling_dollar_vol'])
    
    def test_n_parameter(self, sample_data_multiindex):
        """Test different values of n"""
        result_small = filter_top_liquid(sample_data_multiindex, n=2, window=5)
        result_large = filter_top_liquid(sample_data_multiindex, n=4, window=5)
        
        # Larger n should have more or equal rows
        assert len(result_large) >= len(result_small)
        
        # Check that n is respected per date
        for date in result_small.index.get_level_values('date').unique():
            small_count = len(result_small.loc[date]) if date in result_small.index else 0
            large_count = len(result_large.loc[date]) if date in result_large.index else 0
            assert small_count <= 2
            assert large_count <= 4
    
    def test_different_price_columns(self, sample_data_multiindex):
        """Test using different price columns"""
        # Test with 'open' instead of 'adj close'
        result = filter_top_liquid(sample_data_multiindex, n=3, window=5, price_col='open')
        
        # Check that dollar_volume uses 'open' prices
        expected_dollar_vol = sample_data_multiindex['open'] * sample_data_multiindex['volume']
        
        for idx in result.index:
            if idx in expected_dollar_vol.index:
                assert abs(result.loc[idx, 'dollar_volume'] - expected_dollar_vol.loc[idx]) < 1e-6
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame should raise error due to missing columns"""
        empty_df = pd.DataFrame()
        with pytest.raises(KeyError):
            filter_top_liquid(empty_df, n=3, window=5)
    
    def test_single_ticker(self):
        """Test with single ticker"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        index = pd.MultiIndex.from_product([dates, ['AAPL']], names=['date', 'ticker'])
        
        data = pd.DataFrame({
            'adj close': np.random.uniform(100, 200, len(index)),
            'volume': np.random.randint(1000000, 5000000, len(index)),
        }, index=index)
        
        result = filter_top_liquid(data, n=3, window=5)
        
        # Should keep all data since there's only one ticker
        assert len(result) <= len(data)  # May be less due to rolling window
        assert all(result['dollar_vol_rank'] == 1)  # All ranks should be 1
    
    def test_insufficient_data_for_rolling(self):
        """Test with insufficient data for rolling window"""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        tickers = ['AAPL', 'GOOGL']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        data = pd.DataFrame({
            'adj close': np.random.uniform(100, 200, len(index)),
            'volume': np.random.randint(1000000, 5000000, len(index)),
        }, index=index)
        
        # Use window larger than available data
        result = filter_top_liquid(data, n=2, window=10)
        
        # Should handle gracefully (may have NaN values for early periods)
        assert isinstance(result, pd.DataFrame)
        assert 'rolling_dollar_vol' in result.columns
    
    def test_data_types_preserved(self, sample_data_multiindex):
        """Test that original data types are preserved"""
        result = filter_top_liquid(sample_data_multiindex, n=3, window=5)
        
        # Check that original columns maintain their types
        for col in sample_data_multiindex.columns:
            if col in result.columns:
                assert result[col].dtype == sample_data_multiindex[col].dtype
    
    def test_no_volume_column_raises_error(self):
        """Test that missing volume column raises appropriate error"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        tickers = ['AAPL', 'GOOGL']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        data = pd.DataFrame({
            'adj close': np.random.uniform(100, 200, len(index)),
            # Missing 'volume' column
        }, index=index)
        
        with pytest.raises(KeyError):
            filter_top_liquid(data, n=3, window=5)
    
    def test_no_price_column_raises_error(self):
        """Test that missing price column raises appropriate error"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        tickers = ['AAPL', 'GOOGL']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        data = pd.DataFrame({
            'volume': np.random.randint(1000000, 5000000, len(index)),
            # Missing 'adj close' column
        }, index=index)
        
        with pytest.raises(KeyError):
            filter_top_liquid(data, n=3, window=5, price_col='adj close')


if __name__ == "__main__":
    pytest.main([__file__])