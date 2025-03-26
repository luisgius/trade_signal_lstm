import ta
import pandas as pd

def add_technical_indicators(df):

    #Copy of df
    df = df.copy()

    df.columns = [tuple(str(c).lower() for c in col) if isinstance(col, tuple) else str(col).lower() for col in df.columns]  # Fixes typo AND handles tuples
    
    # Convert tuple columns to proper MultiIndex FIRST
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Now safely extract OHCLV names
    df.columns = df.columns.get_level_values(0).str.lower()
    print(df)
    #Calculate Indicators with ta
    #EMA
    df["ema_10"] = ta.trend.EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_30"] = ta.trend.EMAIndicator(close=df["close"], window=30).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(close=df["close"], window=100).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close=df["close"], window=200).ema_indicator()

    #MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    #RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    #Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df["high"],df["low"],df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    #Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df["close"])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df['bb_low']) / df['bb_mid']

    #atr
    df["atr"] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    #on-Balance Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    #returns
    df["returns_1d"] = df["close"].pct_change()
    df["returns_5d"] = df["close"].pct_change(5)

    #Add target
    df["target"] = df["close"].pct_change().shift(-1)

    # Drop NaN values
    df = df.dropna()

    return df