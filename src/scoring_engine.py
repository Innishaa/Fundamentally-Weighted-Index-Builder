from sklearn.preprocessing import MinMaxScaler

#invert ratios where lower is better
#PE and DE ratios are inverted because lower values are better
#PE: Price to Earnings ratio, DE: Debt to Equity ratio
def score_stocks(df):
    df['inv_PE'] = 1 / df['PE']
    df['inv_DE'] = 1 / df['DE']

    # Normalize the scores
    scaler = MinMaxScaler()
    df[['norm_ROE', 'norm_inv_PE', 'norm_inv_DE']] = scaler.fit_transform(
        df[['ROE', 'inv_PE', 'inv_DE']]
    )

    #Composite score calculation
    #Weights: ROE 40%, inverted PE 30%, inverted DE 30%
    df['Score'] = (
        0.4 * df['norm_ROE'] +
        0.3 * df['norm_inv_PE'] +
        0.3 * df['norm_inv_DE']
    )

    return df.sort_values(by='Score', ascending=False)
