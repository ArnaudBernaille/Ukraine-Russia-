import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

"""
Data gathering
"""

russian_loss_df = pd.read_csv('data/russia_losses_equipment.csv')
russian_loss_columns = ['aircraft', 'helicopter', 'tank', 'naval ship']
russian_loss_df = russian_loss_df[['date'] + russian_loss_columns]


def convert_to_absolute(df):
    df_absolute = df.copy()
    for column in russian_loss_columns:
        df[column] = df[column].astype(int)
        df_absolute[column] = df[column].diff().abs()
    return df_absolute


russian_loss_df = convert_to_absolute(russian_loss_df)

nat_gas_prices = pd.read_csv('data/nat_gas.csv', delimiter='\t', header=None, names=['date', 'gasPrice'])
nat_gas_prices['date'] = pd.to_datetime(nat_gas_prices['date'], format='%b %d %Y')
nat_gas_prices['date'] = nat_gas_prices['date'].dt.strftime('%Y-%m-%d')

merged_df = pd.merge(russian_loss_df, nat_gas_prices, on='date', how='inner')

"""
Correlation matrix
"""

correlation_matrix = merged_df[russian_loss_columns + ['gasPrice']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')


"""
IA Model
"""

df_ia = merged_df[russian_loss_columns + ['gasPrice']]
y = df_ia['gasPrice']
X = df_ia.drop('gasPrice', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE : {mse}')
print(f'R^2 : {r2}')
