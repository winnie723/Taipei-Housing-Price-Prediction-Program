import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Taipei_house.csv")

data = df.copy()

data = data.drop(columns=['交易日期'])

label_cols = ['行政區', '車位類別']
for col in label_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

data = data.fillna(data.median(numeric_only=True))

X = data.drop(columns=['總價'])
y = data['總價']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"平均誤差（MAE）：{mae:.2f} 萬元")
print(f"模型解釋力（R²）：{r2:.3f}")

sample = pd.DataFrame([{
    '行政區': 0,          
    '土地面積': 30,
    '建物總面積': 100,
    '屋齡': 20,
    '樓層': 3,
    '總樓層': 10,
    '用途': 0,
    '房數': 3,
    '廳數': 2,
    '衛數': 2,
    '電梯': 1,
    '車位類別': 0,        
    '經度': 121.55,
    '緯度': 25.03
}])

predicted_price = model.predict(sample)[0]
print(f"預測房價：約 {predicted_price:.0f} 萬元")
