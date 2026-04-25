import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


os.makedirs("figures", exist_ok=True)


df = pd.read_csv("A:/train.csv")


df["Sales"] = df["Sales"].astype(str).str.replace("..", "", regex=False)
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")


df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)


df = df.dropna()



def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()
    print(f"Saved: {filename}")


plt.figure()
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
save_plot("sales_distribution.png")


plt.figure()
sns.barplot(x="Category", y="Sales", data=df, estimator=sum)
plt.title("Sales by Category")
save_plot("sales_by_category.png")


plt.figure()
sns.barplot(x="Region", y="Sales", data=df, estimator=sum)
plt.title("Sales by Region")
save_plot("sales_by_region.png")


df["Year"] = df["Order Date"].dt.year

plt.figure()
sns.lineplot(x="Year", y="Sales", data=df, estimator=sum)
plt.title("Sales Trend Over Years")
save_plot("sales_trend.png")


plt.figure()
sns.barplot(x="Segment", y="Sales", data=df, estimator=sum)
plt.title("Sales by Segment")
save_plot("sales_by_segment.png")


top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)




df["Month"] = df["Order Date"].dt.month

X = df[["Month"]]
y = df["Sales"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("\nModel Training Completed")
print("Sample Predictions:", predictions[:5])


print("\n--- Key Insights ---")
print("Total Sales:", round(df["Sales"].sum(), 2))
print("Top Category:", df.groupby("Category")["Sales"].sum().idxmax())
print("Top Region:", df.groupby("Region")["Sales"].sum().idxmax())

print("\nProject Completed. Check 'figures' folder.")