from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load dataset
data = fetch_olivetti_faces()

X = data.data
y = data.target

# Split dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
joblib.dump(model, "model/savedmodel.pth")

print("Model trained successfully.")
print("Model saved as model/savedmodel.pth")