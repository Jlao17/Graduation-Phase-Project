import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("models/Results/Train/CALCITE_Epoch_Metrics.csv")

eval_losses = []
train_losses = []
for epoch_value in df['Epoch'].unique():
    # Step 3: Filter rows where 'epoch' equals the current value
    epoch_rows = df[df['Epoch'] == epoch_value]
    eval_loss = epoch_rows[epoch_rows["Metrics"] == "eval_loss"]
    train_loss = epoch_rows[epoch_rows["Metrics"] == "train_loss"]
    eval_losses.append(eval_loss["Score"].astype(float))
    train_losses.append(train_loss["Score"].astype(float))

# Step 4: Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df['Epoch'].unique(), eval_losses, label='Evaluation Loss')
plt.plot(df['Epoch'].unique(), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()

