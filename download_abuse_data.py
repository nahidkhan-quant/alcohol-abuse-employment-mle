import wooldridge as woo

# Load the Wooldridge alcohol dataset
df = woo.data("alcohol")

# Check the dataset
print(df.head())
print(df.shape)
print(df.columns)

# Save it as a CSV file
df.to_csv("Abuse.csv", index=False)

print("Abuse.csv was created successfully.")