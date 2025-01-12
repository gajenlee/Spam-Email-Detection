import pandas as pd

data = pd.read_csv("spam_ham_dataset.csv")
new_excel = pd.ExcelWriter("spam_ham_dataset.xlsx")
data.to_excel(new_excel, index=False)
new_excel.save()