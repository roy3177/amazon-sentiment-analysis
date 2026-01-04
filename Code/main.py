
import pandas as pd
import re

# For TF-IDF:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# For Pytorch:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# For PrettyTable:
from prettytable import PrettyTable


def rule_based_predict(text, pos_words, neg_words):
     """
     This method get text(Review) , and positive+negative words, and return:
     1 if its positive, 0 if its negative
     """

     # Lowercase + keep only letters and spaces:
     text = text.lower()
     text=re.sub(r"[^a-z\s]", " ", text)

     words=text.split()

     # Counting positive and negative words:
     pos_count=sum(1 for w in words if w in pos_words)
     neg_count=sum(1 for w in words if w in neg_words)

     # How we decide positive/negative:
     if pos_count>=neg_count:
         return 1
     return 0

def load_fast_text_file(path, n_rows=None):
    """The goal of the method is to read the fastText file, to take
    every row and convert it to text and label, and return organize DataFrame """

    texts=[]
    labels=[]

    # Open the file:
    with open(path, 'r', encoding="utf-8") as f:
        for i,line in enumerate(f):
            if n_rows is not None and i>=n_rows:
                break

            line=line.strip()
            if not line :
                continue

            # Split the line to words:
            parts=line.split(" ")
            label_token=parts[0]  # Give us __label__2 or __label__1

            text = " ".join(parts[1:]) #  take all the sentence without the label

            # Mapping the labels:
            if label_token=="__label__2":
                label=1
            else:
                label=0

            # Putting every line to our future DataFrame:
            texts.append(text)
            labels.append(label)

    # Create the DataFrame:
    df=pd.DataFrame({"text":texts,"label":labels})
    return df




def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "CM": confusion_matrix(y_true, y_pred)
    }

def sparse_to_torch_tensor(x_sparse):
    """
    This method convert sparse tensor(TF-IDF) to torch tensor:
    TfidVectorizer return Sparse Matrix(compassed at the memory)
    So our simple Pytorch model except to get regular Tensor(Dense)
    So our levels: sparse-->dense-->torch tensor
    """

    x_dense=x_sparse.toarray()
    return torch.tensor(x_dense, dtype=torch.float32)

class FFNNSentiment(nn.Module):
    """
    Define Fully Connected Neural Network model
    This is our Pytorch model:
    Fully Connected=Linear layers.
    ReLU adds non-linearity
    Dropout helps against overfitting
    The output is 1 because is binary classification

    """
    def __init__(self, input_dim=10000, hidden_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

if __name__=="__main__":

    torch.manual_seed(42)
    train_path="../Data/train.ft.txt"
    train_df=load_fast_text_file(train_path, n_rows=100000)

    # Dataset overview table:
    data_table=PrettyTable()
    data_table.field_names=["Metric","Value"]

    data_table.add_row(["Number of rows", len(train_df)])
    data_table.add_row(["Positive (1)", int((train_df["label"]==1).sum())])
    data_table.add_row(["Negative (0)", int((train_df["label"]==0).sum())])


    print("\n=== Dataset Overview ===")
    print(data_table)

    # Show first 5 samples:
    head_table=PrettyTable()
    head_table.field_names=["Index", "Text", "Label"]

    for idx, row in train_df.head().iterrows():
        head_table.add_row([idx, row["text"][:80] + "...", row["label"]])

    print("\n=== First 5 Samples ===")
    print(head_table)


    """
    ------------------------------
    Baseline 1 : Majority class
    ------------------------------
    """

    # Find the most common label ( 0 or 1):
    majority_label=train_df['label'].value_counts().idxmax()

    # Accuracy if we always predict the majority class:
    baseline_accuracy=(train_df["label"]==majority_label).mean()



    """
    ------------------------------
    Baseline 2 : Rule-Based (keywords)
    ------------------------------
    """

    pos_words = {
        "good", "great", "excellent", "amazing", "love", "loved", "awesome",
        "perfect", "best", "wonderful", "fantastic", "nice", "happy", "enjoy"
    }

    neg_words = {
        "bad", "terrible", "awful", "worst", "hate", "hated", "refund",
        "disappointed", "poor", "waste", "broken", "boring", "problem"
    }

    # Running for each review (rule_preds create list of 1/0 for every row) :
    rule_preds=train_df["text"].apply(lambda t: rule_based_predict(t, pos_words, neg_words))

    # Getting the Accuracy if Baseline 2:
    rule_accuracy=(rule_preds==train_df["label"]).mean()

    baseline_table = PrettyTable()
    baseline_table.field_names = ["Baseline", "Details", "Accuracy"]

    baseline_table.add_row([
        "Majority Class",
        majority_label,
        f"{baseline_accuracy:.4f}"
    ])

    baseline_table.add_row([
        "Rule-Based",
        "Keyword matching",
        f"{rule_accuracy:.4f}"
    ])

    print("\n=== Baseline Results ===")
    print(baseline_table)

    """
     ------------------------------
     Train / Validation Split
     ------------------------------
     """

    x=train_df["text"] # x is the input-->the text of the reviews
    y=train_df["label"] # y is the real answers (label)--> 0/1 (positive/negative)

    # Takes the data and divide it to train(80%) and validation(20%):
    x_train, x_validation, y_train, y_validation=train_test_split(x,y,test_size=0.2,random_state=42)

    split_table = PrettyTable()
    split_table.field_names = ["Split", "Size"]

    split_table.add_row(["Train", len(x_train)])
    split_table.add_row(["Validation", len(x_validation)])

    print("\n=== Train / Validation Split ===")
    print(split_table)

    """
     ------------------------------
      Metrics on Validation Set
     ------------------------------
     """
    # Baseline 1: Majority class predictions on validation
    majority_val_pred = [majority_label] * len(y_validation)

    # Baseline 2: Rule-based predictions on validation
    rule_val_pred = x_validation.apply(lambda t: rule_based_predict(t, pos_words, neg_words))

    """
     ------------------------------
     TF-IDF Vectorization
     ------------------------------
     """

    # We convert every review to vector numbers with size of 10,000:
    vectorizer=TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')

    # Train step of the vectorizer:
    x_train_tfidf=vectorizer.fit_transform(x_train)

    # Convertion of the validation set:
    x_validation_tfidf=vectorizer.transform(x_validation)

    tfidf_table = PrettyTable()
    tfidf_table.field_names = ["Info", "Value"]
    tfidf_table.add_row(["TF-IDF train shape", str(x_train_tfidf.shape)])
    tfidf_table.add_row(["TF-IDF validation shape", str(x_validation_tfidf.shape)])

    print("\n=== TF-IDF Shapes ===")
    print(tfidf_table)

    """
     ------------------------------
     Logistic Regression Model
     ------------------------------
     """

    # Creating of the model:
    model = LogisticRegression(max_iter=1000)

    # Train the model--->learning step:
    model.fit(x_train_tfidf, y_train)

    # Predict the validation set:
    y_pred=model.predict(x_validation_tfidf)



    results = []

    results.append(("Baseline 1 (Majority)",
                    get_metrics(y_validation, majority_val_pred)))

    results.append(("Baseline 2 (Rule-Based)",
                    get_metrics(y_validation, rule_val_pred)))

    results.append(("Logistic Regression (TF-IDF)",
                    get_metrics(y_validation, y_pred)))

    """
     ------------------------------
     PyTorch FFNN (TF-IDF)
     ------------------------------
     """

    # Convert TF-IDF(sparse) to torch tensors(dense):
    x_train_tensor=sparse_to_torch_tensor(x_train_tfidf)
    x_validation_tensor=sparse_to_torch_tensor(x_validation_tfidf)

    y_train_tensor=torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_validation_tensor=torch.tensor(y_validation.to_numpy(), dtype=torch.float32)

    # Build datasets + dataloaders (mini-batches):
    # TensorDataset is connecting x to y:
    train_ds=TensorDataset(x_train_tensor,y_train_tensor)
    validation_ds=TensorDataset(x_validation_tensor,y_validation_tensor)

    # DataLoader divide the data for batches (for 256 every time):
    train_loader=DataLoader(train_ds, batch_size=256, shuffle=True)
    validation_loader=DataLoader(validation_ds, batch_size=256, shuffle=False)

    # Create model+loss+optimizer:
    model_nn=FFNNSentiment(input_dim=x_train_tensor.shape[1],hidden_dim=128)

    # Using sigmoid:
    criterion=nn.BCEWithLogitsLoss()

    # Using Adam optimizer:
    optimizer=torch.optim.Adam(model_nn.parameters(), lr=1e-3)

    # Training loop:
    epochs=5

    for epoch in range(epochs):
        model_nn.train() # training mode (Active Dropout)
        total_loss=0.0

        for xb, yb in train_loader:
            optimizer.zero_grad() # Reset gradients from the previous batch

            logits=model_nn(xb) # Forward
            loss=criterion(logits,yb) # Compute loss

            loss.backward()  # Compute derivative (gradients)
            optimizer.step() # update weights

            total_loss+=loss.item()

        avg_loss=total_loss/len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_loss:.4f}")

    #Evaluation on validation:
    model_nn.eval() # Evaluation mode (Dropout off)

    all_preds=[]
    all_true=[]

    with torch.no_grad():
        for xb, yb in validation_loader:
            logits=model_nn(xb)
            probs = torch.sigmoid(logits)  # convert logits -> probabilities 0-1
            preds = (probs >= 0.5).int().numpy()  # threshold to 0/1

            all_preds.extend(preds.tolist())
            all_true.extend(yb.int().numpy().tolist())

    # Add PyTorch results to your summary table
    results.append(("PyTorch FFNN (TF-IDF)",
                    get_metrics(all_true, all_preds)))

    summary_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1-score": m["F1"]
        }
        for name, m in results
    ])

    results_table = PrettyTable()
    results_table.field_names = ["Model", "Accuracy", "Precision", "Recall", "F1-score"]

    for name, m in results:
        results_table.add_row([
            name,
            f"{m['Accuracy']:.4f}",
            f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}",
            f"{m['F1']:.4f}"
        ])

    results_table.align["Model"] = "l"

    print("\n=== Validation Results Summary ===")
    print(results_table)

