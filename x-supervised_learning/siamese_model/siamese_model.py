import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the data
graphlet_df = pd.read_csv('D:/Škola/DP1/project/a_dp/input/group_big_out/graphlet_counts.csv', index_col=0)
similarity_df = pd.read_csv('D:/Škola/DP1/project/a_dp/input/group_big_out/filtered_measures.csv')

# Prepare a dictionary mapping graph names to their embeddings
graph_embeddings = {}
for col in graphlet_df.columns:
    graph_embeddings[col] = graphlet_df[col].values


# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()

        # Shared network for both inputs
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(32 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        return self.shared_network(x)

    def forward(self, input1, input2):
        # Get embeddings for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        # Concatenate embeddings
        concat = torch.cat((output1, output2), dim=1)

        # Predict similarity
        similarity = self.fc(concat)
        return similarity


# Custom dataset
class GraphSimilarityDataset(Dataset):
    def __init__(self, similarity_df, graph_embeddings):
        self.similarity_df = similarity_df
        self.graph_embeddings = graph_embeddings

        # Convert boolean labels to integers
        self.similarity_df['label'] = self.similarity_df['Final_Result'].map({True: 1, False: 0})

    def __len__(self):
        return len(self.similarity_df)

    def __getitem__(self, idx):
        row = self.similarity_df.iloc[idx]
        graph1_name = row['Graph1']
        graph2_name = row['Graph2']
        label = row['label']

        # Get embeddings for both graphs
        graph1_embedding = self.graph_embeddings.get(graph1_name)
        graph2_embedding = self.graph_embeddings.get(graph2_name)

        # Convert to tensors
        graph1_tensor = torch.FloatTensor(graph1_embedding)
        graph2_tensor = torch.FloatTensor(graph2_embedding)
        label_tensor = torch.FloatTensor([label])

        return graph1_tensor, graph2_tensor, label_tensor


# Training function
def train_siamese_network(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for graph1, graph2, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(graph1, graph2)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for graph1, graph2, labels in val_loader:
                outputs = model(graph1, graph2)
                val_loss += criterion(outputs, labels).item()

                # Convert probabilities to binary predictions
                predictions = (outputs >= 0.5).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate validation metrics
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_f1 = f1_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions)
        val_recall = recall_score(all_labels, all_predictions)

        val_accuracies.append(val_accuracy)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_siamese_model.pth')

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.4f}')
            print(f'  Val F1 Score: {val_f1:.4f}')
            print(f'  Val Precision: {val_precision:.4f}')
            print(f'  Val Recall: {val_recall:.4f}')

    return train_losses, val_losses, val_accuracies


# Prepare dataset and dataloaders
def prepare_data(similarity_df, graph_embeddings, batch_size=32):
    # Split into train and validation sets
    train_df, val_df = train_test_split(similarity_df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = GraphSimilarityDataset(train_df, graph_embeddings)
    val_dataset = GraphSimilarityDataset(val_df, graph_embeddings)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


# Main execution
def main():

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    print(f"Using device: {device}")

    # Prepare dataloaders
    train_loader, val_loader = prepare_data(similarity_df, graph_embeddings)

    # Initialize model
    input_dim = len(next(iter(graph_embeddings.values())))
    model = SiameseNetwork(input_dim).to(device)

    # Train model
    train_losses, val_losses, val_accuracies = train_siamese_network(
        model, train_loader, val_loader, num_epochs=50
    )

    # Load best model
    model.load_state_dict(torch.load('best_siamese_model.pth'))

    # Evaluate on validation set
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for graph1, graph2, labels in val_loader:
            outputs = model(graph1, graph2)
            predictions = (outputs >= 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()