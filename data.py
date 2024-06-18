from torch.utils.data import Dataset

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticDataset(Dataset):
    def __init__(self, states, actions, target_values):
        self.states = states
        self.actions = actions
        self.target_values = target_values.reshape(-1, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        target_value = self.target_values[idx]
        return state, action, target_value
