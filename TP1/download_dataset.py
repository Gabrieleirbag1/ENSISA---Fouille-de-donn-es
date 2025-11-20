import kagglehub

# Download latest version
path = kagglehub.dataset_download("datasnaek/league-of-legends")

print("Path to dataset files:", path)