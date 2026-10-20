from huggingface_hub import HfApi

api = HfApi()

# Create your new model repository (change repo_id to your username/repo-name)
api.create_repo(
    repo_id="pratheeks/deepseek-moe-16b-chat-GGUF",
    repo_type="model",
    private=False
)

# Upload the GGUF file
api.upload_file(
    path_or_fileobj="./deepseek-moe-16b-chat-bf16.gguf",
    path_in_repo="deepseek-moe-16b-chat-bf16.gguf",
    repo_id="pratheeks/deepseek-moe-16b-chat-GGUF",
    repo_type="model"
)
