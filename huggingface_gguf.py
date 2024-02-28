from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem, HfApi
import inquirer

model_input = input('Huggingface uri / path >> ')

if '.co' in model_input:
    uri_end = model_input.index('.co')
    model_input = model_input[(uri_end + 2):]

model_id=model_input

fs = HfFileSystem()
model_files = fs.glob(model_id + '/*.gguf')
if model_files == '':
    print('no gguf files found in repo')
    exit()
else:
    print('\nfound the following files in the repo:\n')

model_gguf_files = []
for file in model_files:
    model_gguf_files.append(file[(len(model_id) + 1):])

gguf_files = [
    inquirer.List(
        "file",
        message="What file to download?",
        choices=model_gguf_files,
    ),
]

download_file = inquirer.prompt(gguf_files)
download_file = download_file['file']

#user_confirm = input('Download model: "' + model_id + '"? ')
user_confirm = input('Download file: "' + download_file + '"? ')

if user_confirm == 'y' or user_confirm == 'yes':
    hf_hub_download(repo_id=model_id, local_dir="./models/",
                      local_dir_use_symlinks=False, revision="main",
                      filename=download_file)
