import torch
import typer
import engine
import load_data
import model_builder
import utils

'''
uso:
python train.py "epochs" "train_dir" "test_dir"
'''

app = typer.Typer()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"using {device}")

SAVE_PATH = "models/"
MODEL_NAME = "model.pth"


@app.command()
def start_training(epochs_value: int, train_dir: str, test_dir: str, save_path=SAVE_PATH, model_name=MODEL_NAME):

    loaded_train, loaded_test, class_names = load_data.load(train_dir, test_dir, batch_size=32, device=device)

    model = model_builder.GenderModel(in_size=3,
                                      hidden_units=20,
                                      out_size=1
                                      ).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05)

    results = engine.train(epochs_value,
                           model,
                           optimizer,
                           loss_fn,
                           device=device,
                           loaded_train=loaded_train,
                           loaded_test=loaded_test,
                           classes=class_names
                           )

    utils.save_model(model, results, save_path, model_name)


if __name__ == "__main__":
    app()
