from src import *
# Press the green button in the gutter to run the script.

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_model(path):
  model = ViViT(256, 16, 49, 'cuda:0')
  if os.path.isfile(path) == True:
    model.load_state_dict(torch.load(path))
  return model

if __name__ == '__main__':
    video_procesing = VideoProcesing()
    x = video_procesing.get_item(path=r"C:\Users\Admin\Desktop\APP\sign_language_recognize\File 9_noaudio.mp4")
    print(x)
    model = load_model(r"./src/file_model/model_2.bin")
    model.device=DEVICE
    print(model(x))
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
