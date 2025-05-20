import gradio as gr
from fastai.vision.all import *

learn = load_learner('xray_model.pkl') 

def predict_xray(img):
    pred, pred_idx, probs = learn.predict(PILImage.create(img))
    return f"{pred} ({probs[pred_idx]*100:.2f}%)"

demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Zatürre Tespiti",
    description="Göğüs röntgenini yükleyin. Model, PNEUMONIA veya NORMAL olarak sınıflandıracaktır."
)

demo.launch(share=True)
