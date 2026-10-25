"""Upload a query and return five exact-L2 matches from the test gallery."""
import argparse
import numpy as np
import torch
import gradio as gr
from model import EmbeddingModel
from data import transform
from retrieval import L2Index

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',default='runs/baseline')
    a = p.parse_args()
    ck = torch.load(a.run+'/best.pt',map_location='cpu',weights_only=True)
    model = EmbeddingModel(ck['dim']); model.load_state_dict(ck['model']); model.eval()
    data = np.load(a.run+'/gallery.npz',allow_pickle=False)
    index = L2Index(data['embeddings'])
    @torch.no_grad()
    def search(image):
        if image is None:
            return []
        image = image.convert('RGB').resize((32,32))
        z = model(transform()(image).unsqueeze(0)).numpy()
        ids,d = index.search(z,5)
        return [(data['images'][j],f"{data['classes'][data['labels'][j]]} | cosine={1-dist/2:.4f} | L2={np.sqrt(dist):.4f}") for j,dist in zip(ids[0],d[0])]
    with gr.Blocks() as demo:
        gr.Markdown('# Visual similarity search\nRandom-initialization ResNet-18 trained with triplet learning. Upload an image to search the CIFAR-100 test gallery.')
        q = gr.Image(type='pil',label='Query image')
        results = gr.Gallery(label='Top five matches',columns=5)
        gr.Button('Search').click(search,q,results)
    demo.launch(share=False)

if __name__ == '__main__':
    main()
