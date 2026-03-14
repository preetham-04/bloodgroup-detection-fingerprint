import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from resnet9_model import ResNet9

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(3, 8)
model.load_state_dict(torch.load("resnet9_bloodgroup.pth", map_location=device))
model.eval()

classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🔬 Blood Group Detection from Fingerprint")

uploaded_file = st.file_uploader("Upload a fingerprint image", type=['jpg','jpeg','png','bmp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Fingerprint", width=200)

    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(probs, 1)
        st.success(f"🩸 Predicted Blood Group: {classes[pred.item()]}")
        st.write(f"Confidence: {probs[0][pred.item()]*100:.2f}%")
