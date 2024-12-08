from flask import Flask, jsonify, request, send_file
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

class StableDiffusionProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = self.scheduler
        
        self.num_inference_steps = 100
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.timesteps = self.scheduler.timesteps
        
        self.current_latent = None
        self.initial_latent = None
        self.original_noise = None
        self.current_timestep_index = 0

    def process_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return preprocess(image).unsqueeze(0)

    def initialize_from_image(self, image: Image.Image):
        # image is now a PIL Image directly, not base64
        image_tensor = self.process_image(image)
        image_tensor = image_tensor.to(self.device, dtype=self.pipeline.unet.dtype)

        with torch.no_grad():
            self.current_latent = self.pipeline.vae.encode(image_tensor).latent_dist.sample() * 0.18215
            self.current_latent = self.current_latent.detach()
            self.initial_latent = self.current_latent.clone()
            self.original_noise = torch.randn_like(self.current_latent)
            self.current_timestep_index = 0

    def add_noise(self, steps):
        max_steps = len(self.scheduler.timesteps)
        t_index = max(0, max_steps - steps)
        t = self.scheduler.timesteps[t_index]
        
        noisy_latent = self.scheduler.add_noise(self.current_latent, self.original_noise, t)
        
        with torch.no_grad():
            decoded = self.pipeline.vae.decode(noisy_latent / 0.18215).sample
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))
        
        self.current_latent = noisy_latent.detach()
        self.current_timestep_index = t_index
        return pil_image

    def denoise(self, steps, prompt_word=""):
        if steps < 1:
            steps = 1

        noisy_latent = self.current_latent.clone().detach()
        timesteps_to_use = self.scheduler.timesteps[self.current_timestep_index:self.current_timestep_index + steps]

        text_input = self.pipeline.tokenizer(
            prompt_word,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_input)[0]
            
            for t in timesteps_to_use:
                model_output = self.pipeline.unet(
                    noisy_latent, t, encoder_hidden_states=encoder_hidden_states
                ).sample
                
                step_output = self.pipeline.scheduler.step(
                    model_output, t, noisy_latent
                )
                noisy_latent = step_output.prev_sample

            interpolation_factor = 0.3
            noisy_latent = (1 - interpolation_factor) * noisy_latent + interpolation_factor * self.initial_latent

            decoded = self.pipeline.vae.decode(noisy_latent / 0.18215).sample
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))

        self.current_latent = noisy_latent.detach()
        self.current_timestep_index += steps
        return pil_image

sd_processor = StableDiffusionProcessor()

@app.route('/api/process', methods=['POST'])
def process_data():
    try:
        # Now we expect form data with a file upload
        operation_type = request.form.get('type')
        steps = int(request.form.get('steps', 1))
        prompt_word = request.form.get('prompt_word', '')
        image_file = request.files.get('original_image') 

        if operation_type not in ['noise', 'denoise']:
            return jsonify({"error": "Invalid operation type"}), 400

        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        # Convert the uploaded file to a PIL Image
        image = Image.open(image_file.stream).convert("RGB")
        sd_processor.initialize_from_image(image)

        if operation_type == 'noise':
            adjusted_steps = max(1, steps // 2)
            processed_image = sd_processor.add_noise(adjusted_steps)
        else:  # denoise
            adjusted_steps = steps * 4
            processed_image = sd_processor.denoise(adjusted_steps, prompt_word)

        # Convert PIL Image to binary and send as response
        img_io = BytesIO()
        processed_image.save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
